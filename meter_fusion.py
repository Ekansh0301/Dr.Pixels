import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm
from transformers import (
    ViltProcessor, AutoModel, AutoImageProcessor,
    ViTModel, ViTConfig, logging as transformers_logging
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Optional
import math

transformers_logging.set_verbosity_error()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

CONFIG = {
    "vision_model_name": "google/vit-base-patch16-224",
    "text_model_name": "roberta-base",
    "clinical_bert_name": "emilyalsentzer/Bio_ClinicalBERT",
    "radbert_name": "zzxslp/RadBERT-RoBERTa-4m",
    
    "baseline_checkpoint": os.path.join(BASE_DIR, "vqa_vilt_baseline_outputs", "checkpoints", "vilt_baseline_best.pth"),
    "clinical_checkpoint": os.path.join(BASE_DIR, "vqa_vilt_clinicalbert_outputs", "checkpoints", "vilt_clinicalbert_best.pth"),
    "radbert_checkpoint": os.path.join(BASE_DIR, "vqa_vilt_radbert_outputs", "checkpoints", "vilt_radbert_best.pth"),
    
    "data_dir": BASE_DIR,
    "checkpoint_dir": os.path.join(BASE_DIR, "vqa_meter_outputs", "checkpoints"),
    "checkpoint_path": "meter_specialist_best.pth",
    "results_dir": os.path.join(BASE_DIR, "vqa_meter_outputs", "results"),
    "vocab_path": os.path.join(BASE_DIR, "vqa_vilt_baseline_outputs", "checkpoints", "answer_vocab.json"),
    "cache_dir": os.path.join(BASE_DIR, "model_cache"),
    
    "text_encoder_type": "radbert",
    "num_fusion_layers": 6,
    "fusion_heads": 12,
    "fusion_hidden_dim": 768,
    "fusion_dropout": 0.1,
    "freeze_vision": True,
    "freeze_text_encoder": False,
    
    "epochs": 25,
    "lr": 2e-5,
    "fusion_lr": 5e-4,
    "batch_size": 24,
    "use_mixed_precision": True,
    "num_workers": 8,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "pin_memory": True,
    "gradient_accumulation_steps": 2,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "patience": 5,
    "seed": 42,
    "enable_tf32": True,
    "use_torch_compile": False,
}

class CoAttentionLayer(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        
        self.vision_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.text_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.vision_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.text_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.vision_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.ln_v_self = nn.LayerNorm(hidden_dim)
        self.ln_t_self = nn.LayerNorm(hidden_dim)
        self.ln_v_cross = nn.LayerNorm(hidden_dim)
        self.ln_t_cross = nn.LayerNorm(hidden_dim)
        self.ln_v_ffn = nn.LayerNorm(hidden_dim)
        self.ln_t_ffn = nn.LayerNorm(hidden_dim)
    
    def forward(self, vision_feats, text_feats):
        v_self, _ = self.vision_self_attn(vision_feats, vision_feats, vision_feats)
        vision_feats = self.ln_v_self(vision_feats + v_self)
        
        t_self, _ = self.text_self_attn(text_feats, text_feats, text_feats)
        text_feats = self.ln_t_self(text_feats + t_self)
        
        v_cross, _ = self.vision_cross_attn(
            query=vision_feats,
            key=text_feats,
            value=text_feats
        )
        vision_feats = self.ln_v_cross(vision_feats + v_cross)
        
        t_cross, _ = self.text_cross_attn(
            query=text_feats,
            key=vision_feats,
            value=vision_feats
        )
        text_feats = self.ln_t_cross(text_feats + t_cross)
        
        v_ffn = self.vision_ffn(vision_feats)
        vision_feats = self.ln_v_ffn(vision_feats + v_ffn)
        
        t_ffn = self.text_ffn(text_feats)
        text_feats = self.ln_t_ffn(text_feats + t_ffn)
        
        return vision_feats, text_feats

class METERFusionModule(nn.Module):
    def __init__(self, num_layers=6, hidden_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CoAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, vision_feats, text_feats):
        for layer in self.layers:
            vision_feats, text_feats = layer(vision_feats, text_feats)
        return vision_feats, text_feats

class METERWithSpecialistEncoder(nn.Module):
    def __init__(
        self,
        vision_encoder,
        text_encoder,
        num_labels,
        num_fusion_layers=6,
        fusion_heads=12,
        fusion_dropout=0.1,
        freeze_vision=True,
        freeze_text=False,
        text_encoder_type="radbert"
    ):
        super().__init__()
        
        self.text_encoder_type = text_encoder_type
        
        self.vision_encoder = vision_encoder
        
        self.text_embeddings = text_encoder.embeddings
        self.text_encoder_layers = text_encoder.encoder
        
        self.fusion = METERFusionModule(
            num_layers=num_fusion_layers,
            hidden_dim=768,
            num_heads=fusion_heads,
            dropout=fusion_dropout
        )
        
        self.vision_pooler = nn.Linear(768, 768)
        self.text_pooler = nn.Linear(768, 768)
        
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, num_labels)
        )
        
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print("  ✓ Vision encoder frozen")
        
        if freeze_text:
            for param in self.text_embeddings.parameters():
                param.requires_grad = False
            for param in self.text_encoder_layers.parameters():
                param.requires_grad = False
            print("  ✓ Text encoder frozen")
        
        print(f"  ✓ Text encoder: {text_encoder_type}")
        print(f"  ✓ Fusion layers: {num_fusion_layers}")
    
    def forward(self, pixel_values, input_ids, attention_mask, token_type_ids=None):
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_feats = vision_outputs.last_hidden_state
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        text_embeds = self.text_embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=text_embeds.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(text_embeds.dtype).min
        
        text_outputs = self.text_encoder_layers(
            text_embeds,
            attention_mask=extended_attention_mask,
            return_dict=True
        )
        text_feats = text_outputs.last_hidden_state
        
        fused_vision, fused_text = self.fusion(vision_feats, text_feats)
        
        vision_pooled = self.vision_pooler(fused_vision.mean(dim=1))
        
        text_pooled = self.text_pooler(fused_text[:, 0, :])
        
        combined = torch.cat([vision_pooled, text_pooled], dim=-1)
        logits = self.classifier(combined)
        
        return logits

class VQAMedDatasetMETER(Dataset):
    def __init__(self, data_dir, split, image_processor, text_tokenizer, 
                 answer_to_label=None, num_labels=None, min_answer_freq=2, augment=False):
        self.image_processor = image_processor
        self.text_tokenizer = text_tokenizer
        self.split = split
        self.augment = augment and split == 'train'
        
        if split == 'train':
            self.qa_file = os.path.join(data_dir, 'ImageClef-2019-VQA-Med-Training', 'All_QA_Pairs_train.txt')
            self.image_dir = os.path.join(data_dir, 'ImageClef-2019-VQA-Med-Training', 'Train_images')
        elif split == 'val':
            self.qa_file = os.path.join(data_dir, 'ImageClef-2019-VQA-Med-Validation', 'All_QA_Pairs_val.txt')
            self.image_dir = os.path.join(data_dir, 'ImageClef-2019-VQA-Med-Validation', 'Val_images')
        elif split == 'test':
            self.qa_file = os.path.join(data_dir, 'VQAMed2019Test', 'VQAMed2019_Test_Questions_w_Ref_Answers.txt')
            self.image_dir = os.path.join(data_dir, 'VQAMed2019Test', 'VQAMed2019_Test_Images')
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.qa_pairs = self._load_qa_pairs()
        
        if answer_to_label is None:
            self.answer_to_label, self.label_to_answer, self.num_labels = self._build_answer_vocab(min_answer_freq)
        else:
            self.answer_to_label = answer_to_label
            self.label_to_answer = {i: answer for answer, i in self.answer_to_label.items()}
            self.num_labels = num_labels
        
        self.qa_pairs = [qa for qa in self.qa_pairs if qa['answer'] in self.answer_to_label]
        print(f"[{split.upper()}] Loaded {len(self.qa_pairs)} QA pairs")
    
    def _load_qa_pairs(self):
        qa_pairs = []
        with open(self.qa_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    parts = line.strip().split('|')
                    if len(parts) >= 3:
                        img_id = parts[0]
                        question = parts[-2] if len(parts) == 4 else parts[1]
                        answer = parts[-1]
                        answer = answer.lower().strip()
                        question = question.strip()
                        image_path = os.path.join(self.image_dir, f"{img_id}.jpg")
                        
                        if os.path.exists(image_path):
                            qa_pairs.append({
                                "image_path": image_path,
                                "question": question,
                                "answer": answer,
                                "image_id": img_id
                            })
                except:
                    continue
        return qa_pairs
    
    def _build_answer_vocab(self, min_freq):
        all_answers = [pair['answer'] for pair in self.qa_pairs]
        answer_counts = Counter(all_answers)
        filtered_answers = [ans for ans, count in answer_counts.items() if count >= min_freq]
        filtered_answers = sorted(filtered_answers)
        
        answer_to_label = {answer: i for i, answer in enumerate(filtered_answers)}
        label_to_answer = {i: answer for answer, i in answer_to_label.items()}
        return answer_to_label, label_to_answer, len(filtered_answers)
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def _augment_image(self, image):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.3:
            from PIL import ImageEnhance
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            from PIL import ImageEnhance
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        return image
    
    def __getitem__(self, idx):
        try:
            sample = self.qa_pairs[idx]
            
            image = Image.open(sample['image_path']).convert('RGB')
            
            if self.augment:
                image = self._augment_image(image)
            
            image_encoding = self.image_processor(images=image, return_tensors="pt")
            
            text_encoding = self.text_tokenizer(
                sample['question'],
                padding="max_length",
                truncation=True,
                max_length=40,
                return_tensors="pt"
            )
            
            result = {
                'pixel_values': image_encoding['pixel_values'].squeeze(0),
                'input_ids': text_encoding['input_ids'].squeeze(0),
                'attention_mask': text_encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(self.answer_to_label[sample['answer']], dtype=torch.long),
                'question_text': sample['question'],
                'image_id': sample['image_id']
            }
            
            if 'token_type_ids' in text_encoding:
                result['token_type_ids'] = text_encoding['token_type_ids'].squeeze(0)
            
            return result
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__(0)

def custom_collate_fn(batch):
    question_texts = [item.pop('question_text') for item in batch]
    image_ids = [item.pop('image_id') for item in batch]
    
    keys = batch[0].keys()
    collated = {key: torch.stack([item[key] for item in batch]) for key in keys}
    collated['question_texts'] = question_texts
    collated['image_ids'] = image_ids
    return collated

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", position=0, leave=True)
    
    for step, batch in enumerate(progress_bar):
        labels = batch.pop('labels').to(device)
        question_texts = batch.pop('question_texts')
        image_ids = batch.pop('image_ids')
        
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config["use_mixed_precision"]):
            logits = model(pixel_values, input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels) / config["gradient_accumulation_steps"]
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config["gradient_accumulation_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config["gradient_accumulation_steps"]
        correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f"{loss.item()*config['gradient_accumulation_steps']:.4f}",
            'acc': f"{correct/total:.4f}"
        })
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", position=0, leave=True):
            labels = batch.pop('labels').to(device)
            question_texts = batch.pop('question_texts')
            image_ids = batch.pop('image_ids')
            
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=CONFIG["use_mixed_precision"]):
                logits = model(pixel_values, input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        'correct': correct,
        'total': total,
        'predictions': all_predictions,
        'labels': all_labels
    }

def create_comparison_plot(vilt_results, meter_results, save_path):
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    architectures = ['ViLT', 'METER']
    baseline_accs = [
        vilt_results['baseline'],
        meter_results['baseline']
    ]
    clinical_accs = [
        vilt_results['clinical'],
        meter_results['clinical']
    ]
    radbert_accs = [
        vilt_results['radbert'],
        meter_results['radbert']
    ]
    
    x = np.arange(len(architectures))
    width = 0.25
    
    bars1 = ax.bar(x - width, baseline_accs, width, label='Baseline BERT', 
                   color='#95a5a6', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, clinical_accs, width, label='ClinicalBERT',
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, radbert_accs, width, label='RadBERT',
                   color='#3498db', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Architecture', fontsize=14, fontweight='bold')
    ax.set_title('Generalizability of Linguistic Specialization\nAcross Architectures', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(architectures, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for i, arch in enumerate(architectures):
        clinical_gain = clinical_accs[i] - baseline_accs[i]
        radbert_gain = radbert_accs[i] - baseline_accs[i]
        
        ax.annotate(f'+{clinical_gain:.1%}', 
                   xy=(i, clinical_accs[i]), xytext=(i-width, clinical_accs[i] + 0.03),
                   ha='center', fontsize=9, color='#c0392b', fontweight='bold')
        
        ax.annotate(f'+{radbert_gain:.1%}',
                   xy=(i, radbert_accs[i]), xytext=(i+width, radbert_accs[i] + 0.03),
                   ha='center', fontsize=9, color='#2980b9', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Comparison plot saved to {save_path}")

def plot_final_training_curves(history, save_dir, model_name):
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    ax = axes[0]
    ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    ax = axes[1]
    ax.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'{model_name}_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Training curves saved to {plot_path}")

def main():
    set_seed(CONFIG['seed'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if CONFIG.get('enable_tf32', True) and device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for RTX 4080 Super acceleration")
    
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    os.makedirs(CONFIG['cache_dir'], exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"METER-Style Architecture with Specialized Encoder - RTX 4080 Super Optimized")
    print(f"{'='*70}\n")
    print(f"Text encoder: {CONFIG['text_encoder_type']}")
    print(f"Vision encoder: ViT (frozen: {CONFIG['freeze_vision']})")
    print(f"Fusion layers: {CONFIG['num_fusion_layers']}")
    print(f"Base directory: {BASE_DIR}")
    print(f"\nOptimizations:")
    print(f"  - Batch size: {CONFIG['batch_size']}")
    print(f"  - Mixed precision: {CONFIG['use_mixed_precision']}")
    print(f"  - Workers: {CONFIG['num_workers']}\n")
    
    with open(CONFIG['vocab_path'], 'r') as f:
        vocab_data = json.load(f)
        answer_to_label = vocab_data['answer_to_label']
        label_to_answer = {int(k): v for k, v in vocab_data['label_to_answer'].items()}
        num_labels = vocab_data['num_labels']
    
    print("Loading processors...")
    
    image_processor = AutoImageProcessor.from_pretrained(
        CONFIG['vision_model_name'],
        cache_dir=CONFIG['cache_dir']
    )
    
    if CONFIG['text_encoder_type'] == "clinical":
        from transformers import AutoTokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(
            CONFIG['clinical_bert_name'],
            cache_dir=CONFIG['cache_dir']
        )
    elif CONFIG['text_encoder_type'] == "radbert":
        from transformers import AutoTokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(
            CONFIG['radbert_name'],
            cache_dir=CONFIG['cache_dir']
        )
    else:
        from transformers import AutoTokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(
            CONFIG['text_model_name'],
            cache_dir=CONFIG['cache_dir']
        )
    
    print("Loading datasets...")
    
    train_dataset = VQAMedDatasetMETER(
        CONFIG['data_dir'], 'train', image_processor, text_tokenizer,
        answer_to_label, num_labels, augment=True
    )
    val_dataset = VQAMedDatasetMETER(
        CONFIG['data_dir'], 'val', image_processor, text_tokenizer,
        answer_to_label, num_labels
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=CONFIG['num_workers'], 
        pin_memory=CONFIG.get('pin_memory', True),
        prefetch_factor=CONFIG.get('prefetch_factor', 2),
        persistent_workers=CONFIG.get('persistent_workers', True),
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,
        num_workers=CONFIG['num_workers'], 
        pin_memory=CONFIG.get('pin_memory', True),
        prefetch_factor=CONFIG.get('prefetch_factor', 2),
        persistent_workers=CONFIG.get('persistent_workers', True),
        collate_fn=custom_collate_fn
    )
    
    print("Loading encoders...")
    
    vision_encoder = ViTModel.from_pretrained(
        CONFIG['vision_model_name'],
        cache_dir=CONFIG['cache_dir']
    )
    
    if CONFIG['text_encoder_type'] == "clinical":
        text_encoder = AutoModel.from_pretrained(
            CONFIG['clinical_bert_name'],
            cache_dir=CONFIG['cache_dir']
        )
    elif CONFIG['text_encoder_type'] == "radbert":
        text_encoder = AutoModel.from_pretrained(
            CONFIG['radbert_name'],
            cache_dir=CONFIG['cache_dir']
        )
    else:
        text_encoder = AutoModel.from_pretrained(
            CONFIG['text_model_name'],
            cache_dir=CONFIG['cache_dir']
        )
    
    print("  Creating METER-style model...")
    
    model = METERWithSpecialistEncoder(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        num_labels=num_labels,
        num_fusion_layers=CONFIG['num_fusion_layers'],
        fusion_heads=CONFIG['fusion_heads'],
        fusion_dropout=CONFIG['fusion_dropout'],
        freeze_vision=CONFIG['freeze_vision'],
        freeze_text=CONFIG['freeze_text_encoder'],
        text_encoder_type=CONFIG['text_encoder_type']
    ).to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen: {total_params - trainable:,}\n")
    
    if CONFIG.get('use_torch_compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("  ✓ Model compiled with torch.compile")
    
    if CONFIG['freeze_vision'] and CONFIG['freeze_text_encoder']:
        optimizer = AdamW([
            {'params': model.fusion.parameters(), 'lr': CONFIG['fusion_lr']},
            {'params': model.vision_pooler.parameters(), 'lr': CONFIG['fusion_lr']},
            {'params': model.text_pooler.parameters(), 'lr': CONFIG['fusion_lr']},
            {'params': model.classifier.parameters(), 'lr': CONFIG['fusion_lr']}
        ], weight_decay=CONFIG['weight_decay'])
    else:
        optimizer = AdamW([
            {'params': model.vision_encoder.parameters(), 'lr': CONFIG['lr']},
            {'params': model.text_embeddings.parameters(), 'lr': CONFIG['lr']},
            {'params': model.text_encoder_layers.parameters(), 'lr': CONFIG['lr']},
            {'params': model.fusion.parameters(), 'lr': CONFIG['fusion_lr']},
            {'params': model.classifier.parameters(), 'lr': CONFIG['fusion_lr']}
        ], weight_decay=CONFIG['weight_decay'])
    
    scaler = torch.amp.GradScaler('cuda', enabled=CONFIG["use_mixed_precision"])
    
    best_accuracy = 0.0
    patience_counter = 0
    history = []
    
    print(f"{'='*60}\nStarting Training\n{'='*60}\n")
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{CONFIG['epochs']}\n{'='*60}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_dataloader, optimizer, scaler, device, epoch, CONFIG
        )
        eval_results = evaluate(model, val_dataloader, device)
        
        val_acc = eval_results['accuracy']
        val_loss = eval_results['loss']
        
        print(f"\nResults:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        history.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience_counter = 0
            
            checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], CONFIG['checkpoint_path'])
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_accuracy': best_accuracy,
                'history': history,
                'config': CONFIG,
                'vocab': {'answer_to_label': answer_to_label, 'label_to_answer': label_to_answer}
            }, checkpoint_path)
            
            print(f"\n  ✓ NEW BEST MODEL!")
            print(f"    Accuracy: {val_acc:.4f}")
            print(f"    Saved to: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"\n  Patience: {patience_counter}/{CONFIG['patience']}")
        
        if (epoch + 1) % 5 == 0:
            user_input = input(f"\nEpoch {epoch+1} complete. Continue training? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Training stopped by user.")
                break
        
        if patience_counter >= CONFIG['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    with open(os.path.join(CONFIG['results_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    summary = {
        'model': f'METER-{CONFIG["text_encoder_type"]}',
        'architecture': 'METER-style',
        'text_encoder': CONFIG['text_encoder_type'],
        'best_val_accuracy': best_accuracy,
        'total_epochs': len(history),
        'config': CONFIG
    }
    
    with open(os.path.join(CONFIG['results_dir'], 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    plot_final_training_curves(history, CONFIG['results_dir'], f'METER_{CONFIG["text_encoder_type"]}')
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Architecture: METER-style")
    print(f"Text Encoder: {CONFIG['text_encoder_type']}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Results directory: {CONFIG['results_dir']}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()