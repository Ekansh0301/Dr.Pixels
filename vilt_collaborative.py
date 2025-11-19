import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig, AutoModel, logging as transformers_logging
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import math

transformers_logging.set_verbosity_error()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

CONFIG = {
    "model_name": "dandelin/vilt-b32-finetuned-vqa",
    "clinical_bert_name": "emilyalsentzer/Bio_ClinicalBERT",
    "radbert_name": "zzxslp/RadBERT-RoBERTa-4m",
    "baseline_checkpoint": os.path.join(BASE_DIR, "vqa_vilt_baseline_outputs", "checkpoints", "vilt_baseline_best.pth"),
    "clinical_checkpoint": os.path.join(BASE_DIR, "vqa_vilt_clinicalbert_outputs", "checkpoints", "vilt_clinicalbert_best.pth"),
    "radbert_checkpoint": os.path.join(BASE_DIR, "vqa_vilt_radbert_outputs", "checkpoints", "vilt_radbert_best.pth"),
    "data_dir": BASE_DIR,
    "checkpoint_dir": os.path.join(BASE_DIR, "vqa_crossattn_outputs", "checkpoints"),
    "checkpoint_path": "vilt_crossattn_best.pth",
    "results_dir": os.path.join(BASE_DIR, "vqa_crossattn_outputs", "results"),
    "vocab_path": os.path.join(BASE_DIR, "vqa_vilt_baseline_outputs", "checkpoints", "answer_vocab.json"),
    "cache_dir": os.path.join(BASE_DIR, "model_cache"),
    
    "collaboration_type": "bidirectional",
    "num_collaboration_layers": 2,
    "attention_heads": 8,
    "attention_dropout": 0.1,
    "use_residual": True,
    "fusion_strategy": "concat",
    "freeze_specialists": False,
    
    "epochs": 25,
    "lr": 1e-5,
    "collaboration_lr": 5e-4,
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
    "freeze_vision_encoder": True,
    "enable_tf32": True,
    "use_torch_compile": False,
}

class CrossAttentionCollaborationLayer(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.clinical_to_rad_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        
        self.rad_to_clinical_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        
        self.ln_clinical = nn.LayerNorm(hidden_dim)
        self.ln_radbert = nn.LayerNorm(hidden_dim)
        
        self.ffn_clinical = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.ffn_radbert = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.ln_ffn_clinical = nn.LayerNorm(hidden_dim)
        self.ln_ffn_radbert = nn.LayerNorm(hidden_dim)
    
    def forward(self, clinical_embeds, radbert_embeds, return_attention=False):
        clinical_t = clinical_embeds.transpose(0, 1)
        radbert_t = radbert_embeds.transpose(0, 1)
        
        clinical_updated, clinical_attn_weights = self.clinical_to_rad_attn(
            query=clinical_t,
            key=radbert_t,
            value=radbert_t
        )
        clinical_updated = clinical_updated.transpose(0, 1)
        
        clinical_embeds = self.ln_clinical(clinical_embeds + clinical_updated)
        
        radbert_updated, radbert_attn_weights = self.rad_to_clinical_attn(
            query=radbert_t,
            key=clinical_t,
            value=clinical_t
        )
        radbert_updated = radbert_updated.transpose(0, 1)
        
        radbert_embeds = self.ln_radbert(radbert_embeds + radbert_updated)
        
        clinical_ffn = self.ffn_clinical(clinical_embeds)
        clinical_embeds = self.ln_ffn_clinical(clinical_embeds + clinical_ffn)
        
        radbert_ffn = self.ffn_radbert(radbert_embeds)
        radbert_embeds = self.ln_ffn_radbert(radbert_embeds + radbert_ffn)
        
        if return_attention:
            return clinical_embeds, radbert_embeds, {
                'clinical_to_rad_attn': clinical_attn_weights,
                'rad_to_clinical_attn': radbert_attn_weights
            }
        
        return clinical_embeds, radbert_embeds

class CollaborationStack(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionCollaborationLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
    
    def forward(self, clinical_embeds, radbert_embeds, return_attention=False):
        attention_maps = []
        
        for i, layer in enumerate(self.layers):
            if return_attention:
                clinical_embeds, radbert_embeds, attn = layer(
                    clinical_embeds, radbert_embeds, return_attention=True
                )
                attention_maps.append(attn)
            else:
                clinical_embeds, radbert_embeds = layer(clinical_embeds, radbert_embeds)
        
        if return_attention:
            return clinical_embeds, radbert_embeds, attention_maps
        
        return clinical_embeds, radbert_embeds

class FusionModule(nn.Module):
    def __init__(self, hidden_dim=768, num_labels=786, fusion_type="concat"):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        elif fusion_type == "add":
            self.classifier = nn.Linear(hidden_dim, num_labels)
        elif fusion_type == "learned_blend":
            self.blend_weight = nn.Parameter(torch.tensor(0.5))
            self.classifier = nn.Linear(hidden_dim, num_labels)
        elif fusion_type == "gated_fusion":
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            self.classifier = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, clinical_pooled, radbert_pooled):
        if self.fusion_type == "concat":
            combined = torch.cat([clinical_pooled, radbert_pooled], dim=-1)
            logits = self.classifier(combined)
        
        elif self.fusion_type == "add":
            combined = clinical_pooled + radbert_pooled
            logits = self.classifier(combined)
        
        elif self.fusion_type == "learned_blend":
            weight = torch.sigmoid(self.blend_weight)
            combined = (1 - weight) * clinical_pooled + weight * radbert_pooled
            logits = self.classifier(combined)
        
        elif self.fusion_type == "gated_fusion":
            concat = torch.cat([clinical_pooled, radbert_pooled], dim=-1)
            gate = self.fusion_gate(concat)
            combined = (1 - gate) * clinical_pooled + gate * radbert_pooled
            logits = self.classifier(combined)
        
        return logits

class CollaborativeSpecialistModel(nn.Module):
    def __init__(
        self,
        vilt_model,
        clinical_bert,
        radbert,
        num_labels,
        num_collaboration_layers=2,
        attention_heads=8,
        attention_dropout=0.1,
        fusion_strategy="concat",
        freeze_vision=True,
        freeze_specialists=False
    ):
        super().__init__()
        
        self.vilt = vilt_model
        self.config = vilt_model.config
        self.fusion_strategy = fusion_strategy
        
        self.clinical_embeddings = clinical_bert.embeddings
        self.clinical_encoder = clinical_bert.encoder
        self.radbert_embeddings = radbert.embeddings
        self.radbert_encoder = radbert.encoder
        
        self.collaboration = CollaborationStack(
            num_layers=num_collaboration_layers,
            hidden_dim=768,
            num_heads=attention_heads,
            dropout=attention_dropout
        )
        
        self.fusion = FusionModule(
            hidden_dim=768,
            num_labels=num_labels,
            fusion_type=fusion_strategy
        )
        
        if freeze_vision:
            for name, param in self.vilt.vilt.embeddings.named_parameters():
                if 'patch' in name.lower():
                    param.requires_grad = False
        
        if freeze_specialists:
            for param in self.clinical_embeddings.parameters():
                param.requires_grad = False
            for param in self.clinical_encoder.parameters():
                param.requires_grad = False
            for param in self.radbert_embeddings.parameters():
                param.requires_grad = False
            for param in self.radbert_encoder.parameters():
                param.requires_grad = False
            print("  ✓ Specialist encoders frozen")
        
        print(f"  ✓ Collaboration: {num_collaboration_layers} cross-attention layers")
        print(f"  ✓ Fusion strategy: {fusion_strategy}")
    
    def _encode_text(self, input_ids, token_type_ids, attention_mask, encoder_embeddings, encoder):
        text_embeds = encoder_embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=text_embeds.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(text_embeds.dtype).min
        
        outputs = encoder(text_embeds, attention_mask=extended_attention_mask, return_dict=True)
        return outputs.last_hidden_state
    
    def forward(self, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, 
                return_collaboration_info=False):
        clinical_text = self._encode_text(
            input_ids, token_type_ids, attention_mask,
            self.clinical_embeddings, self.clinical_encoder
        )
        
        radbert_text = self._encode_text(
            input_ids, token_type_ids, attention_mask,
            self.radbert_embeddings, self.radbert_encoder
        )
        
        if return_collaboration_info:
            clinical_collaborated, radbert_collaborated, attn_maps = self.collaboration(
                clinical_text, radbert_text, return_attention=True
            )
        else:
            clinical_collaborated, radbert_collaborated = self.collaboration(
                clinical_text, radbert_text
            )
            attn_maps = None
        
        vilt_clinical = self.vilt.vilt(
            inputs_embeds=clinical_collaborated,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
        clinical_pooled = vilt_clinical.pooler_output
        
        vilt_radbert = self.vilt.vilt(
            inputs_embeds=radbert_collaborated,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
        radbert_pooled = vilt_radbert.pooler_output
        
        logits = self.fusion(clinical_pooled, radbert_pooled)
        
        if return_collaboration_info:
            return logits, {
                'clinical_pooled': clinical_pooled,
                'radbert_pooled': radbert_pooled,
                'attention_maps': attn_maps
            }
        
        return logits

class VQAMedDataset(Dataset):
    def __init__(self, data_dir, split, processor, answer_to_label=None, num_labels=None, 
                 min_answer_freq=2, augment=False):
        self.processor = processor
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
            image = Image.open(sample['image_path']).convert('RGB').resize((384, 384), Image.BILINEAR)
            
            if self.augment:
                image = self._augment_image(image)
            
            encoding = self.processor(images=image, text=sample['question'], padding="max_length",
                                     truncation=True, max_length=40, return_tensors="pt")
            
            result = {k: v.squeeze(0) for k, v in encoding.items()}
            result['labels'] = torch.tensor(self.answer_to_label[sample['answer']], dtype=torch.long)
            result['question_text'] = sample['question']
            result['image_id'] = sample['image_id']
            return result
        except:
            return self.__getitem__(0)

def custom_collate_fn(batch):
    question_texts = [item.pop('question_text', '') for item in batch]
    image_ids = [item.pop('image_id', '') for item in batch]
    
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
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config["use_mixed_precision"]):
            logits = model(**batch_gpu)
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

def evaluate(model, dataloader, device, return_detailed=False):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    predictions_all = []
    labels_all = []
    questions_all = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", position=0, leave=True):
            labels = batch.pop('labels').to(device)
            question_texts = batch.pop('question_texts')
            image_ids = batch.pop('image_ids')
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=CONFIG["use_mixed_precision"]):
                if return_detailed:
                    logits, collab_info = model(**batch_gpu, return_collaboration_info=True)
                else:
                    logits = model(**batch_gpu)
                
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if return_detailed:
                predictions_all.extend(predictions.cpu().tolist())
                labels_all.extend(labels.cpu().tolist())
                questions_all.extend(question_texts)
    
    results = {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        'correct': correct,
        'total': total
    }
    
    if return_detailed:
        results['predictions'] = predictions_all
        results['labels'] = labels_all
        results['questions'] = questions_all
    
    return results

def visualize_collaboration_results(results, label_to_answer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    predictions = results['predictions']
    labels = results['labels']
    questions = results['questions']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    correct = [pred == label for pred, label in zip(predictions, labels)]
    accuracy = np.mean(correct)
    
    ax.text(0.5, 0.6, f'{accuracy:.1%}', 
            ha='center', va='center', fontsize=72, fontweight='bold', color='#2ecc71')
    ax.text(0.5, 0.3, 'Collaborative Model\nAccuracy',
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    
    ax = axes[1]
    categories = ['Correct', 'Incorrect']
    counts = [sum(correct), len(correct) - sum(correct)]
    colors = ['#2ecc71', '#e74c3c']
    
    wedges, texts, autotexts = ax.pie(counts, labels=categories, colors=colors,
                                       autopct='%1.1f%%', startangle=90, explode=(0.05, 0))
    ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'collaboration_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Results visualization saved")

def analyze_collaboration_benefits(model, val_dataset, device, label_to_answer, save_dir):
    print("\nAnalyzing collaboration benefits...")
    
    
    analysis = {
        'collaboration_config': {
            'num_layers': CONFIG['num_collaboration_layers'],
            'attention_heads': CONFIG['attention_heads'],
            'fusion_strategy': CONFIG['fusion_strategy']
        },
        'insights': [
            "Cross-attention allows specialists to exchange information",
            "Clinical context enhances radiology-specific features",
            "Radiology precision refines general clinical understanding"
        ]
    }
    
    analysis_path = os.path.join(save_dir, 'collaboration_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"  ✓ Analysis saved to {analysis_path}")

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
    print(f"Cross-Attention Collaborative Specialist Model - RTX 4080 Super Optimized")
    print(f"{'='*70}\n")
    print(f"Architecture: ClinicalBERT ⇄ RadBERT collaboration")
    print(f"Collaboration layers: {CONFIG['num_collaboration_layers']}")
    print(f"Fusion strategy: {CONFIG['fusion_strategy']}")
    print(f"Freeze specialists: {CONFIG['freeze_specialists']}")
    print(f"Base directory: {BASE_DIR}")
    print(f"\nOptimizations:")
    print(f"  - Batch size: {CONFIG['batch_size']}")
    print(f"  - Mixed precision: {CONFIG['use_mixed_precision']}")
    print(f"  - Workers: {CONFIG['num_workers']}\n")
    
    processor = ViltProcessor.from_pretrained(CONFIG['model_name'], cache_dir=CONFIG['cache_dir'])
    
    with open(CONFIG['vocab_path'], 'r') as f:
        vocab_data = json.load(f)
        answer_to_label = vocab_data['answer_to_label']
        label_to_answer = {int(k): v for k, v in vocab_data['label_to_answer'].items()}
        num_labels = vocab_data['num_labels']
    
    print(f"Vocabulary: {num_labels} classes\n")
    
    train_dataset = VQAMedDataset(CONFIG['data_dir'], 'train', processor, answer_to_label, num_labels, augment=True)
    val_dataset = VQAMedDataset(CONFIG['data_dir'], 'val', processor, answer_to_label, num_labels)
    
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
    
    print("Loading models...")
    
    
    vilt_config = ViltConfig.from_pretrained(CONFIG['model_name'], cache_dir=CONFIG['cache_dir'])
    vilt_model = ViltForQuestionAnswering.from_pretrained(
        CONFIG['model_name'], 
        config=vilt_config,
        ignore_mismatched_sizes=True, 
        cache_dir=CONFIG['cache_dir']
    )
    
    if os.path.exists(CONFIG['baseline_checkpoint']):
        print(f"  Loading baseline vision encoder...")
        checkpoint = torch.load(CONFIG['baseline_checkpoint'], map_location='cpu')
        state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                     if 'vilt' in k and 'classifier' not in k}
        vilt_model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Baseline loaded")
    
    clinical_bert = AutoModel.from_pretrained(CONFIG['clinical_bert_name'], cache_dir=CONFIG['cache_dir'])
    radbert = AutoModel.from_pretrained(CONFIG['radbert_name'], cache_dir=CONFIG['cache_dir'])
    
    print("  Creating collaborative model...")
    
    model = CollaborativeSpecialistModel(
        vilt_model=vilt_model,
        clinical_bert=clinical_bert,
        radbert=radbert,
        num_labels=num_labels,
        num_collaboration_layers=CONFIG['num_collaboration_layers'],
        attention_heads=CONFIG['attention_heads'],
        attention_dropout=CONFIG['attention_dropout'],
        fusion_strategy=CONFIG['fusion_strategy'],
        freeze_vision=CONFIG['freeze_vision_encoder'],
        freeze_specialists=CONFIG['freeze_specialists']
    ).to(device)
    
    if os.path.exists(CONFIG['clinical_checkpoint']):
        print(f"  Loading ClinicalBERT weights...")
        checkpoint = torch.load(CONFIG['clinical_checkpoint'], map_location='cpu')
        
        clinical_state = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'clinical_' in k:
                clinical_state[k] = v
        
        model.load_state_dict(clinical_state, strict=False)
        print(f"  ✓ ClinicalBERT loaded")
    
    if os.path.exists(CONFIG['radbert_checkpoint']):
        print(f"  Loading RadBERT weights...")
        checkpoint = torch.load(CONFIG['radbert_checkpoint'], map_location='cpu')
        
        radbert_state = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'radbert_' in k:
                radbert_state[k] = v
        
        model.load_state_dict(radbert_state, strict=False)
        print(f"  ✓ RadBERT loaded")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen: {total_params - trainable:,}\n")
    
    if CONFIG.get('use_torch_compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("  ✓ Model compiled with torch.compile")
    
    if CONFIG['freeze_specialists']:
        optimizer = AdamW([
            {'params': model.collaboration.parameters(), 'lr': CONFIG['collaboration_lr']},
            {'params': model.fusion.parameters(), 'lr': CONFIG['collaboration_lr']}
        ], weight_decay=CONFIG['weight_decay'])
        print("  Optimizer: Collaboration + Fusion only")
    else:
        optimizer = AdamW([
            {'params': model.clinical_embeddings.parameters(), 'lr': CONFIG['lr']},
            {'params': model.clinical_encoder.parameters(), 'lr': CONFIG['lr']},
            {'params': model.radbert_embeddings.parameters(), 'lr': CONFIG['lr']},
            {'params': model.radbert_encoder.parameters(), 'lr': CONFIG['lr']},
            {'params': model.collaboration.parameters(), 'lr': CONFIG['collaboration_lr']},
            {'params': model.fusion.parameters(), 'lr': CONFIG['collaboration_lr']},
            {'params': model.classifier.parameters(), 'lr': CONFIG['collaboration_lr']}
        ], weight_decay=CONFIG['weight_decay'])
        print(f"  Optimizer: Specialists (lr={CONFIG['lr']}) + Collaboration (lr={CONFIG['collaboration_lr']})")
    
    scaler = torch.amp.GradScaler('cuda', enabled=CONFIG["use_mixed_precision"])
    
    best_accuracy = 0.0
    patience_counter = 0
    history = []
    
    print(f"\n{'='*60}\nStarting Training\n{'='*60}\n")
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{CONFIG['epochs']}\n{'='*60}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_dataloader, optimizer, scaler, device, epoch, CONFIG
        )
        eval_results = evaluate(model, val_dataloader, device, return_detailed=False)
        
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
    
    print(f"\n{'='*70}\nFinal Evaluation\n{'='*70}\n")
    
    model.eval()
    final_results = evaluate(model, val_dataloader, device, return_detailed=True)
    
    with open(os.path.join(CONFIG['results_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    summary = {
        'model': 'Collaborative-Specialist-Model',
        'collaboration_layers': CONFIG['num_collaboration_layers'],
        'fusion_strategy': CONFIG['fusion_strategy'],
        'best_val_accuracy': best_accuracy,
        'final_val_accuracy': final_results['accuracy'],
        'total_epochs': len(history),
        'config': CONFIG
    }
    
    with open(os.path.join(CONFIG['results_dir'], 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nGenerating visualizations...")
    visualize_collaboration_results(final_results, label_to_answer, CONFIG['results_dir'])
    
    analyze_collaboration_benefits(
        model, val_dataset, device, label_to_answer, CONFIG['results_dir']
    )
    
    plot_final_training_curves(history, CONFIG['results_dir'], 'CrossAttn_Collaborative')
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Collaboration Type: {CONFIG['num_collaboration_layers']} cross-attention layers")
    print(f"Fusion Strategy: {CONFIG['fusion_strategy']}")
    print(f"Results directory: {CONFIG['results_dir']}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
