import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig, AutoModel
from transformers import logging as transformers_logging
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

transformers_logging.set_verbosity_error()
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

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
    "checkpoint_dir": os.path.join(BASE_DIR, "vqa_vilt_gated_outputs", "checkpoints"),
    "checkpoint_path": "vilt_gated_ensemble_best.pth",
    "results_dir": os.path.join(BASE_DIR, "vqa_vilt_gated_outputs", "results"),
    "plots_dir": os.path.join(BASE_DIR, "vqa_vilt_gated_outputs", "plots"),
    "vocab_path": os.path.join(BASE_DIR, "vqa_vilt_baseline_outputs", "checkpoints", "answer_vocab.json"),
    "cache_dir": os.path.join(BASE_DIR, "model_cache"),
    
    "gate_type": "question_based",
    "gate_hidden_dim": 256,
    "gate_dropout": 0.1,
    "use_gate_regularization": True,
    "gate_entropy_weight": 0.001,
    "freeze_branches": False,
    
    "epochs": 25,
    "lr": 1e-5,
    "gate_lr": 1e-4,
    "batch_size": 24,
    "use_mixed_precision": True,
    "use_torch_compile": True,
    "enable_tf32": True,
    "num_workers": 8,
    "gradient_accumulation_steps": 2,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "patience": 5,
    "seed": 42,
    "freeze_vision_encoder": True,
    "pin_memory": True,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "interactive_mode": True,
    "check_interval": 5,
}

class QuestionBasedGate(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, question_embeds):
        pooled = question_embeds.mean(dim=1)
        gate_weight = self.gate_network(pooled)
        return gate_weight

class MultimodalGate(nn.Module):
    def __init__(self, text_dim=768, image_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, question_embeds, image_embeds):
        text_pooled = self.text_projection(question_embeds.mean(dim=1))
        image_pooled = self.image_projection(image_embeds.mean(dim=1))
        combined = torch.cat([text_pooled, image_pooled], dim=1)
        gate_weight = self.gate_network(combined)
        return gate_weight

class HierarchicalGate(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.general_vs_specialized = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.clinical_vs_radiology = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, question_embeds):
        pooled = question_embeds.mean(dim=1)
        spec_weight = self.general_vs_specialized(pooled)
        specialist_weight = self.clinical_vs_radiology(pooled)
        
        gate_weight = 0.5 + (specialist_weight - 0.5) * spec_weight
        
        return gate_weight, spec_weight, specialist_weight

class ViltDualBranch(nn.Module):
    def __init__(self, vilt_model, clinical_bert, radbert, num_labels, freeze_vision=True):
        super().__init__()
        
        self.vilt = vilt_model
        self.config = vilt_model.config
        
        self.clinical_embeddings = clinical_bert.embeddings
        self.clinical_encoder = clinical_bert.encoder
        
        self.radbert_embeddings = radbert.embeddings
        self.radbert_encoder = radbert.encoder
        
        self.clinical_classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.radbert_classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        if freeze_vision:
            for name, param in self.vilt.vilt.embeddings.named_parameters():
                if 'patch' in name.lower():
                    param.requires_grad = False
    
    def _encode_text(self, input_ids, token_type_ids, attention_mask, encoder_embeddings, encoder):
        text_embeds = encoder_embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=text_embeds.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(text_embeds.dtype).min
        
        outputs = encoder(text_embeds, attention_mask=extended_attention_mask, return_dict=True)
        return outputs.last_hidden_state
    
    def encode_with_clinical(self, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask):
        text_encoded = self._encode_text(input_ids, token_type_ids, attention_mask, 
                                         self.clinical_embeddings, self.clinical_encoder)
        
        vilt_outputs = self.vilt.vilt(
            inputs_embeds=text_encoded,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
        
        return vilt_outputs.pooler_output, text_encoded
    
    def encode_with_radbert(self, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask):
        text_encoded = self._encode_text(input_ids, token_type_ids, attention_mask,
                                         self.radbert_embeddings, self.radbert_encoder)
        
        vilt_outputs = self.vilt.vilt(
            inputs_embeds=text_encoded,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
        
        return vilt_outputs.pooler_output, text_encoded

class GatedSpecialistEnsemble(nn.Module):
    def __init__(self, dual_branch, num_labels, gate_type="question_based", 
                 gate_hidden_dim=256, gate_dropout=0.1, freeze_branches=False):
        super().__init__()
        
        self.dual_branch = dual_branch
        self.gate_type = gate_type
        self.num_labels = num_labels
        
        if gate_type == "question_based":
            self.gate = QuestionBasedGate(768, gate_hidden_dim, gate_dropout)
        elif gate_type == "multimodal":
            self.gate = MultimodalGate(768, 768, gate_hidden_dim, gate_dropout)
        elif gate_type == "hierarchical":
            self.gate = HierarchicalGate(768, gate_hidden_dim, gate_dropout)
        
        if freeze_branches:
            for param in self.dual_branch.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask, return_gate_info=False):
        clinical_pooled, clinical_text = self.dual_branch.encode_with_clinical(
            input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask
        )
        radbert_pooled, radbert_text = self.dual_branch.encode_with_radbert(
            input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask
        )
        
        clinical_logits = self.dual_branch.clinical_classifier(clinical_pooled)
        radbert_logits = self.dual_branch.radbert_classifier(radbert_pooled)
        
        if self.gate_type == "question_based":
            avg_text = (clinical_text + radbert_text) / 2
            gate_weight = self.gate(avg_text)
            gate_info = {'gate_weight': gate_weight}
        elif self.gate_type == "multimodal":
            avg_text = (clinical_text + radbert_text) / 2
            image_embeds = self.dual_branch.vilt.vilt.embeddings.patch_embeddings(pixel_values)
            gate_weight = self.gate(avg_text, image_embeds)
            gate_info = {'gate_weight': gate_weight}
        elif self.gate_type == "hierarchical":
            avg_text = (clinical_text + radbert_text) / 2
            gate_weight, spec_weight, specialist_weight = self.gate(avg_text)
            gate_info = {
                'gate_weight': gate_weight,
                'spec_weight': spec_weight,
                'specialist_weight': specialist_weight
            }
        
        blended_logits = (1 - gate_weight) * clinical_logits + gate_weight * radbert_logits
        
        if return_gate_info:
            gate_info.update({
                'clinical_logits': clinical_logits,
                'radbert_logits': radbert_logits,
            })
            return blended_logits, gate_info
        
        return blended_logits

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
            return result
        except:
            return self.__getitem__(0)

def custom_collate_fn(batch):
    question_texts = [item.pop('question_text', '') for item in batch]
    
    keys = batch[0].keys()
    collated = {key: torch.stack([item[key] for item in batch]) for key in keys}
    collated['question_texts'] = question_texts
    return collated

def compute_gated_loss(logits, labels, gate_info, config):
    criterion = nn.CrossEntropyLoss()
    cls_loss = criterion(logits, labels)
    
    total_loss = cls_loss
    loss_dict = {'cls_loss': cls_loss.item()}
    
    if config["use_gate_regularization"]:
        gate_weight = gate_info['gate_weight']
        
        eps = 1e-6
        gate_weight_clamped = torch.clamp(gate_weight, min=eps, max=1.0 - eps)
        
        gate_entropy = -(
            gate_weight_clamped * torch.log(gate_weight_clamped + eps) + 
            (1 - gate_weight_clamped) * torch.log(1 - gate_weight_clamped + eps)
        ).mean()
        
        if torch.isnan(gate_entropy) or torch.isinf(gate_entropy):
            entropy_loss = torch.tensor(0.0, device=gate_weight.device)
        else:
            entropy_loss = config["gate_entropy_weight"] * gate_entropy
        
        total_loss = total_loss + entropy_loss
        loss_dict['entropy_loss'] = entropy_loss.item()
        loss_dict['avg_gate'] = gate_weight.mean().item()
    
    return total_loss, loss_dict

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    gate_stats = {'sum': 0, 'count': 0, 'distribution': []}
    nan_detected = False
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", position=0, leave=True)
    
    for step, batch in enumerate(progress_bar):
        labels = batch.pop('labels').to(device)
        question_texts = batch.pop('question_texts')
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config["use_mixed_precision"]):
            base_model = model
            logits, gate_info = base_model(**batch_gpu, return_gate_info=True)
            
            loss, loss_dict = compute_gated_loss(logits, labels, gate_info, config)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠ NaN/Inf loss detected at step {step}! Skipping batch...")
                nan_detected = True
                optimizer.zero_grad()
                continue
            
            loss = loss / config["gradient_accumulation_steps"]
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config["gradient_accumulation_steps"] == 0:
            scaler.unscale_(optimizer)
            
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"\n⚠ NaN/Inf gradient in {name}! Skipping update...")
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                optimizer.zero_grad()
                continue
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        if not torch.isnan(loss):
            total_loss += loss.item() * config["gradient_accumulation_steps"]
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += labels.size(0)
            
            gate_weights = gate_info['gate_weight'].detach().cpu().numpy().flatten()
            if not np.isnan(gate_weights).any():
                gate_stats['sum'] += gate_weights.sum()
                gate_stats['count'] += len(gate_weights)
                gate_stats['distribution'].extend(gate_weights.tolist())
        
        if gate_stats['count'] > 0:
            avg_gate = gate_stats['sum'] / gate_stats['count']
        else:
            avg_gate = 0.5
            
        progress_bar.set_postfix({
            'loss': f"{loss.item()*config['gradient_accumulation_steps'] if not torch.isnan(loss) else float('nan'):.4f}",
            'acc': f"{correct/total if total > 0 else 0:.4f}",
            'gate': f"{avg_gate:.3f}"
        })
    
    if nan_detected:
        print(f"⚠ NaN detected during training. Consider reducing learning rate.")
    
    avg_gate = gate_stats['sum'] / gate_stats['count'] if gate_stats['count'] > 0 else 0.5
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('nan')
    avg_acc = correct / total if total > 0 else 0.0
    
    return avg_loss, avg_acc, avg_gate, gate_stats['distribution']

def evaluate(model, dataloader, dataset, device, return_detailed=False):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    
    gate_analysis = {
        'gate_weights': [],
        'predictions': [],
        'labels': [],
        'questions': [],
        'clinical_correct': [],
        'radbert_correct': [],
        'ensemble_correct': [],
        'clinical_predictions': [],
        'radbert_predictions': []
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", position=0, leave=True):
            labels = batch.pop('labels').to(device)
            question_texts = batch.pop('question_texts')
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=CONFIG["use_mixed_precision"]):
                base_model = model
                logits, gate_info = base_model(**batch_gpu, return_gate_info=True)
                
                loss, _ = compute_gated_loss(logits, labels, gate_info, CONFIG)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if return_detailed:
                clinical_preds = torch.argmax(gate_info['clinical_logits'], dim=1)
                radbert_preds = torch.argmax(gate_info['radbert_logits'], dim=1)
                
                gate_analysis['gate_weights'].extend(gate_info['gate_weight'].cpu().numpy().flatten().tolist())
                gate_analysis['predictions'].extend(predictions.cpu().tolist())
                gate_analysis['labels'].extend(labels.cpu().tolist())
                gate_analysis['questions'].extend(question_texts)
                gate_analysis['clinical_predictions'].extend(clinical_preds.cpu().tolist())
                gate_analysis['radbert_predictions'].extend(radbert_preds.cpu().tolist())
                gate_analysis['clinical_correct'].extend((clinical_preds == labels).cpu().tolist())
                gate_analysis['radbert_correct'].extend((radbert_preds == labels).cpu().tolist())
                gate_analysis['ensemble_correct'].extend((predictions == labels).cpu().tolist())
    
    results = {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total,
        'correct': correct,
        'total': total
    }
    
    if return_detailed:
        results['gate_analysis'] = gate_analysis
    
    return results

def visualize_gate_analysis(gate_analysis, label_to_answer, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    gate_weights = np.array(gate_analysis['gate_weights'])
    questions = gate_analysis['questions']
    labels = gate_analysis['labels']
    predictions = gate_analysis['predictions']
    clinical_correct = gate_analysis['clinical_correct']
    radbert_correct = gate_analysis['radbert_correct']
    ensemble_correct = gate_analysis['ensemble_correct']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    ax.hist(gate_weights, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(gate_weights.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {gate_weights.mean():.3f}')
    ax.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Equal (0.5)')
    ax.set_xlabel('Gate Weight (0=Clinical, 1=RadBERT)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Gate Weight Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    correct_gates = [gate_weights[i] for i in range(len(gate_weights)) if ensemble_correct[i]]
    incorrect_gates = [gate_weights[i] for i in range(len(gate_weights)) if not ensemble_correct[i]]
    
    ax.hist(correct_gates, bins=30, alpha=0.5, label=f'Correct (n={len(correct_gates)})', color='green', edgecolor='black')
    ax.hist(incorrect_gates, bins=30, alpha=0.5, label=f'Incorrect (n={len(incorrect_gates)})', color='red', edgecolor='black')
    ax.set_xlabel('Gate Weight', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Gate Distribution by Correctness', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    
    gate_helps = []
    gate_hurts = []
    
    for i in range(len(predictions)):
        clinical_wrong = not clinical_correct[i]
        radbert_wrong = not radbert_correct[i]
        ensemble_right = ensemble_correct[i]
        
        if (clinical_wrong or radbert_wrong) and ensemble_right:
            gate_helps.append(gate_weights[i])
        elif (clinical_correct[i] or radbert_correct[i]) and not ensemble_right:
            gate_hurts.append(gate_weights[i])
    
    ax.hist(gate_helps, bins=20, alpha=0.6, label=f'Gate Helps (n={len(gate_helps)})', color='green', edgecolor='black')
    ax.hist(gate_hurts, bins=20, alpha=0.6, label=f'Gate Hurts (n={len(gate_hurts)})', color='red', edgecolor='black')
    ax.set_xlabel('Gate Weight', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Gate Impact on Predictions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    
    for i in range(len(bins)-1):
        mask = (gate_weights >= bins[i]) & (gate_weights < bins[i+1])
        if mask.sum() > 0:
            bin_acc = np.mean([ensemble_correct[j] for j in range(len(ensemble_correct)) if mask[j]])
            bin_accuracies.append(bin_acc)
        else:
            bin_accuracies.append(0)
    
    ax.plot(bin_centers, bin_accuracies, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.axhline(np.mean(ensemble_correct), color='red', linestyle='--', label='Overall Acc')
    ax.set_xlabel('Gate Weight Bin', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Gate Weight', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gate_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Gate analysis saved to {save_dir}/gate_analysis.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes[0]
    question_lengths = [len(q.split()) for q in questions]
    
    scatter = ax.scatter(question_lengths, gate_weights, c=ensemble_correct, 
                        cmap='RdYlGn', alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
    ax.set_xlabel('Question Length (words)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gate Weight (0=Clinical, 1=RadBERT)', fontsize=12, fontweight='bold')
    ax.set_title('Gate Decision vs Question Complexity', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Correct')
    
    ax = axes[1]
    clinical_acc = np.mean(clinical_correct)
    radbert_acc = np.mean(radbert_correct)
    ensemble_acc = np.mean(ensemble_correct)
    
    models = ['ClinicalBERT\nBranch', 'RadBERT\nBranch', 'Gated\nEnsemble']
    accuracies = [clinical_acc, radbert_acc, ensemble_acc]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Model comparison saved to {save_dir}/model_comparison.png")
    
    stats = {
        'overall': {
            'avg_gate_weight': float(gate_weights.mean()),
            'std_gate_weight': float(gate_weights.std()),
            'clinical_only_acc': float(clinical_acc),
            'radbert_only_acc': float(radbert_acc),
            'ensemble_acc': float(ensemble_acc),
            'improvement_over_clinical': float(ensemble_acc - clinical_acc),
            'improvement_over_radbert': float(ensemble_acc - radbert_acc)
        },
        'gate_helps_cases': len(gate_helps),
        'gate_hurts_cases': len(gate_hurts),
        'gate_distribution': {
            'mean': float(gate_weights.mean()),
            'median': float(np.median(gate_weights)),
            'std': float(gate_weights.std()),
            'min': float(gate_weights.min()),
            'max': float(gate_weights.max()),
            'percentiles': {
                '25': float(np.percentile(gate_weights, 25)),
                '50': float(np.percentile(gate_weights, 50)),
                '75': float(np.percentile(gate_weights, 75))
            }
        }
    }
    
    with open(os.path.join(save_dir, 'gate_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  ✓ Statistics saved to {save_dir}/gate_statistics.json")

def analyze_gate_decisions(gate_analysis, label_to_answer, save_path):
    
    strong_clinical = []
    strong_radbert = []
    uncertain = []
    
    for i in range(len(gate_analysis['gate_weights'])):
        gate = gate_analysis['gate_weights'][i]
        question = gate_analysis['questions'][i]
        label = gate_analysis['labels'][i]
        pred = gate_analysis['predictions'][i]
        correct = gate_analysis['ensemble_correct'][i]
        
        case = {
            'question': question,
            'true_answer': label_to_answer[label],
            'prediction': label_to_answer[pred],
            'gate_weight': gate,
            'correct': correct
        }
        
        if gate < 0.3:
            strong_clinical.append(case)
        elif gate > 0.7:
            strong_radbert.append(case)
        else:
            uncertain.append(case)
    
    analysis = {
        'strong_clinical_preference': {
            'count': len(strong_clinical),
            'examples': strong_clinical[:20]
        },
        'strong_radbert_preference': {
            'count': len(strong_radbert),
            'examples': strong_radbert[:20]
        },
        'uncertain_cases': {
            'count': len(uncertain),
            'examples': uncertain[:20]
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"  ✓ Gate decision analysis saved to {save_path}")
    
    print(f"\n  Gate Decision Summary:")
    print(f"    Strong Clinical preference (<0.3): {len(strong_clinical)} cases")
    print(f"    Strong RadBERT preference (>0.7):  {len(strong_radbert)} cases")
    print(f"    Uncertain (0.3-0.7):                {len(uncertain)} cases")

def plot_final_training_curves(history, save_dir, model_name):
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    train_gates = [h['train_avg_gate'] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(epochs, train_gates, 'g-', linewidth=2)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1.5, label='Equal weight')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Gate Weight', fontsize=12, fontweight='bold')
    ax.set_title('Gate Weight Evolution (0=Clinical, 1=RadBERT)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1 = ax.plot(epochs, val_acc, 'r-', label='Val Accuracy', linewidth=2)
    l2 = ax2.plot(epochs, train_gates, 'g--', label='Avg Gate Weight', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold', color='r')
    ax2.set_ylabel('Average Gate Weight', fontsize=12, fontweight='bold', color='g')
    ax.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='g')
    ax.set_title('Accuracy vs Gate Weight Evolution', fontsize=14, fontweight='bold')
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'{model_name}_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Training curves saved to {plot_path}")

def generate_latex_table(gate_analysis, label_to_answer, save_path):
    
    gate_weights = np.array(gate_analysis['gate_weights'])
    clinical_correct = np.array(gate_analysis['clinical_correct'])
    radbert_correct = np.array(gate_analysis['radbert_correct'])
    ensemble_correct = np.array(gate_analysis['ensemble_correct'])
    
    latex_content = "% =============== Table 1: Overall Model Performance ===============\n"
    latex_content += "\\begin{table}[h]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Performance Comparison of Specialist Models and Gated Ensemble}\n"
    latex_content += "\\begin{tabular}{lcccc}\n"
    latex_content += "\\hline\n"
    latex_content += "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Correct} & \\textbf{Total} & \\textbf{Improvement} \\\\\n"
    latex_content += "\\hline\n"
    
    clinical_acc = np.mean(clinical_correct)
    radbert_acc = np.mean(radbert_correct)
    ensemble_acc = np.mean(ensemble_correct)
    total = len(ensemble_correct)
    
    latex_content += f"ClinicalBERT Branch & {clinical_acc:.4f} & {int(clinical_correct.sum())} & {total} & - \\\\\n"
    latex_content += f"RadBERT Branch & {radbert_acc:.4f} & {int(radbert_correct.sum())} & {total} & - \\\\\n"
    latex_content += f"Gated Ensemble & {ensemble_acc:.4f} & {int(ensemble_correct.sum())} & {total} & +{(ensemble_acc - max(clinical_acc, radbert_acc))*100:.2f}\\% \\\\\n"
    latex_content += "\\hline\n"
    latex_content += "\\end{tabular}\n"
    latex_content += "\\label{tab:overall_performance}\n"
    latex_content += "\\end{table}\n\n"
    
    latex_content += "% =============== Table 2: Gate Weight Statistics ===============\n"
    latex_content += "\\begin{table}[h]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Gate Weight Distribution Statistics}\n"
    latex_content += "\\begin{tabular}{lc}\n"
    latex_content += "\\hline\n"
    latex_content += "\\textbf{Statistic} & \\textbf{Value} \\\\\n"
    latex_content += "\\hline\n"
    latex_content += f"Mean & {gate_weights.mean():.4f} \\\\\n"
    latex_content += f"Median & {np.median(gate_weights):.4f} \\\\\n"
    latex_content += f"Std. Deviation & {gate_weights.std():.4f} \\\\\n"
    latex_content += f"Min & {gate_weights.min():.4f} \\\\\n"
    latex_content += f"Max & {gate_weights.max():.4f} \\\\\n"
    latex_content += f"25th Percentile & {np.percentile(gate_weights, 25):.4f} \\\\\n"
    latex_content += f"75th Percentile & {np.percentile(gate_weights, 75):.4f} \\\\\n"
    latex_content += "\\hline\n"
    latex_content += "\\end{tabular}\n"
    latex_content += "\\label{tab:gate_statistics}\n"
    latex_content += "\\end{table}\n\n"
    
    strong_clinical = (gate_weights < 0.3).sum()
    balanced = ((gate_weights >= 0.3) & (gate_weights <= 0.7)).sum()
    strong_radbert = (gate_weights > 0.7).sum()
    
    latex_content += "% =============== Table 3: Gate Preference Distribution ===============\n"
    latex_content += "\\begin{table}[h]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Distribution of Gate Preferences}\n"
    latex_content += "\\begin{tabular}{lcc}\n"
    latex_content += "\\hline\n"
    latex_content += "\\textbf{Preference} & \\textbf{Count} & \\textbf{Percentage} \\\\\n"
    latex_content += "\\hline\n"
    latex_content += f"Strong Clinical ($w < 0.3$) & {strong_clinical} & {strong_clinical/total*100:.1f}\\% \\\\\n"
    latex_content += f"Balanced ($0.3 \\leq w \\leq 0.7$) & {balanced} & {balanced/total*100:.1f}\\% \\\\\n"
    latex_content += f"Strong RadBERT ($w > 0.7$) & {strong_radbert} & {strong_radbert/total*100:.1f}\\% \\\\\n"
    latex_content += "\\hline\n"
    latex_content += f"Total & {total} & 100.0\\% \\\\\n"
    latex_content += "\\hline\n"
    latex_content += "\\end{tabular}\n"
    latex_content += "\\label{tab:gate_distribution}\n"
    latex_content += "\\end{table}\n\n"
    
    with open(save_path, 'w') as f:
        f.write(latex_content)
    
    print(f"  ✓ LaTeX tables saved to {save_path}")

def generate_detailed_report(history, gate_analysis, label_to_answer, config, save_path):
    
    gate_weights = np.array(gate_analysis['gate_weights'])
    clinical_correct = np.array(gate_analysis['clinical_correct'])
    radbert_correct = np.array(gate_analysis['radbert_correct'])
    ensemble_correct = np.array(gate_analysis['ensemble_correct'])
    
    report = []
    report.append("="*80)
    report.append("GATED SPECIALIST ENSEMBLE - DETAILED RESEARCH REPORT")
    report.append("="*80)
    report.append("")
    
    report.append("1. MODEL CONFIGURATION")
    report.append("-" * 80)
    report.append(f"Gate Type: {config['gate_type']}")
    report.append(f"Gate Hidden Dim: {config['gate_hidden_dim']}")
    report.append(f"Gate Dropout: {config['gate_dropout']}")
    report.append(f"Freeze Branches: {config['freeze_branches']}")
    report.append(f"Learning Rate (Branches): {config['lr']}")
    report.append(f"Learning Rate (Gate): {config['gate_lr']}")
    report.append(f"Batch Size: {config['batch_size']}")
    report.append(f"Entropy Weight: {config['entropy_weight']}")
    report.append("")
    
    report.append("2. TRAINING SUMMARY")
    report.append("-" * 80)
    report.append(f"Total Epochs Trained: {len(history)}")
    
    best_epoch = max(history, key=lambda x: x['val_acc'])
    report.append(f"Best Epoch: {best_epoch['epoch']}")
    report.append(f"Best Validation Accuracy: {best_epoch['val_acc']:.4f} ({best_epoch['val_acc']*100:.2f}%)")
    report.append(f"Best Training Accuracy: {best_epoch['train_acc']:.4f} ({best_epoch['train_acc']*100:.2f}%)")
    report.append(f"Best Validation Loss: {best_epoch['val_loss']:.4f}")
    report.append(f"Best Training Loss: {best_epoch['train_loss']:.4f}")
    report.append(f"Gate Weight at Best Epoch: {best_epoch['train_avg_gate']:.4f}")
    report.append("")
    
    report.append("3. FINAL MODEL PERFORMANCE")
    report.append("-" * 80)
    clinical_acc = np.mean(clinical_correct)
    radbert_acc = np.mean(radbert_correct)
    ensemble_acc = np.mean(ensemble_correct)
    
    report.append(f"ClinicalBERT Branch Accuracy: {clinical_acc:.4f} ({clinical_acc*100:.2f}%)")
    report.append(f"RadBERT Branch Accuracy: {radbert_acc:.4f} ({radbert_acc*100:.2f}%)")
    report.append(f"Gated Ensemble Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
    report.append("")
    report.append(f"Improvement over ClinicalBERT: {(ensemble_acc - clinical_acc)*100:+.2f}%")
    report.append(f"Improvement over RadBERT: {(ensemble_acc - radbert_acc)*100:+.2f}%")
    report.append(f"Improvement over Best Individual: {(ensemble_acc - max(clinical_acc, radbert_acc))*100:+.2f}%")
    report.append("")
    
    report.append("4. GATE BEHAVIOR ANALYSIS")
    report.append("-" * 80)
    report.append(f"Average Gate Weight: {gate_weights.mean():.4f}")
    report.append(f"Median Gate Weight: {np.median(gate_weights):.4f}")
    report.append(f"Std Deviation: {gate_weights.std():.4f}")
    report.append(f"Min Gate Weight: {gate_weights.min():.4f}")
    report.append(f"Max Gate Weight: {gate_weights.max():.4f}")
    report.append("")
    
    strong_clinical = (gate_weights < 0.3).sum()
    balanced = ((gate_weights >= 0.3) & (gate_weights <= 0.7)).sum()
    strong_radbert = (gate_weights > 0.7).sum()
    total = len(gate_weights)
    
    report.append("Gate Preference Distribution:")
    report.append(f"  Strong Clinical Preference (w < 0.3): {strong_clinical} ({strong_clinical/total*100:.1f}%)")
    report.append(f"  Balanced (0.3 ≤ w ≤ 0.7): {balanced} ({balanced/total*100:.1f}%)")
    report.append(f"  Strong RadBERT Preference (w > 0.7): {strong_radbert} ({strong_radbert/total*100:.1f}%)")
    report.append("")
    
    report.append("5. ERROR ANALYSIS")
    report.append("-" * 80)
    
    both_wrong_ensemble_right = (~clinical_correct & ~radbert_correct & ensemble_correct).sum()
    report.append(f"Both branches WRONG, Ensemble CORRECT: {both_wrong_ensemble_right}")
    
    both_right_ensemble_wrong = (clinical_correct & radbert_correct & ~ensemble_correct).sum()
    report.append(f"Both branches CORRECT, Ensemble WRONG: {both_right_ensemble_wrong}")
    
    one_right_ensemble_right = ((clinical_correct & ~radbert_correct) | (~clinical_correct & radbert_correct)) & ensemble_correct
    report.append(f"One branch CORRECT, Ensemble CORRECT: {one_right_ensemble_right.sum()}")
    
    one_right_ensemble_wrong = ((clinical_correct & ~radbert_correct) | (~clinical_correct & radbert_correct)) & ~ensemble_correct
    report.append(f"One branch CORRECT, Ensemble WRONG: {one_right_ensemble_wrong.sum()}")
    report.append("")
    
    report.append("6. QUESTION COMPLEXITY ANALYSIS")
    report.append("-" * 80)
    questions = gate_analysis['questions']
    question_lengths = [len(q.split()) for q in questions]
    
    report.append(f"Average Question Length: {np.mean(question_lengths):.1f} words")
    report.append(f"Median Question Length: {np.median(question_lengths):.0f} words")
    report.append(f"Min/Max Length: {min(question_lengths)} / {max(question_lengths)} words")
    report.append("")
    
    short_q = np.array(question_lengths) <= np.median(question_lengths)
    long_q = np.array(question_lengths) > np.median(question_lengths)
    
    report.append(f"Accuracy on Short Questions: {ensemble_correct[short_q].mean():.4f}")
    report.append(f"Accuracy on Long Questions: {ensemble_correct[long_q].mean():.4f}")
    report.append("")
    
    report.append("7. KEY RESEARCH FINDINGS")
    report.append("-" * 80)
    
    if ensemble_acc > max(clinical_acc, radbert_acc):
        report.append(f"✓ Gating mechanism successfully improved over individual specialists")
    else:
        report.append(f"✗ Gating did not improve over best individual specialist")
    
    if gate_weights.mean() > 0.5:
        report.append(f"✓ Gate shows preference for RadBERT (radiology specialist)")
    else:
        report.append(f"✓ Gate shows preference for ClinicalBERT (general medical specialist)")
    
    if gate_weights.std() > 0.2:
        report.append(f"✓ Gate shows dynamic behavior (high variance = adaptive selection)")
    else:
        report.append(f"✗ Gate shows limited dynamics (low variance = near-constant weighting)")
    
    report.append("")
    
    report.append("8. RECOMMENDATIONS FOR FUTURE WORK")
    report.append("-" * 80)
    
    if ensemble_acc - max(clinical_acc, radbert_acc) < 0.01:
        report.append("• Consider increasing gate network capacity or complexity")
        report.append("• Try different gate architectures (hierarchical, attention-based)")
    
    if gate_weights.std() < 0.15:
        report.append("• Gate may be under-utilizing specialist diversity")
        report.append("• Consider adding diversity regularization or confidence calibration")
    
    if both_right_ensemble_wrong > 5:
        report.append("• Investigate cases where ensemble fails despite both specialists being correct")
        report.append("• May indicate gate learning artifacts or overconfidence")
    
    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"  ✓ Detailed research report saved to {save_path}")

def export_predictions_csv(gate_analysis, label_to_answer, save_path):
    
    import csv
    
    gate_weights = gate_analysis['gate_weights']
    clinical_preds = gate_analysis['clinical_predictions']
    radbert_preds = gate_analysis['radbert_predictions']
    ensemble_preds = gate_analysis['predictions']
    labels = gate_analysis['labels']
    questions = gate_analysis['questions']
    clinical_correct = gate_analysis['clinical_correct']
    radbert_correct = gate_analysis['radbert_correct']
    ensemble_correct = gate_analysis['ensemble_correct']
    
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow([
            'Sample_ID',
            'Question',
            'True_Answer',
            'Clinical_Prediction',
            'Clinical_Correct',
            'RadBERT_Prediction',
            'RadBERT_Correct',
            'Ensemble_Prediction',
            'Ensemble_Correct',
            'Gate_Weight',
            'Question_Length',
            'Agreement_Clinical_RadBERT',
            'Gate_Helped'
        ])
        
        for i in range(len(questions)):
            question = questions[i]
            true_answer = label_to_answer[labels[i]]
            clinical_pred = label_to_answer[clinical_preds[i]]
            radbert_pred = label_to_answer[radbert_preds[i]]
            ensemble_pred = label_to_answer[ensemble_preds[i]]
            gate_weight = gate_weights[i]
            question_len = len(question.split())
            
            agreement = 'Yes' if clinical_preds[i] == radbert_preds[i] else 'No'
            
            gate_helped = 'No'
            if ensemble_correct[i] and not (clinical_correct[i] and radbert_correct[i]):
                gate_helped = 'Yes'
            elif not ensemble_correct[i] and (clinical_correct[i] or radbert_correct[i]):
                gate_helped = 'Hurt'
            
            writer.writerow([
                i + 1,
                question,
                true_answer,
                clinical_pred,
                'Yes' if clinical_correct[i] else 'No',
                radbert_pred,
                'Yes' if radbert_correct[i] else 'No',
                ensemble_pred,
                'Yes' if ensemble_correct[i] else 'No',
                f'{gate_weight:.4f}',
                question_len,
                agreement,
                gate_helped
            ])
    
    print(f"  ✓ Predictions CSV exported to {save_path}")

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
    os.makedirs(CONFIG.get('plots_dir', CONFIG['results_dir']), exist_ok=True)
    
    print(f"\n{'='*70}\nGated Specialist Ensemble Training - RTX 4080 Super Optimized\n{'='*70}\n")
    print(f"Architecture: ClinicalBERT + RadBERT with {CONFIG['gate_type']} gating")
    print(f"Freeze branches: {CONFIG['freeze_branches']}")
    print(f"Base directory: {BASE_DIR}")
    print(f"\nOptimizations:")
    print(f"  - Batch size: {CONFIG['batch_size']}")
    print(f"  - Mixed precision: {CONFIG['use_mixed_precision']}")
    print(f"  - Torch compile: {CONFIG.get('use_torch_compile', False)}")
    print(f"  - Workers: {CONFIG['num_workers']}\n")
    
    processor = ViltProcessor.from_pretrained(CONFIG['model_name'])
    
    with open(CONFIG['vocab_path'], 'r') as f:
        vocab_data = json.load(f)
        answer_to_label = vocab_data['answer_to_label']
        label_to_answer = {int(k): v for k, v in vocab_data['label_to_answer'].items()}
        num_labels = vocab_data['num_labels']
    
    print("Loading datasets...")
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
    
    vilt_config = ViltConfig.from_pretrained(CONFIG['model_name'])
    vilt_model = ViltForQuestionAnswering.from_pretrained(
        CONFIG['model_name'], 
        config=vilt_config,
        ignore_mismatched_sizes=True
    )
    
    if os.path.exists(CONFIG['baseline_checkpoint']):
        print(f"  Loading baseline vision encoder...")
        checkpoint = torch.load(CONFIG['baseline_checkpoint'], map_location='cpu')
        state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'vilt' in k and 'classifier' not in k}
        vilt_model.load_state_dict(state_dict, strict=False)
    
    clinical_bert = AutoModel.from_pretrained(CONFIG['clinical_bert_name'], cache_dir=CONFIG['cache_dir'])
    radbert = AutoModel.from_pretrained(CONFIG['radbert_name'], cache_dir=CONFIG['cache_dir'])
    
    dual_branch = ViltDualBranch(vilt_model, clinical_bert, radbert, num_labels, CONFIG['freeze_vision_encoder'])
    
    if os.path.exists(CONFIG['clinical_checkpoint']):
        print(f"  Loading ClinicalBERT branch weights...")
        checkpoint = torch.load(CONFIG['clinical_checkpoint'], map_location='cpu')
        
        clinical_state = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'clinical_embeddings' in k or 'clinical_encoder' in k:
                clinical_state[k] = v
            elif 'classifier' in k:
                clinical_state[k.replace('classifier', 'clinical_classifier')] = v
        
        dual_branch.load_state_dict(clinical_state, strict=False)
        print(f"  ✓ ClinicalBERT weights loaded")
    
    if os.path.exists(CONFIG['radbert_checkpoint']):
        print(f"  Loading RadBERT branch weights...")
        checkpoint = torch.load(CONFIG['radbert_checkpoint'], map_location='cpu')
        
        radbert_state = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'radbert_embeddings' in k or 'radbert_encoder' in k:
                radbert_state[k] = v
            elif 'classifier' in k:
                radbert_state[k.replace('classifier', 'radbert_classifier')] = v
        
        dual_branch.load_state_dict(radbert_state, strict=False)
        print(f"  ✓ RadBERT weights loaded")
    
    model = GatedSpecialistEnsemble(
        dual_branch=dual_branch,
        num_labels=num_labels,
        gate_type=CONFIG['gate_type'],
        gate_hidden_dim=CONFIG['gate_hidden_dim'],
        gate_dropout=CONFIG['gate_dropout'],
        freeze_branches=CONFIG['freeze_branches']
    ).to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen: {total - trainable:,}\n")
    
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("  ✓ Model compiled with torch.compile")
    
    if CONFIG['freeze_branches']:
        optimizer = AdamW(model.gate.parameters(), lr=CONFIG['gate_lr'], weight_decay=CONFIG['weight_decay'])
        print("  Optimizer: Gate only")
    else:
        optimizer = AdamW([
            {'params': model.dual_branch.parameters(), 'lr': CONFIG['lr']},
            {'params': model.gate.parameters(), 'lr': CONFIG['gate_lr']}
        ], weight_decay=CONFIG['weight_decay'])
        print(f"  Optimizer: Branches (lr={CONFIG['lr']}) + Gate (lr={CONFIG['gate_lr']})")
    
    scaler = torch.amp.GradScaler('cuda', enabled=CONFIG["use_mixed_precision"])
    
    best_accuracy = 0.0
    patience_counter = 0
    history = []
    
    print(f"\n{'='*60}\nStarting Training\n{'='*60}\n")
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{CONFIG['epochs']}\n{'='*60}")
        
        train_loss, train_acc, train_gate, gate_dist = train_one_epoch(
            model, train_dataloader, optimizer, scaler, device, epoch, CONFIG
        )
        eval_results = evaluate(model, val_dataloader, val_dataset, device, return_detailed=False)
        
        val_acc = eval_results['accuracy']
        val_loss = eval_results['loss']
        
        print(f"\nResults:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, Gate={train_gate:.3f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        history.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_avg_gate': train_gate,
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
    
    print(f"\n{'='*70}\nFinal Evaluation with Gate Analysis\n{'='*70}\n")
    
    model.eval()
    final_results = evaluate(model, val_dataloader, val_dataset, device, return_detailed=True)
    
    with open(os.path.join(CONFIG['results_dir'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    summary = {
        'model': 'Gated-Specialist-Ensemble',
        'gate_type': CONFIG['gate_type'],
        'best_val_accuracy': best_accuracy,
        'final_val_accuracy': final_results['accuracy'],
        'total_epochs': len(history),
        'config': CONFIG
    }
    
    with open(os.path.join(CONFIG['results_dir'], 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nGenerating comprehensive research outputs...")
    print("-" * 70)
    
    visualize_gate_analysis(
        final_results['gate_analysis'],
        label_to_answer,
        CONFIG['results_dir']
    )
    
    analyze_gate_decisions(
        final_results['gate_analysis'],
        label_to_answer,
        os.path.join(CONFIG['results_dir'], 'gate_decisions.json')
    )
    
    plot_final_training_curves(history, CONFIG['results_dir'], 'Gated_Ensemble')
    
    generate_latex_table(
        final_results['gate_analysis'],
        label_to_answer,
        os.path.join(CONFIG['results_dir'], 'latex_tables.tex')
    )
    
    generate_detailed_report(
        history,
        final_results['gate_analysis'],
        label_to_answer,
        CONFIG,
        os.path.join(CONFIG['results_dir'], 'research_report.txt')
    )
    
    export_predictions_csv(
        final_results['gate_analysis'],
        label_to_answer,
        os.path.join(CONFIG['results_dir'], 'predictions_detailed.csv')
    )
    
    print("\n" + "="*70)
    print("ALL RESEARCH OUTPUTS GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutput Files:")
    print(f"  📊 Visualizations:")
    print(f"     - gate_analysis.png (4 plots: distribution, correctness, gate helps, entropy)")
    print(f"     - model_comparison.png (2 plots: complexity analysis, performance comparison)")
    print(f"     - Gated_Ensemble_training_curves.png (4 plots: loss, accuracy, gate evolution)")
    print(f"\n  📝 Textual Analysis:")
    print(f"     - research_report.txt (Comprehensive 8-section research report)")
    print(f"     - gate_statistics.json (Detailed statistical metrics)")
    print(f"     - gate_decisions.json (Gate preference patterns)")
    print(f"     - training_history.json (Epoch-by-epoch metrics)")
    print(f"     - training_summary.json (Final performance summary)")
    print(f"\n  📄 Research Paper Ready:")
    print(f"     - latex_tables.tex (3 publication-ready LaTeX tables)")
    print(f"     - predictions_detailed.csv (All predictions for external analysis)")
    print(f"\n  💾 Model Checkpoint:")
    print(f"     - {CONFIG['checkpoint_path']}")
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Gate Type: {CONFIG['gate_type']}")
    print(f"Results directory: {CONFIG['results_dir']}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()