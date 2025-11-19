import os
import json
import torch
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig, AutoModel
from transformers import logging as transformers_logging
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    "radbert_name": "zzxslp/RadBERT-RoBERTa-4m",
    "baseline_checkpoint": os.path.join(BASE_DIR, "vqa_vilt_baseline_outputs", "checkpoints", "vilt_baseline_best.pth"),
    "use_baseline_encoder": True,
    "data_dir": BASE_DIR,
    "checkpoint_dir": os.path.join(BASE_DIR, "vqa_vilt_radbert_outputs", "checkpoints"),
    "checkpoint_path": "vilt_radbert_best.pth",
    "results_dir": os.path.join(BASE_DIR, "vqa_vilt_radbert_outputs", "results"),
    "plots_dir": os.path.join(BASE_DIR, "vqa_vilt_radbert_outputs", "plots"),
    "vocab_path": os.path.join(BASE_DIR, "vqa_vilt_baseline_outputs", "checkpoints", "answer_vocab.json"),
    "epochs": 30,
    "lr": 2e-5,
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

class ViltWithRadBERT(torch.nn.Module):
    def __init__(self, vilt_model, radbert, num_labels, freeze_vision=True):
        super().__init__()
        
        self.vilt = vilt_model
        self.config = vilt_model.config
        
        self.radbert_embeddings = radbert.embeddings
        self.radbert_encoder = radbert.encoder
        
        self.classifier = torch.nn.Linear(self.config.hidden_size, num_labels)
        
        if freeze_vision:
            for name, param in self.vilt.vilt.embeddings.named_parameters():
                if 'patch' in name.lower():
                    param.requires_grad = False
            print("  ✓ Vision encoder frozen")
        
        print("  ✓ RadBERT: Complete text encoder (embeddings + 12 layers)")
    
    def forward(self, input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask):
        device = input_ids.device
        
        text_embeds = self.radbert_embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=text_embeds.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(text_embeds.dtype).min
        
        radbert_outputs = self.radbert_encoder(
            text_embeds,
            attention_mask=extended_attention_mask,
            return_dict=True
        )
        text_encoded = radbert_outputs.last_hidden_state
        
        vilt_outputs = self.vilt.vilt(
            inputs_embeds=text_encoded,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            return_dict=True
        )
        
        pooled_output = vilt_outputs.pooler_output
        
        class OutputWrapper:
            def __init__(self, pooler_output):
                self.pooler_output = pooler_output
        
        return OutputWrapper(pooled_output)

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
            return result
        except:
            return self.__getitem__(0)

def custom_collate_fn(batch):
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}", position=0, leave=True)
    
    for step, batch in enumerate(progress_bar):
        labels = batch.pop('labels').to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config["use_mixed_precision"]):
            base_model = model
            outputs = base_model(**batch)
            logits = base_model.classifier(outputs.pooler_output)
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
        
        progress_bar.set_postfix({'loss': f"{loss.item()*config['gradient_accumulation_steps']:.4f}", 
                                 'acc': f"{correct/total:.4f}"})
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", position=0, leave=True):
            labels = batch.pop('labels').to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=CONFIG["use_mixed_precision"]):
                base_model = model
                outputs = base_model(**batch)
                logits = base_model.classifier(outputs.pooler_output)
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

def plot_final_training_curves(history, save_dir, model_name="radbert"):
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training and Validation Loss - {model_name.upper()}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Training and Validation Accuracy - {model_name.upper()}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'{model_name}_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {plot_path}")

def save_complete_checkpoint(model, optimizer, scaler, epoch, metrics, config, vocab_dict, checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'config': config,
        'vocabulary': vocab_dict,
        'model_config': {
            'model_type': 'ViltWithRadBERT',
            'radbert_name': config.get('radbert_name', 'zzxslp/RadBERT-RoBERTa-4m'),
            'num_labels': vocab_dict['num_labels']
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    summary_path = checkpoint_path.replace('.pth', '_summary.json')
    summary = {
        'epoch': epoch,
        'metrics': metrics,
        'timestamp': str(torch.cuda.Event(enable_timing=False)),
        'model_type': 'ViltWithRadBERT'
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return checkpoint_path

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
    print(f"\n{'='*70}\nViLT-RadBERT Training - RTX 4080 Super Optimized\n{'='*70}\n")
    print(f"Base directory: {BASE_DIR}")
    print(f"Checkpoints: {CONFIG['checkpoint_dir']}")
    print(f"Results: {CONFIG['results_dir']}")
    print(f"\nOptimizations:")
    print(f"  - Batch size: {CONFIG['batch_size']}")
    print(f"  - Mixed precision: {CONFIG['use_mixed_precision']}")
    print(f"  - Torch compile: {CONFIG.get('use_torch_compile', False)}")
    print(f"  - Workers: {CONFIG['num_workers']}\n")
    
    print("Loading ViLT processor...")
    processor = ViltProcessor.from_pretrained(CONFIG['model_name'])
    
    print(f"Loading vocabulary from {CONFIG['vocab_path']}...")
    
    with open(CONFIG['vocab_path'], 'r') as f:
        vocab_data = json.load(f)
        answer_to_label = vocab_data['answer_to_label']
        label_to_answer = {int(k): v for k, v in vocab_data['label_to_answer'].items()}
        num_labels = vocab_data['num_labels']
    
    print(f"Vocabulary: {num_labels} classes\n")
    
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
    print(f"  - ViLT base: {CONFIG['model_name']}")
    print(f"  - RadBERT: {CONFIG['radbert_name']}")
    
    vilt_config = ViltConfig.from_pretrained(CONFIG['model_name'])
    vilt_model = ViltForQuestionAnswering.from_pretrained(
        CONFIG['model_name'], 
        config=vilt_config, 
        ignore_mismatched_sizes=True
    )
    
    if CONFIG['baseline_checkpoint'] and os.path.exists(CONFIG['baseline_checkpoint']):
        print(f"  - Loading baseline from {CONFIG['baseline_checkpoint']}")
        checkpoint = torch.load(CONFIG['baseline_checkpoint'], map_location='cpu')
        state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if not k.startswith('classifier')}
        vilt_model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Baseline encoder loaded")
    else:
        print(f"  ⚠ Baseline checkpoint not found at {CONFIG['baseline_checkpoint']}")
        print(f"  → Training from scratch")
    
    print(f"  - Loading RadBERT from {CONFIG['radbert_name']}...")
    radbert = AutoModel.from_pretrained(CONFIG['radbert_name'])
    
    print("  ✓ RadBERT loaded")
    print("\n  Creating hybrid model...")
    model = ViltWithRadBERT(vilt_model, radbert, num_labels, CONFIG['freeze_vision_encoder']).to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters: {total_params - trainable:,}\n")
    
    if CONFIG.get('use_torch_compile', False) and hasattr(torch, 'compile'):
        print("Applying torch.compile optimization...")
        model = torch.compile(model)
        print("  ✓ torch.compile enabled\n")
    
    optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=CONFIG["use_mixed_precision"])
    
    best_accuracy = 0.0
    patience_counter = 0
    history = []
    
    print(f"{'='*60}")
    print(f"Starting training for {CONFIG['epochs']} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{CONFIG['epochs']}\n{'='*60}")
        
        train_loss, train_acc = train_one_epoch(model, train_dataloader, optimizer, scaler, device, epoch, CONFIG)
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
            
            checkpoint_full_path = os.path.join(CONFIG['checkpoint_dir'], CONFIG['checkpoint_path'])
            
            vocab_dict = {
                'answer_to_label': answer_to_label,
                'label_to_answer': label_to_answer,
                'num_labels': num_labels
            }
            
            save_complete_checkpoint(
                model,
                optimizer,
                scaler,
                epoch,
                {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_accuracy': best_accuracy
                },
                CONFIG,
                vocab_dict,
                checkpoint_full_path
            )
            
            print(f"\n  ✓ NEW BEST MODEL SAVED!")
            print(f"    Accuracy: {val_acc:.4f}")
            print(f"    Saved to: {checkpoint_full_path}")
        else:
            patience_counter += 1
            print(f"\n  No improvement. Patience: {patience_counter}/{CONFIG['patience']}")
        
        if CONFIG.get('interactive_mode', False) and (epoch + 1) % CONFIG.get('check_interval', 5) == 0:
            print(f"\n{'='*60}")
            print(f"Checkpoint: {epoch+1} epochs completed")
            print(f"Best accuracy so far: {best_accuracy:.4f}")
            user_input = input("Continue training? (y/n or press Enter for yes): ").strip().lower()
            if user_input in ['n', 'no', 'stop']:
                print("Training stopped by user.")
                break
            print(f"{'='*60}\n")
        
        if patience_counter >= CONFIG['patience']:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"{'='*60}")
            break
    
    history_path = os.path.join(CONFIG['results_dir'], 'training_history_radbert.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    plots_dir = CONFIG.get('plots_dir', CONFIG['results_dir'])
    os.makedirs(plots_dir, exist_ok=True)
    plot_final_training_curves(history, plots_dir, "radbert")
    
    summary_path = os.path.join(CONFIG['results_dir'], 'training_summary_radbert.json')
    summary = {
        'model': 'ViLT-RadBERT',
        'best_val_accuracy': best_accuracy,
        'total_epochs': len(history),
        'config': CONFIG,
        'final_results': history[-1] if history else None
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to: {summary_path}")
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"Model checkpoint: {os.path.join(CONFIG['checkpoint_dir'], CONFIG['checkpoint_path'])}")
    print(f"Results directory: {CONFIG['results_dir']}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
