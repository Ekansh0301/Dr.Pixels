import os
import json
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
from transformers import logging as transformers_logging
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from collections import Counter
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

transformers_logging.set_verbosity_error()
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "model_name": "dandelin/vilt-b32-finetuned-vqa",
    "data_dir": BASE_DIR,
    "checkpoint_dir": os.path.join(BASE_DIR, "vqa_vilt_baseline_outputs", "checkpoints"),
    "checkpoint_path": "vilt_baseline_best.pth",
    "results_dir": os.path.join(BASE_DIR, "vqa_vilt_baseline_outputs", "results"),
    "plots_dir": os.path.join(BASE_DIR, "vqa_vilt_baseline_outputs", "plots"),
    "epochs": 30,
    "lr": 3e-5,
    "batch_size": 24,
    "use_mixed_precision": True,
    "use_torch_compile": True,
    "enable_tf32": True,
    "num_workers": 8,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 200,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "patience": 5,
    "min_answer_frequency": 2,
    "seed": 42,
    "pin_memory": True,
    "interactive_mode": True,
    "check_interval": 5,
}

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
        
        print(f"[{split.upper()}] Loaded {len(self.qa_pairs)} QA pairs with {self.num_labels} unique answers")
    
    def _load_qa_pairs(self):
        qa_pairs = []
        if not os.path.exists(self.qa_file):
            raise FileNotFoundError(f"QA file not found: {self.qa_file}")
        
        with open(self.qa_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split('|')
                    
                    if len(parts) == 3:
                        img_id, question, answer = parts
                    elif len(parts) == 4:
                        img_id, category, question, answer = parts
                    else:
                        continue
                    
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
                except Exception as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    continue
        
        return qa_pairs
    
    def _build_answer_vocab(self, min_freq):
        all_answers = [pair['answer'] for pair in self.qa_pairs]
        answer_counts = Counter(all_answers)
        
        filtered_answers = [ans for ans, count in answer_counts.items() if count >= min_freq]
        filtered_answers = sorted(filtered_answers)
        
        answer_to_label = {answer: i for i, answer in enumerate(filtered_answers)}
        label_to_answer = {i: answer for answer, i in answer_to_label.items()}
        num_labels = len(filtered_answers)
        
        print(f"Built vocabulary: {num_labels} answers (min_freq={min_freq})")
        return answer_to_label, label_to_answer, num_labels
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def _augment_image(self, image):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        if random.random() < 0.3:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() < 0.3:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        
        return image
    
    def __getitem__(self, idx):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                sample = self.qa_pairs[idx]
                question = sample['question']
                answer = sample['answer']
                
                if answer not in self.answer_to_label:
                    idx = (idx + 1) % len(self)
                    continue
                
                label_index = self.answer_to_label[answer]
                
                image = Image.open(sample['image_path']).convert('RGB')
                
                image = image.resize((384, 384), Image.BILINEAR)
                
                if self.augment:
                    image = self._augment_image(image)
                
                encoding = self.processor(
                    images=image,
                    text=question,
                    padding="max_length",
                    truncation=True,
                    max_length=40,
                    return_tensors="pt"
                )
                
                result = {}
                for k, v in encoding.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v.squeeze(0).clone()
                    else:
                        result[k] = torch.tensor(v).squeeze(0)
                
                result['labels'] = torch.tensor(label_index, dtype=torch.long)
                
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error loading sample {idx} after {max_retries} attempts: {e}")
                    return self._get_dummy_sample()
                idx = (idx + 1) % len(self)
    
    def _get_dummy_sample(self):
        dummy_image = Image.new('RGB', (384, 384), color='white')
        dummy_encoding = self.processor(
            images=dummy_image,
            text="dummy question",
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt"
        )
        result = {k: v.squeeze(0).clone() for k, v in dummy_encoding.items()}
        result['labels'] = torch.tensor(0, dtype=torch.long)
        return result

def custom_collate_fn(batch):
    if len(batch) == 0:
        return {}
    
    keys = batch[0].keys()
    
    collated = {}
    for key in keys:
        try:
            if key == 'labels':
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                tensors = []
                for item in batch:
                    if isinstance(item[key], torch.Tensor):
                        tensors.append(item[key].clone())
                    else:
                        tensors.append(torch.tensor(item[key]))
                
                if len(tensors) > 0:
                    first_shape = tensors[0].shape
                    for i, t in enumerate(tensors):
                        if t.shape != first_shape:
                            print(f"Warning: Shape mismatch for key '{key}' at index {i}: {t.shape} vs {first_shape}")
                            raise ValueError(f"Shape mismatch in batch for key {key}")
                    
                    collated[key] = torch.stack(tensors)
        except Exception as e:
            print(f"Error collating key '{key}': {e}")
            raise
    
    return collated

def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()
    
    criterion = torch.nn.CrossEntropyLoss()
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", position=0, leave=True)
    
    for step, batch in enumerate(progress_bar):
        labels = batch.pop('labels').to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config["use_mixed_precision"]):
            base_model = model
            
            outputs = base_model.vilt(**batch)
            pooled_output = outputs.pooler_output
            logits = base_model.classifier(pooled_output)
            
            loss = criterion(logits, labels) / config["gradient_accumulation_steps"]
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config["gradient_accumulation_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config["gradient_accumulation_steps"]
        
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f"{loss.item() * config['gradient_accumulation_steps']:.4f}",
            'acc': f"{correct/total:.4f}"
        })
    
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, device, label_to_answer=None, split_name="Val"):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"Evaluating {split_name}", position=0, leave=True)
        
        for batch in progress_bar:
            labels = batch.pop('labels').to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=CONFIG["use_mixed_precision"]):
                base_model = model
                
                outputs = base_model.vilt(**batch)
                pooled_output = outputs.pooler_output
                logits = base_model.classifier(pooled_output)
                
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct_predictions,
        'total': total_samples,
        'predictions': all_predictions,
        'labels': all_labels
    }

def plot_final_training_curves(history, save_dir, model_name="baseline"):
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o')
    ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training and Validation Loss - {model_name}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2, marker='o')
    ax2.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Training and Validation Accuracy - {model_name}', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'final_training_curves_{model_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Final training curves saved to: {save_path}")
    return save_path

def save_complete_checkpoint(model, optimizer, scaler, epoch, metrics, config, vocab_dict, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'metrics': metrics,
        'config': config,
        'vocabulary': vocab_dict,
        'model_config': {
            'model_name': config['model_name'],
            'num_labels': vocab_dict['num_labels'],
            'hidden_size': 768
        },
        'timestamp': str(torch.cuda.Event(enable_timing=False))
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    summary = {
        'epoch': epoch,
        'metrics': metrics,
        'config': config,
        'num_labels': vocab_dict['num_labels'],
    }
    summary_path = checkpoint_path.replace('.pth', '_summary.json')
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
    print(f"\nBase directory: {CONFIG['data_dir']}")
    print(f"Checkpoints: {CONFIG['checkpoint_dir']}")
    print(f"Results: {CONFIG['results_dir']}")
    print(f"\nOptimizations:")
    print(f"  - Batch size: {CONFIG['batch_size']}")
    print(f"  - Mixed precision: {CONFIG['use_mixed_precision']}")
    print(f"  - Torch compile: {CONFIG.get('use_torch_compile', False)}")
    print(f"  - Workers: {CONFIG['num_workers']}")
    
    processor = ViltProcessor.from_pretrained(CONFIG['model_name'])
    
    print("\nLoading datasets...")
    
    train_dataset = VQAMedDataset(
        data_dir=CONFIG['data_dir'],
        split='train',
        processor=processor,
        min_answer_freq=CONFIG['min_answer_frequency'],
        augment=True
    )
    
    val_dataset = VQAMedDataset(
        data_dir=CONFIG['data_dir'],
        split='val',
        processor=processor,
        answer_to_label=train_dataset.answer_to_label,
        num_labels=train_dataset.num_labels
    )
    
    vocab_path = os.path.join(CONFIG['checkpoint_dir'], 'answer_vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump({
            'answer_to_label': train_dataset.answer_to_label,
            'label_to_answer': train_dataset.label_to_answer,
            'num_labels': train_dataset.num_labels
        }, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")
    
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
    
    num_labels = train_dataset.num_labels
    print(f"\nInitializing ViLT model for {num_labels} classes...")
    
    config = ViltConfig.from_pretrained(CONFIG['model_name'])
    config.num_labels = num_labels
    
    model = ViltForQuestionAnswering.from_pretrained(
        CONFIG['model_name'],
        config=config,
        ignore_mismatched_sizes=True
    ).to(device)
    
    model.classifier = torch.nn.Linear(config.hidden_size, num_labels).to(device)
    
    model.config.id2label = train_dataset.label_to_answer
    model.config.label2id = train_dataset.answer_to_label
    
    if CONFIG.get('use_torch_compile', False) and hasattr(torch, 'compile'):
        print("Applying torch.compile optimization...")
        model = torch.compile(model)
        print("  ✓ torch.compile enabled\n")
    
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=CONFIG["use_mixed_precision"])
    
    best_accuracy = 0.0
    patience_counter = 0
    training_history = []
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*60}")
        
        train_loss, train_acc = train_one_epoch(model, train_dataloader, optimizer, scaler, device, epoch, CONFIG)
        
        eval_results = evaluate(model, val_dataloader, device, train_dataset.label_to_answer, "Validation")
        
        val_accuracy = eval_results['accuracy']
        avg_val_loss = eval_results['loss']
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.4f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_accuracy
        })
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], CONFIG['checkpoint_path'])
            
            vocab_dict = {
                'answer_to_label': train_dataset.answer_to_label,
                'label_to_answer': train_dataset.label_to_answer,
                'num_labels': train_dataset.num_labels
            }
            
            save_complete_checkpoint(
                model,
                optimizer,
                scaler,
                epoch,
                {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': avg_val_loss,
                    'val_acc': val_accuracy,
                    'best_accuracy': best_accuracy
                },
                CONFIG,
                vocab_dict,
                checkpoint_path
            )
            print(f"  ✓ New best model saved! Accuracy: {val_accuracy:.4f}")
            print(f"    Checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{CONFIG['patience']}")
        
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
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    history_path = os.path.join(CONFIG['results_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    plots_dir = CONFIG.get('plots_dir', CONFIG['results_dir'])
    os.makedirs(plots_dir, exist_ok=True)
    plot_final_training_curves(training_history, plots_dir, "baseline")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print(f"Training history saved to: {history_path}")
    print(f"Best model checkpoint: {os.path.join(CONFIG['checkpoint_dir'], CONFIG['checkpoint_path'])}")
    print(f"Vocabulary saved to: {os.path.join(CONFIG['checkpoint_dir'], 'answer_vocab.json')}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()