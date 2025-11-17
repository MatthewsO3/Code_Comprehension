"""
Train UniXcoder on MLM task for C++ code
Includes early stopping, dropout, learning rate warmup, mixed precision,
and overfitting prevention techniques.
WITH COMPREHENSIVE LOSS AND PERFORMANCE TRACKING
"""
import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaForMaskedLM, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from tqdm import tqdm

try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp as tscpp
    TS_AVAILABLE = True
    CPP_LANGUAGE = Language(tscpp.language())
    ts_parser = Parser(CPP_LANGUAGE)
except ImportError:
    TS_AVAILABLE = False
    print("Warning: tree_sitter not available.")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ============================================================================
# Performance Tracker with Early Stopping
# ============================================================================

class PerformanceTracker:
    """Tracks metrics with early stopping support and batch-level logging."""
    def __init__(self, output_dir: str, patience: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.patience_counter = 0
        self.best_val_loss = float('inf')

        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_batch_losses': [],
            'val_loss': [],
            'val_batch_losses': [],
            'learning_rate': [],
            'best_val_loss': None,
            'best_epoch': None,
        }

    def log_batch(self, phase: str, loss):
        """Log individual batch metrics."""
        if phase == 'train':
            self.history['train_batch_losses'].append(loss)
        else:
            self.history['val_batch_losses'].append(loss)

    def log_epoch(self, epoch: int, phase: str, loss, lr=None):
        """Log epoch-level metrics."""
        if phase == 'train':
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(loss)
            if lr is not None:
                self.history['learning_rate'].append(lr)
        else:
            self.history['val_loss'].append(loss)

    def should_stop_early(self, val_loss: float, epoch: int) -> bool:
        """Check if training should stop early."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.history['best_val_loss'] = val_loss
            self.history['best_epoch'] = epoch
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
        return False

    def save(self):
        """Save metrics to JSON and CSV."""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Saved training history to {history_path}")

        # Save summary
        summary = self._compute_summary()
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved training summary to {summary_path}")

        try:
            import csv
            csv_path = self.output_dir / 'training_metrics.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Learning Rate'])
                for i in range(len(self.history['epoch'])):
                    writer.writerow([
                        self.history['epoch'][i],
                        self.history['train_loss'][i],
                        self.history['val_loss'][i] if i < len(self.history['val_loss']) else '',
                        self.history['learning_rate'][i] if i < len(self.history['learning_rate']) else '',
                    ])
            print(f"✓ Saved metrics CSV to {csv_path}")
        except Exception as e:
            print(f"⚠️ Could not save CSV: {e}")

    def _compute_summary(self) -> Dict:
        """Compute summary statistics."""
        return {
            'total_epochs': len(self.history['epoch']),
            'best_epoch': self.history['best_epoch'],
            'best_val_loss': self.history['best_val_loss'],
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'min_train_loss': min(self.history['train_loss']) if self.history['train_loss'] else None,
            'min_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else None,
            'total_batches_train': len(self.history['train_batch_losses']),
            'total_batches_val': len(self.history['val_batch_losses']),
            'avg_batch_loss_train': np.mean(self.history['train_batch_losses']) if self.history['train_batch_losses'] else None,
            'std_batch_loss_train': np.std(self.history['train_batch_losses']) if self.history['train_batch_losses'] else None,
        }


# ============================================================================
# Dataset
# ============================================================================

class UniXcoderDataset(Dataset):
    """Dataset for UniXcoder MLM training."""
    def __init__(self, jsonl_file: str, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        print(f"Loading data from {jsonl_file}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading samples"):
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.convert_sample_to_features(self.samples[idx])

    def convert_sample_to_features(self, sample: Dict) -> Dict:
        """Convert code to features."""
        code_tokens = sample['code_tokens']

        if len(code_tokens) > self.max_length - 2:
            code_tokens = code_tokens[:self.max_length - 2]

        tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding_len = self.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
        attention_mask.extend([0] * padding_len)

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
        }


@dataclass
class MLMCollator:
    """Data collator for MLM task."""
    tokenizer: RobertaTokenizer
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attention_mask = torch.stack([ex['attention_mask'] for ex in examples])

        labels = input_ids.clone()
        masked_ids = input_ids.clone()

        for i in range(len(examples)):
            special_tokens_mask = [
                1 if token_id in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                                  self.tokenizer.pad_token_id] else 0
                for token_id in input_ids[i].tolist()
            ]

            maskable_positions = [
                pos for pos, is_special in enumerate(special_tokens_mask)
                if not is_special
            ]

            if len(maskable_positions) == 0:
                continue

            num_mask = max(1, int(len(maskable_positions) * self.mlm_probability))
            mask_positions = random.sample(maskable_positions, min(num_mask, len(maskable_positions)))

            for pos in mask_positions:
                rand = random.random()
                if rand < 0.8:
                    masked_ids[i, pos] = self.tokenizer.mask_token_id
                elif rand < 0.9:
                    masked_ids[i, pos] = random.randint(0, self.tokenizer.vocab_size - 1)

            mask_indicator = torch.zeros_like(labels[i], dtype=torch.bool)
            mask_indicator[mask_positions] = True
            labels[i, ~mask_indicator] = -100

        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': masked_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# ============================================================================
# Training and Validation
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, tracker: PerformanceTracker, scaler, use_amp=False):
    """Train for one epoch with mixed precision support and loss tracking."""
    model.train()
    total_loss = 0
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        batch_count += 1

        # Log batch metrics
        tracker.log_batch('train', batch_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update progress bar
        avg_loss = total_loss / batch_count
        progress_bar.set_postfix({'loss': f'{batch_loss:.4f}', 'avg': f'{avg_loss:.4f}', 'lr': f'{current_lr:.2e}'})

    return total_loss / batch_count


def validate(model, dataloader, device, tracker: PerformanceTracker, use_amp=False):
    """Validate the model with loss tracking."""
    model.eval()
    total_loss = 0
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for batch in progress_bar:
            if use_amp:
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        labels=batch['labels'].to(device)
                    )
            else:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=batch['labels'].to(device)
                )

            batch_loss = outputs.loss.item()
            total_loss += batch_loss
            batch_count += 1

            # Log batch metrics
            tracker.log_batch('val', batch_loss)

            # Update progress bar
            avg_loss = total_loss / batch_count
            progress_bar.set_postfix({'loss': batch_loss, 'avg': avg_loss})

    return total_loss / batch_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train UniXcoder on MLM for C++')
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--mlm_probability', type=float, default=None)
    parser.add_argument('--validation_split', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')

    config = {}
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            full_config = json.load(f)
            config = full_config.get("train", {})

    parser.set_defaults(**config)
    args = parser.parse_args()

    if not args.data_file:
        parser.error("data_file must be specified in config.json or via arguments.")

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
        use_amp = False
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
        use_amp = args.use_amp
    else:
        device = torch.device("cpu")
        print("Using CPU")
        use_amp = False

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tracker = PerformanceTracker(args.output_dir, patience=args.early_stopping_patience)

    print("Loading UniXcoder base-nine...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
    model = RobertaForMaskedLM.from_pretrained("microsoft/unixcoder-base-nine")

    # Add dropout for regularization
    model.config.hidden_dropout_prob = 0.2
    model.config.attention_probs_dropout_prob = 0.2

    model = model.to(device)
    print("✓ Loaded UniXcoder base-nine with dropout regularization")

    # Load and split dataset
    full_dataset = UniXcoderDataset(args.data_file, tokenizer, args.max_length)
    val_size = int(args.validation_split * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [len(full_dataset) - val_size, val_size]
    )

    collator = MLMCollator(tokenizer, mlm_probability=args.mlm_probability)
    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        collate_fn=collator,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )

    # Optimizer with weight decay (L2 regularization)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dl) * args.epochs
    )

    scaler = GradScaler() if use_amp else None

    print("\n--- Training Configuration ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  use_amp: {use_amp}")
    print(f"  device: {device}")
    print("------------------------------\n")

    for epoch in range(args.epochs):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 70}")

        train_loss = train_epoch(model, train_dl, optimizer, scheduler, device, tracker, scaler, use_amp)
        val_loss = validate(model, val_dl, device, tracker, use_amp)

        current_lr = optimizer.param_groups[0]['lr']
        tracker.log_epoch(epoch, 'train', train_loss, current_lr)
        tracker.log_epoch(epoch, 'val', val_loss)

        print(f"\n{'─' * 70}")
        print(f"Epoch {epoch + 1} Results:")
        print(f"  Train Loss:     {train_loss:.6f}")
        print(f"  Val Loss:       {val_loss:.6f}")
        print(f"  Learning Rate:  {current_lr:.6e}")
        print(f"  Best Val Loss:  {tracker.best_val_loss:.6f} (Epoch {tracker.history['best_epoch'] + 1 if tracker.history['best_epoch'] is not None else 'N/A'})")
        print(f"  Patience:       {tracker.patience_counter}/{args.early_stopping_patience}")
        print(f"{'─' * 70}")

        if val_loss < tracker.best_val_loss:
            checkpoint_path = Path(args.output_dir) / "best_model"
            print(f"\n✓ New best model! Saving to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

        if tracker.should_stop_early(val_loss, epoch):
            print(f"\n⚠️  Early stopping triggered!")
            print(f"   No improvement for {args.early_stopping_patience} epochs")
            print(f"   Best loss: {tracker.best_val_loss:.6f} at epoch {tracker.history['best_epoch'] + 1}")
            break

    print(f"\n{'=' * 60}")
    print(f"Training completed!")
    print(f"Best val loss: {tracker.best_val_loss:.4f} at epoch {tracker.history['best_epoch']}")
    print(f"{'=' * 60}\n")

    # Save all performance metrics
    print("="*60)
    print("SAVING PERFORMANCE METRICS...")
    print("="*60)
    tracker.save()


if __name__ == "__main__":
    main()