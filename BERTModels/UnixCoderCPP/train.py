"""
Train UniXcoder on MLM task for C++ code.
Based on the UniXcoder paper - uses unified architecture with mask attention.
Supports config.json for settings.
Includes AST (Abstract Syntax Tree) extraction using tree-sitter.
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
from tqdm import tqdm

try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp as tscpp
    TS_AVAILABLE = True
    CPP_LANGUAGE = Language(tscpp.language())
    ts_parser = Parser(CPP_LANGUAGE)
except ImportError:
    TS_AVAILABLE = False
    print("Warning: tree_sitter not available. AST extraction will fail.")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ============================================================================
# Performance Tracker
# ============================================================================

class PerformanceTracker:
    """Tracks and saves all performance metrics during training."""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_batch_losses': [],  # All batch losses
            'val_loss': [],
            'val_batch_losses': [],  # All batch losses
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

    def update_best(self, val_loss, epoch):
        """Update best validation loss."""
        if self.history['best_val_loss'] is None or val_loss < self.history['best_val_loss']:
            self.history['best_val_loss'] = val_loss
            self.history['best_epoch'] = epoch
            return True
        return False

    def save(self):
        """Save all metrics to JSON files."""
        # Save detailed history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Saved training history to {history_path}")

        # Save summary statistics
        summary = self._compute_summary()
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved training summary to {summary_path}")

        # Save CSV for easy plotting
        self._save_csv()

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
        }

    def _save_csv(self):
        """Save epoch-level metrics as CSV."""
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


# ============================================================================
# AST Extraction
# ============================================================================

def extract_ast_sequence(tree) -> List[str]:
    """
    Extract AST node sequence from tree-sitter parse tree.
    Performs DFS traversal and collects node types.
    """
    if not TS_AVAILABLE:
        return []

    ast_nodes = []

    def traverse(node):
        ast_nodes.append(node.type)
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return ast_nodes


# ============================================================================
# Dataset
# ============================================================================

class UniXcoderDataset(Dataset):
    """
    Dataset for UniXcoder MLM training.
    Unlike GraphCodeBERT, UniXcoder doesn't require DFG preprocessing.
    Includes AST extraction using tree-sitter.
    """
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
        """
        Convert code to input features for UniXcoder.
        UniXcoder uses simpler preprocessing than GraphCodeBERT.
        Includes AST from sample if available.
        """
        code_tokens = sample['code_tokens']
        ast = sample.get('ast', [])

        # Truncate if too long
        if len(code_tokens) > self.max_length - 2:  # -2 for CLS and SEP
            code_tokens = code_tokens[:self.max_length - 2]

        # Add special tokens: [CLS] code [SEP]
        tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]

        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Create attention mask (all 1s for UniXcoder encoder mode)
        # UniXcoder uses different mask patterns for encoder/decoder modes
        # For MLM (encoder mode), all tokens attend to each other
        attention_mask = [1] * len(input_ids)

        # Padding
        padding_len = self.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
        attention_mask.extend([0] * padding_len)

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'ast': ast  # Include AST for potential future use
        }


@dataclass
class MLMCollator:
    """
    Data collator for MLM task.
    Masks tokens according to BERT-style MLM.
    """
    tokenizer: RobertaTokenizer
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attention_mask = torch.stack([ex['attention_mask'] for ex in examples])

        # Prepare labels and masked input
        labels = input_ids.clone()
        masked_ids = input_ids.clone()

        for i in range(len(examples)):
            # Find positions to mask (exclude special tokens and padding)
            special_tokens_mask = [
                1 if token_id in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id,
                                  self.tokenizer.pad_token_id] else 0
                for token_id in input_ids[i].tolist()
            ]

            # Get maskable positions
            maskable_positions = [
                pos for pos, is_special in enumerate(special_tokens_mask)
                if not is_special
            ]

            if len(maskable_positions) == 0:
                continue

            # Calculate number of tokens to mask
            num_mask = max(1, int(len(maskable_positions) * self.mlm_probability))

            # Randomly select positions
            mask_positions = random.sample(maskable_positions, min(num_mask, len(maskable_positions)))

            # Apply masking strategy (80% MASK, 10% random, 10% unchanged)
            for pos in mask_positions:
                rand = random.random()
                if rand < 0.8:
                    # 80% replace with [MASK]
                    masked_ids[i, pos] = self.tokenizer.mask_token_id
                elif rand < 0.9:
                    # 10% replace with random token
                    masked_ids[i, pos] = random.randint(0, self.tokenizer.vocab_size - 1)
                # 10% keep original (else clause - do nothing)

            # Set labels: -100 for non-masked tokens
            mask_indicator = torch.zeros_like(labels[i], dtype=torch.bool)
            mask_indicator[mask_positions] = True
            labels[i, ~mask_indicator] = -100

        # Set padding labels to -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': masked_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# ============================================================================
# Training and Validation
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, tracker: PerformanceTracker):
    """Train for one epoch with loss tracking."""
    model.train()
    total_loss = 0
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()

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

        total_loss += loss.item()
        batch_count += 1

        # Log batch metrics
        tracker.log_batch('train', loss.item())

        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / batch_count


def validate(model, dataloader, device, tracker: PerformanceTracker):
    """Validate the model with loss tracking."""
    model.eval()
    total_loss = 0
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )

            total_loss += outputs.loss.item()
            batch_count += 1

            # Log batch metrics
            tracker.log_batch('val', outputs.loss.item())

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

    # Load config from file
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
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create output directory and tracker
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tracker = PerformanceTracker(args.output_dir)

    # Load UniXcoder tokenizer and model
    print("Loading UniXcoder base-nine (trained on C++)...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base-nine")

    # UniXcoder base-nine doesn't have MLM head released, so we add it
    # We load the base model and add MLM head
    model = RobertaForMaskedLM.from_pretrained("microsoft/unixcoder-base-nine")
    print("✓ Loaded UniXcoder base-nine with MLM head")

    model = model.to(device)

    # Load dataset
    full_dataset = UniXcoderDataset(args.data_file, tokenizer, args.max_length)
    val_size = int(args.validation_split * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [len(full_dataset) - val_size, val_size]
    )

    # Create dataloaders
    collator = MLMCollator(tokenizer, mlm_probability=args.mlm_probability)
    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=min(4, os.cpu_count() or 1)
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=min(4, os.cpu_count() or 1)
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dl) * args.epochs
    )

    # Print configuration
    print("\n--- Training Configuration ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("------------------------------\n")

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\n{'=' * 20} Epoch {epoch + 1}/{args.epochs} {'=' * 20}")

        train_loss = train_epoch(model, train_dl, optimizer, scheduler, device, tracker)
        val_loss = validate(model, val_dl, device, tracker)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch metrics
        tracker.log_epoch(epoch, 'train', train_loss, current_lr)
        tracker.log_epoch(epoch, 'val', val_loss)

        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")

        if tracker.update_best(val_loss, epoch):
            best_val_loss = val_loss
            checkpoint_path = Path(args.output_dir) / "best_model"
            print(f"New best model found! Saving to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

    print(f"\n{'=' * 15} Training complete! Best validation loss: {best_val_loss:.4f} {'=' * 15}")

    # Save all performance metrics
    print("\n" + "="*50)
    print("SAVING PERFORMANCE METRICS...")
    print("="*50)
    tracker.save()


if __name__ == "__main__":
    main()