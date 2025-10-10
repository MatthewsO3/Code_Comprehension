"""
Train GraphCodeBERT on MLM task with DFG for C++ code
"""

import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW  # Changed: Import from torch.optim instead
from tqdm import tqdm
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class GraphCodeBERTDataset(Dataset):
    """Dataset for GraphCodeBERT with DFG"""

    def __init__(self, data_dir: str, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Load all processed chunks
        data_path = Path(data_dir)
        chunk_files = sorted(data_path.glob("processed_chunk_*.pkl"))

        print(f"Loading {len(chunk_files)} chunks...")
        for chunk_file in tqdm(chunk_files):
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
                self.samples.extend(chunk_data)

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        input_ids = sample['input_ids'][:self.max_length]
        position_idx = sample['position_idx'][:self.max_length]
        dfg_to_code = sample['dfg_to_code']

        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
        position_idx = position_idx + [0] * padding_length

        # Create attention mask
        attention_mask = [1] * len(sample['input_ids'][:self.max_length]) + [0] * padding_length

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'position_idx': torch.tensor(position_idx, dtype=torch.long),
            'dfg_to_code': dfg_to_code  # Keep as list for now
        }


@dataclass
class MLMCollator:
    """Data collator for masked language modeling with DFG"""

    tokenizer: RobertaTokenizer
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        # Stack inputs
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attention_mask = torch.stack([ex['attention_mask'] for ex in examples])

        # Create labels for MLM (clone input_ids)
        labels = input_ids.clone()

        # Create masked input_ids
        masked_input_ids = input_ids.clone()

        # Mask tokens
        batch_size, seq_length = input_ids.shape

        for i in range(batch_size):
            # Get non-padding positions
            non_pad_positions = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]

            # Calculate number of tokens to mask
            num_to_mask = max(1, int(len(non_pad_positions) * self.mlm_probability))

            # Randomly select positions to mask
            mask_positions = non_pad_positions[torch.randperm(len(non_pad_positions))[:num_to_mask]]

            for pos in mask_positions:
                prob = random.random()
                if prob < 0.8:
                    # 80%: Replace with [MASK]
                    masked_input_ids[i, pos] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    # 10%: Replace with random token
                    masked_input_ids[i, pos] = random.randint(0, len(self.tokenizer) - 1)
                # 10%: Keep original token

            # Set labels to -100 for non-masked positions (ignore in loss)
            labels[i, ~torch.isin(torch.arange(seq_length), mask_positions)] = -100

        # Set padding positions to -100 in labels
        labels[attention_mask == 0] = -100

        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train GraphCodeBERT on MLM')
    parser.add_argument('--data_dir', type=str, default='.GraphCodeBert/processed_data',
                        help='Directory with processed data')
    parser.add_argument('--output_dir', type=str, default='.GraphCodeBert/graphcodebert-cpp-mlm',
                        help='Output directory for model checkpoints')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--warmup_steps', type=int, default=10,
                        help='Warmup steps for scheduler')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--mlm_probability', type=float, default=0.15,
                        help='Probability of masking tokens')

    args = parser.parse_args()

    # Setup
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = RobertaForMaskedLM.from_pretrained("microsoft/graphcodebert-base")
    model.to(device)

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Load dataset
    print("Loading dataset...")
    dataset = GraphCodeBERTDataset(args.data_dir, tokenizer, args.max_length)

    # Split into train/val
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create data collator
    data_collator = MLMCollator(tokenizer, mlm_probability=args.mlm_probability)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(args.epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 50}")

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss = validate(model, val_dataloader, device)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_path / "best_model"
            print(f"Saving best model to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

        # Save epoch checkpoint
        epoch_path = output_path / f"checkpoint-epoch-{epoch + 1}"
        model.save_pretrained(epoch_path)
        tokenizer.save_pretrained(epoch_path)

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()