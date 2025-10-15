"""
UniXcoder-style Masked Language Modeling (MLM) Training for C++
Streaming version with custom MLM head and sample limit
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    get_linear_schedule_with_warmup,

)
from torch.optim import AdamW
from tqdm import tqdm
import random
import numpy as np
from pathlib import Path
import itertools
import logging


# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- Reproducibility ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# ---------------- Dataset ----------------
class CPPCodeDataset(Dataset):
    """Dataset for C++ code with MLM masking"""

    def __init__(self, dataset_iter, tokenizer, max_length=512, mlm_probability=0.15):
        # dataset_iter is a list of dicts when sampled from stream
        self.samples = list(dataset_iter)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code = self.samples[idx]['code']

        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # UniXcoder masking rule: 80% mask, 10% random, 10% unchanged
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# ---------------- Model ----------------
class UniXcoderMLM(nn.Module):
    """
    UniXcoder + Custom MLM Head
    """

    def __init__(self, base_model_name, vocab_size, hidden_size, device):
        super().__init__()
        self.device = device

        # Load pretrained UniXcoder encoder
        self.encoder = RobertaModel.from_pretrained(base_model_name)

        # Custom MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        ).to(device)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        logits = self.mlm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {'loss': loss, 'logits': logits}


# ---------------- Collate Function ----------------
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


# ---------------- Training & Evaluation ----------------
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, grad_accum_steps):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss'] / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

        progress_bar.set_postfix({
            'loss': loss.item() * grad_accum_steps,
            'avg_loss': total_loss / (batch_idx + 1),
            'lr': scheduler.get_last_lr()[0]
        })

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, labels)
            total_loss += outputs['loss'].item()

    return total_loss / len(dataloader)


# ---------------- Main ----------------
def main():
    config = {
        'model_name': 'microsoft/unixcoder-base',
        'max_length': 512,
        'batch_size': 1,
        'learning_rate': 2e-4,
        'num_epochs': 1,
        'warmup_steps': 10,
        'mlm_probability': 0.15,
        'max_samples': 1000,  # how many samples to stream
        'save_dir': 'UnixCoderCPP/unixcoder_cpp_mlm',
        'gradient_accumulation_steps': 4,
    }

    if torch.backends.mps.is_available():
        device = torch.device("mps")  # ✅ Apple Silicon GPU (Metal)
        logger.info("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # ✅ NVIDIA GPU
        logger.info("Using CUDA GPU")
    else:
        device = torch.device("cpu")  # ✅ CPU fallback
        logger.info("Using CPU")

    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(config['model_name'])

    logger.info("Streaming dataset...")
    stream = load_dataset(
        "codeparrot/github-code-clean",
        "C++-all",
        split='train',
        streaming=True

    )

    # Sample first N examples from the stream
    sampled_data = list(itertools.islice(stream, config['max_samples']))
    logger.info(f"Streamed and sampled {len(sampled_data)} examples")

    # Split manually (95% train, 5% val)
    split_idx = int(0.95 * len(sampled_data))
    train_samples = sampled_data[:split_idx]
    val_samples = sampled_data[split_idx:]

    train_data = CPPCodeDataset(train_samples, tokenizer, config['max_length'], config['mlm_probability'])
    val_data = CPPCodeDataset(val_samples, tokenizer, config['max_length'], config['mlm_probability'])

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0)

    logger.info("Initializing model...")
    base_model = RobertaModel.from_pretrained(config['model_name'])
    hidden_size = base_model.config.hidden_size

    model = UniXcoderMLM(config['model_name'], vocab_size=len(tokenizer), hidden_size=hidden_size, device=device)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01,betas=(0.9, 0.999), eps=1e-8)
    total_steps = len(train_loader) * config['num_epochs'] // config['gradient_accumulation_steps']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=total_steps)

    logger.info("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch + 1, config['gradient_accumulation_steps'])
        logger.info(f"Train Loss: {train_loss:.4f}")

        val_loss = evaluate(model, val_loader, device)
        logger.info(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best model! Saving to {config['save_dir']}")
            torch.save(model.state_dict(), f"{config['save_dir']}/best_model.pt")
            tokenizer.save_pretrained(f"{config['save_dir']}/tokenizer")

            torch.save({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, f"{config['save_dir']}/training_info.pt")

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
