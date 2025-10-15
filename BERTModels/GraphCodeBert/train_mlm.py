"""
Train GraphCodeBERT on MLM task with DFG for C++ code.
Supports a structured config.json file for settings.
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
from collections import defaultdict



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


set_seed(42)


class GraphCodeBERTDataset(Dataset):
    def __init__(self, jsonl_file: str, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        print(f"Loading and processing data from {jsonl_file}...")
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
        code_tokens = sample['code_tokens']
        dfg = sample.get('dataflow_graph', [])
        adj = defaultdict(list)
        dfg_nodes, node_to_idx = [], {}
        for var, use_pos, _, _, dep_pos_list in dfg:
            if use_pos not in node_to_idx:
                node_to_idx[use_pos] = len(dfg_nodes);
                dfg_nodes.append((var, use_pos))
            for def_pos in dep_pos_list:
                if def_pos not in node_to_idx:
                    node_to_idx[def_pos] = len(dfg_nodes);
                    dfg_nodes.append((var, def_pos))
                adj[node_to_idx[use_pos]].append(node_to_idx[def_pos])

        max_code_len = self.max_length - len(dfg_nodes) - 3
        if len(code_tokens) > max_code_len: code_tokens = code_tokens[:max_code_len]

        tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        dfg_start_pos = len(tokens)
        tokens.extend([self.tokenizer.unk_token] * len(dfg_nodes))
        tokens.append(self.tokenizer.sep_token)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        position_idx = list(range(len(code_tokens) + 2)) + [0] * len(dfg_nodes) + [len(code_tokens) + 2]

        attn_mask = np.zeros((self.max_length, self.max_length), dtype=bool)
        code_len = len(code_tokens) + 2
        attn_mask[:code_len, :code_len] = True
        for i in range(len(tokens)): attn_mask[i, i] = True
        for i, (_, code_pos) in enumerate(dfg_nodes):
            # Ensure code_pos is within the bounds of original, non-subword-tokenized list
            # This check is tricky. A safer check is if the target position is within code_len
            if code_pos + 1 < code_len:
                dfg_abs = dfg_start_pos + i;
                code_abs = code_pos + 1
                attn_mask[dfg_abs, code_abs] = attn_mask[code_abs, dfg_abs] = True
        for i, adjs in adj.items():
            for j in adjs:
                u, v = dfg_start_pos + i, dfg_start_pos + j
                attn_mask[u, v] = attn_mask[v, u] = True

        padding_len = self.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
        position_idx.extend([0] * padding_len)

        return {'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(attn_mask),
                'position_idx': torch.tensor(position_idx)}


@dataclass
class MLMCollator:
    tokenizer: RobertaTokenizer
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attn_mask = torch.stack([ex['attention_mask'] for ex in examples])
        pos_idx = torch.stack([ex['position_idx'] for ex in examples])
        labels, masked_ids = input_ids.clone(), input_ids.clone()
        for i in range(len(examples)):
            code_indices = (pos_idx[i] > 1).nonzero(as_tuple=True)[0]
            if len(code_indices) > 1: code_indices = code_indices[:-1]
            if len(code_indices) == 0: continue
            num_mask = max(1, int(len(code_indices) * self.mlm_probability))
            mask_pos = code_indices[torch.randperm(len(code_indices))[:num_mask]]
            for pos in mask_pos:
                if random.random() < 0.8:
                    masked_ids[i, pos] = self.tokenizer.mask_token_id
                elif random.random() < 0.5:
                    masked_ids[i, pos] = random.randint(0, self.tokenizer.vocab_size - 1)
            mask_ind = torch.zeros_like(labels[i], dtype=torch.bool);
            mask_ind[mask_pos] = True
            labels[i, ~mask_ind] = -100
        labels[masked_ids == self.tokenizer.pad_token_id] = -100
        return {'input_ids': masked_ids, 'attention_mask': attn_mask, 'position_ids': pos_idx, 'labels': labels}


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train();
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            position_ids=batch['position_ids'].to(device),
            labels=batch['labels'].to(device)
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step();
        scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval();
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                position_ids=batch['position_ids'].to(device),
                labels=batch['labels'].to(device)
            )
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GraphCodeBERT on MLM for C++ with config file support.')
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--mlm_probability', type=float, default=None)
    parser.add_argument('--validation_split', type=float, default=None)

    config = {}
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f).get("train", {})
    parser.set_defaults(**config)
    args = parser.parse_args()
    if not args.data_file: parser.error("data_file must be specified in config.json or via arguments.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")  # ✅ Apple Silicon GPU (Metal)
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # ✅ NVIDIA GPU
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")  # ✅ CPU fallback
        print("Using CPU")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = RobertaForMaskedLM.from_pretrained("microsoft/graphcodebert-base").to(device)

    full_dataset = GraphCodeBERTDataset(args.data_file, tokenizer, args.max_length)
    val_size = int(args.validation_split * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [len(full_dataset) - val_size, val_size])

    collator = MLMCollator(tokenizer, mlm_probability=args.mlm_probability)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
                          num_workers=min(4, os.cpu_count() or 1))
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collator,
                        num_workers=min(4, os.cpu_count() or 1))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=len(train_dl) * args.epochs)

    print("\n--- Running with configuration ---")
    for k, v in vars(args).items(): print(f"  {k}: {v}")
    print("----------------------------------\n")

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\n{'=' * 20} Epoch {epoch + 1}/{args.epochs} {'=' * 20}")
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, device)
        val_loss = validate(model, val_dl, device)
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(args.output_dir) / "best_model"
            print(f"New best model found! Saving to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

    print(f"\n{'=' * 15} Training complete! Best validation loss: {best_val_loss:.4f} {'=' * 15}")


if __name__ == "__main__":
    main()

