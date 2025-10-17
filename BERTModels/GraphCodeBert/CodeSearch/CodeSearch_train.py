# train_codesearch.py

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from collections import defaultdict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


set_seed(42)


def dfg_to_adj(dfg, node_map):
    adj = defaultdict(list)
    for _, use_pos, _, _, dep_pos_list in dfg:
        for def_pos in dep_pos_list:
            if use_pos in node_map and def_pos in node_map:
                adj[node_map[use_pos]].append(node_map[def_pos])
    return adj


class CodeSearchDataset(Dataset):
    def __init__(self, jsonl_file: str):
        self.samples = [json.loads(line) for line in open(jsonl_file, 'r', encoding='utf-8')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CodeSearchCollator:
    def __init__(self, tokenizer, max_code_len, max_query_len):
        self.tokenizer = tokenizer
        self.max_code_len = max_code_len
        self.max_query_len = max_query_len

    def _process_item(self, tokens, dfg, max_len):
        # Process Code
        if dfg:
            # Recreate node map based on token positions
            node_map = {pos: i for i, (_, pos) in enumerate(dfg_nodes)} if (
                dfg_nodes := [(var, p) for var, p, _, _, deps in dfg for p in [p] + deps]) else {}
            adj = dfg_to_adj(dfg, node_map)

            max_code_tok_len = max_len - len(node_map) - 3
            if len(tokens) > max_code_tok_len:
                tokens = tokens[:max_code_tok_len]

            # Re-filter node_map for truncated tokens
            node_map = {pos: i for i, (_, pos) in enumerate(dfg_nodes) if pos < len(tokens)}

            # Prepare input with DFG nodes
            input_tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            dfg_start = len(input_tokens)
            input_tokens.extend([self.tokenizer.unk_token] * len(node_map))
            input_tokens.append(self.tokenizer.sep_token)

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            position_ids = list(range(len(tokens) + 2)) + [0] * len(node_map) + [len(tokens) + 2]

            attn_mask = np.zeros((max_len, max_len), dtype=bool)
            code_len = len(tokens) + 2
            attn_mask[:code_len, :code_len] = True
            for i in range(len(input_tokens)): attn_mask[i, i] = True

            for code_pos, node_idx in node_map.items():
                attn_mask[dfg_start + node_idx, code_pos + 1] = True
                attn_mask[code_pos + 1, dfg_start + node_idx] = True

            for u, adjs in adj.items():
                for v in adjs:
                    if u in node_map and v in node_map:
                        attn_mask[dfg_start + u, dfg_start + v] = True
                        attn_mask[dfg_start + v, dfg_start + u] = True
        else:  # Process Query (no DFG)
            if len(tokens) > max_len - 2:
                tokens = tokens[:max_len - 2]
            input_tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            position_ids = list(range(len(input_ids)))
            attn_mask = np.ones((max_len, max_len), dtype=bool)  # Full attention for queries

        # Padding
        padding_len = max_len - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
        position_ids.extend([0] * padding_len)

        return torch.tensor(input_ids), torch.tensor(attn_mask), torch.tensor(position_ids)

    def __call__(self, batch):
        code_batch = []
        query_batch = []
        for sample in batch:
            code_tokens = sample['code_tokens']
            dfg = sample.get('dataflow_graph', [])
            query_tokens = sample['docstring_tokens']

            code_batch.append(self._process_item(code_tokens, dfg, self.max_code_len))
            query_batch.append(self._process_item(query_tokens, [], self.max_query_len))

        return {
            "code_ids": torch.stack([b[0] for b in code_batch]),
            "code_mask": torch.stack([b[1] for b in code_batch]),
            "code_pos": torch.stack([b[2] for b in code_batch]),
            "query_ids": torch.stack([b[0] for b in query_batch]),
            "query_mask": torch.stack([b[1] for b in query_batch]),
            "query_pos": torch.stack([b[2] for b in query_batch]),
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    cos = nn.CosineSimilarity(dim=1)
    loss_fn = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        code_vecs = model(
            input_ids=batch['code_ids'].to(device),
            attention_mask=batch['code_mask'].to(device),
            position_ids=batch['code_pos'].to(device)
        ).pooler_output

        query_vecs = model(
            input_ids=batch['query_ids'].to(device),
            attention_mask=batch['query_mask'].to(device),
            position_ids=batch['query_pos'].to(device)
        ).pooler_output

        # Multiple Negatives Ranking Loss
        scores = cos(query_vecs.unsqueeze(1), code_vecs.unsqueeze(0))
        labels = torch.arange(len(scores)).to(device)
        loss = loss_fn(scores, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    config = json.load(open('config.json')).get('codesearch')

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = RobertaTokenizer.from_pretrained(config['mlm_model_path'])
    model = RobertaModel.from_pretrained(config['mlm_model_path']).to(device)

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CodeSearchDataset(config['processed_data_file'])
    collator = CodeSearchCollator(tokenizer, config['max_code_len'], config['max_query_len'])

    train_dl = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100,
        num_training_steps=len(train_dl) * config['epochs']
    )

    for epoch in range(config['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['epochs']} ---")
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, device)
        print(f"Training Loss: {train_loss:.4f}")

        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}"
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == '__main__':
    main()