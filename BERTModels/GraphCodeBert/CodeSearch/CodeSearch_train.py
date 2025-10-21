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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# --- Dataset class for preprocessed JSONL ---
class CodeSearchDataset(Dataset):
    def __init__(self, jsonl_file: str):
        print(f"Loading preprocessed samples from {jsonl_file}...")
        self.samples = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return already-processed data
        sample = self.samples[idx]
        return {
            'code': sample['code'],
            'docstring': sample['docstring'],
            'dfg': sample['dfg']
        }


class CodeSearchCollator:
    def __init__(self, tokenizer, max_code_len, max_query_len):
        self.tokenizer = tokenizer
        self.max_code_len = max_code_len
        self.max_query_len = max_query_len

    def _process_item(self, code, dfg, tokens, max_len, is_code=True):
        """Process code or query into model input."""
        if is_code and dfg:  # This is code with DFG
            code_tokens = self.tokenizer.tokenize(code, add_prefix_space=True)

            # Build DFG node map
            node_map = {}
            nodes_data = []
            for edge in dfg:
                use_pos = edge[1]
                dep_list = edge[4]
                for pos in [use_pos] + dep_list:
                    if pos < len(code_tokens) and pos not in node_map:
                        node_map[pos] = len(nodes_data)
                        nodes_data.append(("", pos))

            # Build adjacency list for DFG edges
            adj = defaultdict(list)
            for edge in dfg:
                use_pos = edge[1]
                dep_list = edge[4]
                for def_pos in dep_list:
                    if use_pos in node_map and def_pos in node_map:
                        adj[node_map[use_pos]].append(node_map[def_pos])

            # Truncate tokens to fit max_len
            max_code_tok_len = max_len - len(node_map) - 3
            if len(code_tokens) > max_code_tok_len:
                code_tokens = code_tokens[:max_code_tok_len]

            # Build input
            input_tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
            dfg_start = len(input_tokens)
            input_tokens.extend([self.tokenizer.unk_token] * len(node_map))
            input_tokens.append(self.tokenizer.sep_token)

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            position_ids = list(range(len(code_tokens) + 2)) + [0] * len(node_map) + [len(code_tokens) + 2]

            # Build attention mask
            attn_mask = np.zeros((max_len, max_len), dtype=bool)
            code_len = len(code_tokens) + 2
            attn_mask[:code_len, :code_len] = True
            for i in range(len(input_tokens)):
                attn_mask[i, i] = True

            # Connect code tokens to DFG nodes
            for code_pos, node_idx in node_map.items():
                if code_pos < len(code_tokens):
                    attn_mask[dfg_start + node_idx, code_pos + 1] = True
                    attn_mask[code_pos + 1, dfg_start + node_idx] = True

            # Connect DFG nodes
            for u_idx, v_indices in adj.items():
                for v_idx in v_indices:
                    attn_mask[dfg_start + u_idx, dfg_start + v_idx] = True
                    attn_mask[dfg_start + v_idx, dfg_start + u_idx] = True

            attn_mask = attn_mask.tolist()
        else:  # This is a query (no DFG)
            query_tokens = self.tokenizer.tokenize(tokens, add_prefix_space=True)
            if len(query_tokens) > max_len - 2:
                query_tokens = query_tokens[:max_len - 2]

            input_tokens = [self.tokenizer.cls_token] + query_tokens + [self.tokenizer.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            position_ids = list(range(len(input_ids)))
            attn_mask = [[True] * max_len for _ in range(max_len)]

        # Pad to max_len
        padding_len = max_len - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
        position_ids.extend([0] * padding_len)

        return torch.tensor(input_ids), torch.tensor(attn_mask), torch.tensor(position_ids)

    def __call__(self, batch):
        code_batch = []
        query_batch = []

        for sample in batch:
            code = sample['code']
            dfg = sample['dfg']
            docstring = sample['docstring']

            # Process code with DFG
            code_batch.append(self._process_item(code, dfg, code, self.max_code_len, is_code=True))

            # Process query without DFG
            query_batch.append(self._process_item("", [], docstring, self.max_query_len, is_code=False))

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

        scores = cos(query_vecs.unsqueeze(1), code_vecs.unsqueeze(0))
        labels = torch.arange(len(scores)).to(device)
        loss = loss_fn(scores, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    with open('/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/GraphCodeBert/config.json') as f:
        config = json.load(f).get('codesearch')

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = RobertaTokenizer.from_pretrained(config['mlm_model_path'])
    model = RobertaModel.from_pretrained(config['mlm_model_path']).to(device)

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed dataset
    dataset = CodeSearchDataset(config['processed_data_file'])
    collator = CodeSearchCollator(tokenizer, config['max_code_len'], config['max_query_len'])

    # Set num_workers based on device
    num_workers = 0 if device.type == 'mps' else min(4, os.cpu_count() or 1)

    train_dl = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers
    )

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dl) * config['epochs']
    )

    for epoch in range(config['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['epochs']} ---")
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, device)
        print(f"Training Loss: {train_loss:.4f}")

        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}"
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == '__main__':
    main()