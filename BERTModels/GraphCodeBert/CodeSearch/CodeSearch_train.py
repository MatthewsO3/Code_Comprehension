# ============================================================================
# CodeSearch_train.py - Training pipeline (uses original data, not encoded)
# ============================================================================
# NOTE: This file remains largely the same. Training still uses the original
# preprocessed_data_file to allow the collator to process code+DFG dynamically.
# The encoded_corpus_file is used separately for evaluation efficiency.

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


class CodeSearchDataset(Dataset):
    """Load preprocessed dataset with code, docstring, and DFG."""
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
        sample = self.samples[idx]
        return {
            'code': sample['code'],
            'docstring': sample['docstring'],
            'dfg': sample['dfg']
        }


class CodeSearchCollator:
    """Process code/query data into model inputs with DFG attention."""
    def __init__(self, tokenizer, max_code_len, max_query_len):
        self.tokenizer = tokenizer
        self.max_code_len = max_code_len
        self.max_query_len = max_query_len

    def _process_item(self, code, dfg, tokens, max_len, is_code=True):
        """Process code or query into model input."""
        if is_code and dfg:  # Code with DFG
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

            # Truncate both tokens AND DFG nodes to fit within max_len
            # Format: [CLS] code_tokens [SEP] dfg_nodes [SEP]
            # So we need: 1 + code_len + 1 + dfg_len + 1 <= max_len
            # Available space: max_len - 3
            available_space = max_len - 3

            # Start by allocating space: give priority to code tokens
            max_code_tok_len = max(available_space - len(node_map), available_space // 2)
            max_dfg_nodes = available_space - max_code_tok_len

            # Truncate code tokens
            if len(code_tokens) > max_code_tok_len:
                code_tokens = code_tokens[:max_code_tok_len]
                # Rebuild node_map after truncating code
                node_map = {}
                for edge in dfg:
                    use_pos = edge[1]
                    dep_list = edge[4]
                    for pos in [use_pos] + dep_list:
                        if pos < len(code_tokens) and pos not in node_map:
                            node_map[pos] = len(node_map)
                # Rebuild adjacency list
                adj = defaultdict(list)
                for edge in dfg:
                    use_pos = edge[1]
                    dep_list = edge[4]
                    for def_pos in dep_list:
                        if use_pos in node_map and def_pos in node_map:
                            adj[node_map[use_pos]].append(node_map[def_pos])

            # Truncate DFG nodes if needed
            if len(node_map) > max_dfg_nodes:
                # Keep only the first max_dfg_nodes
                old_to_new = {}
                new_node_map = {}
                idx = 0
                for old_pos in sorted(node_map.keys()):
                    if idx < max_dfg_nodes:
                        old_to_new[node_map[old_pos]] = idx
                        new_node_map[old_pos] = idx
                        idx += 1
                node_map = new_node_map
                # Update adjacency list
                new_adj = defaultdict(list)
                for u_idx, v_indices in adj.items():
                    if u_idx in old_to_new:
                        for v_idx in v_indices:
                            if v_idx in old_to_new:
                                new_adj[old_to_new[u_idx]].append(old_to_new[v_idx])
                adj = new_adj

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
            # Cap at max_len to avoid index out of bounds
            for i in range(min(len(input_tokens), max_len)):
                attn_mask[i, i] = True

            # Connect code tokens to DFG nodes
            for code_pos, node_idx in node_map.items():
                if code_pos < len(code_tokens):
                    dfg_idx = dfg_start + node_idx
                    code_idx = code_pos + 1
                    # Only connect if within bounds
                    if dfg_idx < max_len and code_idx < max_len:
                        attn_mask[dfg_idx, code_idx] = True
                        attn_mask[code_idx, dfg_idx] = True

            # Connect DFG nodes
            for u_idx, v_indices in adj.items():
                for v_idx in v_indices:
                    u_pos = dfg_start + u_idx
                    v_pos = dfg_start + v_idx
                    # Only connect if within bounds
                    if u_pos < max_len and v_pos < max_len:
                        attn_mask[u_pos, v_pos] = True
                        attn_mask[v_pos, u_pos] = True

            # Ensure attn_mask is always max_len x max_len
            attn_mask_array = np.array(attn_mask, dtype=bool)
            if attn_mask_array.shape[0] != max_len or attn_mask_array.shape[1] != max_len:
                # Pad attention mask to max_len x max_len
                padded_mask = np.zeros((max_len, max_len), dtype=bool)
                actual_len = min(attn_mask_array.shape[0], max_len)
                padded_mask[:actual_len, :actual_len] = attn_mask_array[:actual_len, :actual_len]
                attn_mask = padded_mask.tolist()
            else:
                attn_mask = attn_mask_array.tolist()

        else:  # Query without DFG
            query_tokens = self.tokenizer.tokenize(tokens, add_prefix_space=True)
            if len(query_tokens) > max_len - 2:
                query_tokens = query_tokens[:max_len - 2]

            input_tokens = [self.tokenizer.cls_token] + query_tokens + [self.tokenizer.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            position_ids = list(range(len(input_ids)))
            attn_mask = [[True] * max_len for _ in range(max_len)]

        # Pad to max_len
        padding_len = max_len - len(input_ids)
        if padding_len < 0:
            # Should not happen now, but just in case, truncate
            input_ids = input_ids[:max_len]
            position_ids = position_ids[:max_len]
            padding_len = 0
        else:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
            position_ids.extend([0] * padding_len)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.bool), torch.tensor(
            position_ids, dtype=torch.long)

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
    """Train one epoch with contrastive loss."""
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
    script_dir = Path(__file__).parent.absolute()

    # Navigate up to repo root, then to config
    config_path = script_dir.parent.parent / 'GraphCodeBert/config.json'
    with open(config_path)as f:
        config = json.load(f).get('codesearch')

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    script_dir = Path(__file__).parent.absolute()
    model_dir = Path(config['mlm_model_path'])
    # Navigate up to repo root, then to config
    model_path = script_dir.parent.parent / model_dir
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path).to(device)

    script_dir = Path(__file__).parent.absolute()
    output_dir = Path(config['output_dir'])
    # Navigate up to repo root, then to config
    output_dir = script_dir.parent.parent / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent.absolute()
    data_dir = Path(config['processed_data_file'])
    # Navigate up to repo root, then to config
    data_path = script_dir.parent.parent / data_dir
    # Load preprocessed dataset (not encoded, allows dynamic collator processing)
    dataset = CodeSearchDataset(data_path)
    collator = CodeSearchCollator(tokenizer, config['max_code_len'], config['max_query_len'])

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

    checkpoint_path = output_dir / "best_model"
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == '__main__':
    main()