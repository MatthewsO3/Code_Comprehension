# train_codesearch.py (Corrected and Self-Contained)

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

# --- NEW: Added DFG extraction dependencies ---
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp

# --- NEW: Initialize Tree-sitter and Tokenizer for processing ---
CPP_LANGUAGE = Language(tscpp.language())
ts_parser = Parser(CPP_LANGUAGE)
# This tokenizer will be used for on-the-fly processing
processing_tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


set_seed(42)


# --- NEW: DFG extraction function (adapted from your dataset.py) ---
def extract_dataflow_graph(code_bytes, tree):
    # This is a simplified version for demonstration.
    # For a more robust implementation, refer to the original GraphCodeBERT repo.
    try:
        root_node = tree.root_node
        tokens = []
        node_to_token_pos = {}

        def get_tokens_recursive(node):
            if not node.children:
                tokens.append(node)
                node_to_token_pos[node.id] = len(tokens) - 1
            for child in node.children:
                get_tokens_recursive(child)

        get_tokens_recursive(root_node)

        var_definitions = defaultdict(list)
        var_uses = defaultdict(list)

        def is_definition(node):
            parent = node.parent
            if not parent: return False
            if parent.type in ['declaration', 'init_declarator', 'parameter_declaration']: return True
            if parent.type == 'assignment_expression' and node.id == parent.child_by_field_name('left').id: return True
            return False

        queue = [root_node]
        while queue:
            node = queue.pop(0)
            if node.type in ['identifier', 'field_identifier']:
                var_name = code_bytes[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
                token_pos = node_to_token_pos.get(node.id)
                if token_pos is not None:
                    if is_definition(node):
                        var_definitions[var_name].append(token_pos)
                    else:
                        var_uses[var_name].append(token_pos)
            queue.extend(node.children)

        dfg_edges = []
        for var_name, uses in var_uses.items():
            defs = sorted(var_definitions.get(var_name, []))
            for use_pos in uses:
                preceding_defs = [d for d in defs if d < use_pos]
                if preceding_defs:
                    def_pos = preceding_defs[-1]
                    dfg_edges.append((var_name, use_pos, "comesFrom", [var_name], [def_pos]))
        return dfg_edges
    except Exception:
        return []


# --- MODIFIED: The Dataset class now performs processing in __getitem__ ---
class CodeSearchDataset(Dataset):
    def __init__(self, jsonl_file: str):
        print(f"Loading raw samples from {jsonl_file}...")
        self.samples = [json.loads(line) for line in open(jsonl_file, 'r', encoding='utf-8')]
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Process one sample at a time when requested by the DataLoader
        raw_sample = self.samples[idx]
        code = raw_sample['code']
        docstring = raw_sample['docstring']

        # Process Code
        code_bytes = code.encode('utf8')
        tree = ts_parser.parse(code_bytes)
        code_tokens = processing_tokenizer.tokenize(code, add_prefix_space=True)
        dfg = extract_dataflow_graph(code_bytes, tree)

        # Process Docstring (Query)
        docstring_tokens = processing_tokenizer.tokenize(docstring, add_prefix_space=True)

        return {
            'code_tokens': code_tokens,
            'docstring_tokens': docstring_tokens,
            'dataflow_graph': dfg
        }


# The rest of the script (Collator, training loop, etc.) remains the same
# as it now receives the correctly processed data it was always expecting.

class CodeSearchCollator:
    def __init__(self, tokenizer, max_code_len, max_query_len):
        self.tokenizer = tokenizer
        self.max_code_len = max_code_len
        self.max_query_len = max_query_len

    def _process_item(self, tokens, dfg, max_len):
        # This function remains largely the same, but simplified for clarity
        if dfg:  # This is code
            node_map = {}
            nodes_data = []
            for _, use_pos, _, _, dep_list in dfg:
                for pos in [use_pos] + dep_list:
                    if pos < len(tokens) and pos not in node_map:
                        node_map[pos] = len(nodes_data)
                        nodes_data.append(("", pos))

            adj = defaultdict(list)
            for _, use_pos, _, _, dep_list in dfg:
                for def_pos in dep_list:
                    if use_pos in node_map and def_pos in node_map:
                        adj[node_map[use_pos]].append(node_map[def_pos])

            max_code_tok_len = max_len - len(node_map) - 3
            if len(tokens) > max_code_tok_len:
                tokens = tokens[:max_code_tok_len]

            input_tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            dfg_start = len(input_tokens)
            input_tokens.extend([self.tokenizer.unk_token] * len(node_map))
            input_tokens.append(self.tokenizer.sep_token)

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            position_ids = list(range(len(tokens) + 2)) + [0] * len(node_map) + [len(tokens) + 2]

            attn_mask = np.zeros((max_len, max_len), dtype=bool)
            code_len = len(tokens) + 2
            attn_mask[:code_len, :code_len] = True
            for i, _ in enumerate(input_tokens): attn_mask[i, i] = True

            for code_pos, node_idx in node_map.items():
                if code_pos < len(tokens):
                    attn_mask[dfg_start + node_idx, code_pos + 1] = True
                    attn_mask[code_pos + 1, dfg_start + node_idx] = True

            for u_idx, v_indices in adj.items():
                for v_idx in v_indices:
                    attn_mask[dfg_start + u_idx, dfg_start + v_idx] = True
                    attn_mask[dfg_start + v_idx, dfg_start + u_idx] = True

            attn_mask = attn_mask.tolist()
        else:  # This is a query
            if len(tokens) > max_len - 2:
                tokens = tokens[:max_len - 2]
            input_tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            position_ids = list(range(len(input_ids)))
            attn_mask = [[True] * max_len for _ in range(max_len)]

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

        scores = cos(query_vecs.unsqueeze(1), code_vecs.unsqueeze(0))
        labels = torch.arange(len(scores)).to(device)
        loss = loss_fn(scores, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    config = json.load(open('/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/GraphCodeBert/config.json')).get('codesearch')

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # This tokenizer is passed to the collator for converting tokens to IDs
    tokenizer = RobertaTokenizer.from_pretrained(config['mlm_model_path'])
    model = RobertaModel.from_pretrained(config['mlm_model_path']).to(device)

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CodeSearchDataset(config['processed_data_file'])
    collator = CodeSearchCollator(tokenizer, config['max_code_len'], config['max_query_len'])

    # Set num_workers to 0 on MPS to avoid issues with multiprocessing
    num_workers = 0 if device.type == 'mps' else min(4, os.cpu_count() or 1)

    train_dl = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collator,
                          num_workers=num_workers)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100,
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