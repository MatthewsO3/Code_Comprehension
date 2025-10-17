"""
Train GraphCodeBERT on MLM + Edge Prediction tasks with DFG for C++ code.
Implements the dual-objective pre-training from GraphCodeBERT paper.
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
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
                node_to_idx[use_pos] = len(dfg_nodes)
                dfg_nodes.append((var, use_pos))
            for def_pos in dep_pos_list:
                if def_pos not in node_to_idx:
                    node_to_idx[def_pos] = len(dfg_nodes)
                    dfg_nodes.append((var, def_pos))
                adj[node_to_idx[use_pos]].append(node_to_idx[def_pos])

        max_code_len = self.max_length - len(dfg_nodes) - 3
        if len(code_tokens) > max_code_len:
            code_tokens = code_tokens[:max_code_len]

        tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        dfg_start_pos = len(tokens)
        tokens.extend([self.tokenizer.unk_token] * len(dfg_nodes))
        tokens.append(self.tokenizer.sep_token)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        position_idx = list(range(len(code_tokens) + 2)) + [0] * len(dfg_nodes) + [len(code_tokens) + 2]

        attn_mask = np.zeros((self.max_length, self.max_length), dtype=bool)
        code_len = len(code_tokens) + 2
        attn_mask[:code_len, :code_len] = True
        for i in range(len(tokens)):
            attn_mask[i, i] = True
        for i, (_, code_pos) in enumerate(dfg_nodes):
            if code_pos + 1 < code_len:
                dfg_abs = dfg_start_pos + i
                code_abs = code_pos + 1
                attn_mask[dfg_abs, code_abs] = attn_mask[code_abs, dfg_abs] = True
        for i, adjs in adj.items():
            for j in adjs:
                u, v = dfg_start_pos + i, dfg_start_pos + j
                attn_mask[u, v] = attn_mask[v, u] = True

        padding_len = self.max_length - len(input_ids)
        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
        position_idx.extend([0] * padding_len)

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attn_mask),
            'position_idx': torch.tensor(position_idx),
            'dfg_info': {  # NEW - needed for edge prediction
                'nodes': dfg_nodes,
                'edges': [(i, j) for i, adjs in adj.items() for j in adjs]
            }
        }


class GraphCodeBERTWithEdgePrediction(nn.Module):
    """GraphCodeBERT with MLM and Edge Prediction heads"""
    def __init__(self, base_model_name: str = "microsoft/graphcodebert-base"):
        super().__init__()
        self.roberta_mlm = RobertaForMaskedLM.from_pretrained(base_model_name)
        hidden_size = self.roberta_mlm.config.hidden_size
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask, position_ids, labels=None,
                edge_batch_idx=None, edge_node1_pos=None, edge_node2_pos=None, edge_labels=None):
        mlm_outputs = self.roberta_mlm(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, labels=labels, output_hidden_states=True
        )
        mlm_loss = mlm_outputs.loss if labels is not None else None

        edge_loss = None
        if (edge_batch_idx is not None and len(edge_batch_idx) > 0 and
            edge_node1_pos is not None and edge_node2_pos is not None and edge_labels is not None):
            hidden_states = mlm_outputs.hidden_states[-1]
            batch_size, seq_len, hidden_size = hidden_states.shape

            # Gather node representations
            node1_repr = hidden_states[edge_batch_idx, edge_node1_pos]
            node2_repr = hidden_states[edge_batch_idx, edge_node2_pos]
            edge_repr = torch.cat([node1_repr, node2_repr], dim=-1)
            edge_logits = self.edge_classifier(edge_repr).squeeze(-1)
            edge_loss = nn.functional.binary_cross_entropy_with_logits(edge_logits, edge_labels)

        if mlm_loss is not None and edge_loss is not None:
            total_loss = mlm_loss + edge_loss
        elif mlm_loss is not None:
            total_loss = mlm_loss
        else:
            total_loss = edge_loss

        return total_loss, mlm_loss, edge_loss

    def save_pretrained(self, save_directory):
        self.roberta_mlm.save_pretrained(save_directory)
        torch.save(self.edge_classifier.state_dict(), f"{save_directory}/edge_classifier.pt")


@dataclass
class MLMWithEdgePredictionCollator:
    tokenizer: RobertaTokenizer
    mlm_probability: float = 0.15
    edge_sample_ratio: float = 0.3

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attn_mask = torch.stack([ex['attention_mask'] for ex in examples])
        pos_idx = torch.stack([ex['position_idx'] for ex in examples])

        # MLM masking
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
            mask_ind = torch.zeros_like(labels[i], dtype=torch.bool)
            mask_ind[mask_pos] = True
            labels[i, ~mask_ind] = -100
        labels[masked_ids == self.tokenizer.pad_token_id] = -100

        # Edge prediction
        edge_pairs = []
        max_pairs = 20
        for i in range(len(examples)):
            if 'dfg_info' not in examples[i]: continue
            dfg_nodes = examples[i]['dfg_info']['nodes']
            dfg_edges = examples[i]['dfg_info']['edges']
            if len(dfg_nodes) < 2: continue

            edge_set = set(dfg_edges)
            edge_set.update((v, u) for u, v in dfg_edges)

            num_nodes = len(dfg_nodes)
            num_pairs = min(max_pairs, int(num_nodes * (num_nodes - 1) / 2 * self.edge_sample_ratio))
            sampled = set()
            attempts = 0
            while len(sampled) < num_pairs and attempts < num_pairs * 3:
                u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
                if u != v and (u, v) not in sampled and (v, u) not in sampled:
                    sampled.add((u, v))
                attempts += 1

            for u, v in sampled:
                has_edge = 1 if (u, v) in edge_set else 0
                u_pos = dfg_nodes[u][1] + 1
                v_pos = dfg_nodes[v][1] + 1
                edge_pairs.append((i, u_pos, v_pos, has_edge))

        if edge_pairs:
            edge_batch_idx = torch.tensor([p[0] for p in edge_pairs], dtype=torch.long)
            edge_node1_pos = torch.tensor([p[1] for p in edge_pairs], dtype=torch.long)
            edge_node2_pos = torch.tensor([p[2] for p in edge_pairs], dtype=torch.long)
            edge_labels = torch.tensor([p[3] for p in edge_pairs], dtype=torch.float)
        else:
            edge_batch_idx = torch.zeros(0, dtype=torch.long)
            edge_node1_pos = torch.zeros(0, dtype=torch.long)
            edge_node2_pos = torch.zeros(0, dtype=torch.long)
            edge_labels = torch.zeros(0, dtype=torch.float)

        return {
            'input_ids': masked_ids, 'attention_mask': attn_mask,
            'position_ids': pos_idx, 'labels': labels,
            'edge_batch_idx': edge_batch_idx, 'edge_node1_pos': edge_node1_pos,
            'edge_node2_pos': edge_node2_pos, 'edge_labels': edge_labels
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = total_mlm = total_edge = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        loss, mlm_loss, edge_loss = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            position_ids=batch['position_ids'].to(device),
            labels=batch['labels'].to(device),
            edge_batch_idx=batch['edge_batch_idx'].to(device),
            edge_node1_pos=batch['edge_node1_pos'].to(device),
            edge_node2_pos=batch['edge_node2_pos'].to(device),
            edge_labels=batch['edge_labels'].to(device)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if mlm_loss: total_mlm += mlm_loss.item()
        if edge_loss: total_edge += edge_loss.item()
        progress_bar.set_postfix({
            'loss': loss.item(),
            'mlm': mlm_loss.item() if mlm_loss else 0,
            'edge': edge_loss.item() if edge_loss else 0
        })
    return total_loss / len(dataloader), total_mlm / len(dataloader), total_edge / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = total_mlm = total_edge = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            loss, mlm_loss, edge_loss = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                position_ids=batch['position_ids'].to(device),
                labels=batch['labels'].to(device),
                edge_batch_idx=batch['edge_batch_idx'].to(device),
                edge_node1_pos=batch['edge_node1_pos'].to(device),
                edge_node2_pos=batch['edge_node2_pos'].to(device),
                edge_labels=batch['edge_labels'].to(device)
            )
            total_loss += loss.item()
            if mlm_loss: total_mlm += mlm_loss.item()
            if edge_loss: total_edge += edge_loss.item()
    return total_loss / len(dataloader), total_mlm / len(dataloader), total_edge / len(dataloader)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GraphCodeBERT with Edge Prediction')
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
        with open('../config.json', 'r') as f:
            config = json.load(f).get("train", {})
    parser.set_defaults(**config)
    args = parser.parse_args()
    if not args.data_file: parser.error("data_file must be specified.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = GraphCodeBERTWithEdgePrediction("microsoft/graphcodebert-base").to(device)

    full_dataset = GraphCodeBERTDataset(args.data_file, tokenizer, args.max_length)
    val_size = int(args.validation_split * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [len(full_dataset) - val_size, val_size]
    )

    collator = MLMWithEdgePredictionCollator(tokenizer, mlm_probability=args.mlm_probability)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=collator, num_workers=min(4, os.cpu_count() or 1))
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size,
                       collate_fn=collator, num_workers=min(4, os.cpu_count() or 1))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dl) * args.epochs
    )

    print("\n--- Training Configuration ---")
    for k, v in vars(args).items(): print(f"  {k}: {v}")
    print("------------------------------\n")

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\n{'=' * 20} Epoch {epoch + 1}/{args.epochs} {'=' * 20}")
        train_loss, train_mlm, train_edge = train_epoch(model, train_dl, optimizer, scheduler, device)
        val_loss, val_mlm, val_edge = validate(model, val_dl, device)
        print(f"Train - Total: {train_loss:.4f}, MLM: {train_mlm:.4f}, Edge: {train_edge:.4f}")
        print(f"Val   - Total: {val_loss:.4f}, MLM: {val_mlm:.4f}, Edge: {val_edge:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(args.output_dir) / "best_model"
            print(f"New best model! Saving to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

    print(f"\n{'=' * 15} Training complete! Best val loss: {best_val_loss:.4f} {'=' * 15}")


if __name__ == "__main__":
    main()