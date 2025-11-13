"""
Train GraphCodeBERT on MLM + Edge Prediction tasks with DFG for C++ code.
Implements the dual-objective pre-training from GraphCodeBERT paper.
OPTIMIZED VERSION with early stopping, dropout, mixed precision, and weight decay.
WITH COMPREHENSIVE LOSS AND PERFORMANCE TRACKING
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
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from collections import defaultdict


"""
Set random seeds for reproducibility
"""
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)


class PerformanceTracker:
    """Tracks and saves all performance metrics during training."""
    def __init__(self, output_dir: str, patience: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.patience_counter = 0
        self.best_val_loss = float('inf')

        self.history = {
            'epoch': [],
            'train_total_loss': [],
            'train_mlm_loss': [],
            'train_edge_loss': [],
            'train_batch_losses': [],
            'train_mlm_batch_losses': [],
            'train_edge_batch_losses': [],
            'val_total_loss': [],
            'val_mlm_loss': [],
            'val_edge_loss': [],
            'val_batch_losses': [],
            'val_mlm_batch_losses': [],
            'val_edge_batch_losses': [],
            'learning_rate': [],
            'best_val_loss': None,
            'best_epoch': None,
        }

    def log_batch(self, phase: str, total_loss, mlm_loss, edge_loss):
        """Log individual batch metrics."""
        if phase == 'train':
            self.history['train_batch_losses'].append(total_loss)
            self.history['train_mlm_batch_losses'].append(mlm_loss if mlm_loss else 0)
            self.history['train_edge_batch_losses'].append(edge_loss if edge_loss else 0)
        else:
            self.history['val_batch_losses'].append(total_loss)
            self.history['val_mlm_batch_losses'].append(mlm_loss if mlm_loss else 0)
            self.history['val_edge_batch_losses'].append(edge_loss if edge_loss else 0)

    def log_epoch(self, epoch: int, phase: str, total_loss, mlm_loss, edge_loss, lr=None):
        """Log epoch-level metrics."""
        if phase == 'train':
            self.history['epoch'].append(epoch)
            self.history['train_total_loss'].append(total_loss)
            self.history['train_mlm_loss'].append(mlm_loss)
            self.history['train_edge_loss'].append(edge_loss)
            if lr is not None:
                self.history['learning_rate'].append(lr)
        else:
            self.history['val_total_loss'].append(total_loss)
            self.history['val_mlm_loss'].append(mlm_loss)
            self.history['val_edge_loss'].append(edge_loss)

    def update_best(self, val_loss, epoch):
        """Update best validation loss and handle early stopping."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.history['best_val_loss'] = val_loss
            self.history['best_epoch'] = epoch
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False

    def should_stop_early(self) -> bool:
        """Check if training should stop early."""
        return self.patience_counter >= self.patience

    def save(self):
        """Save all metrics to JSON files."""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Saved training history to {history_path}")

        summary = self._compute_summary()
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved training summary to {summary_path}")

        self._save_csv()

    def _compute_summary(self) -> Dict:
        """Compute summary statistics."""
        return {
            'total_epochs': len(self.history['epoch']),
            'best_epoch': self.history['best_epoch'],
            'best_val_loss': self.history['best_val_loss'],
            'final_train_loss': self.history['train_total_loss'][-1] if self.history['train_total_loss'] else None,
            'final_val_loss': self.history['val_total_loss'][-1] if self.history['val_total_loss'] else None,
            'min_train_loss': min(self.history['train_total_loss']) if self.history['train_total_loss'] else None,
            'min_val_loss': min(self.history['val_total_loss']) if self.history['val_total_loss'] else None,
            'final_train_mlm_loss': self.history['train_mlm_loss'][-1] if self.history['train_mlm_loss'] else None,
            'final_train_edge_loss': self.history['train_edge_loss'][-1] if self.history['train_edge_loss'] else None,
            'final_val_mlm_loss': self.history['val_mlm_loss'][-1] if self.history['val_mlm_loss'] else None,
            'final_val_edge_loss': self.history['val_edge_loss'][-1] if self.history['val_edge_loss'] else None,
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
                writer.writerow([
                    'Epoch', 'Train Total Loss', 'Train MLM Loss', 'Train Edge Loss',
                    'Val Total Loss', 'Val MLM Loss', 'Val Edge Loss', 'Learning Rate'
                ])
                for i in range(len(self.history['epoch'])):
                    writer.writerow([
                        self.history['epoch'][i],
                        self.history['train_total_loss'][i],
                        self.history['train_mlm_loss'][i],
                        self.history['train_edge_loss'][i],
                        self.history['val_total_loss'][i] if i < len(self.history['val_total_loss']) else '',
                        self.history['val_mlm_loss'][i] if i < len(self.history['val_mlm_loss']) else '',
                        self.history['val_edge_loss'][i] if i < len(self.history['val_edge_loss']) else '',
                        self.history['learning_rate'][i] if i < len(self.history['learning_rate']) else '',
                    ])
            print(f"✓ Saved metrics CSV to {csv_path}")
        except Exception as e:
            print(f"⚠️ Could not save CSV: {e}")


class GraphCodeBERTDataset(Dataset):
    """Custom Dataset for GraphCodeBERT with DFG processing"""
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

        # Extract DFG relationships
        for var, use_pos, _, _, dep_pos_list in dfg:
            if use_pos not in node_to_idx:
                node_to_idx[use_pos] = len(dfg_nodes)
                dfg_nodes.append((var, use_pos))
            for def_pos in dep_pos_list:
                if def_pos not in node_to_idx:
                    node_to_idx[def_pos] = len(dfg_nodes)
                    dfg_nodes.append((var, def_pos))
                adj[node_to_idx[use_pos]].append(node_to_idx[def_pos])

        # Calculate max code length considering DFG nodes
        # Format: [CLS] + code_tokens + [SEP] + dfg_tokens + [SEP]
        dfg_token_count = len(dfg_nodes)
        max_code_len = self.max_length - dfg_token_count - 3  # 3 for [CLS], [SEP], [SEP]

        if len(code_tokens) > max_code_len:
            code_tokens = code_tokens[:max_code_len]

        # Build token sequence
        tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
        dfg_start_pos = len(tokens)
        tokens.extend([self.tokenizer.unk_token] * dfg_token_count)
        tokens.append(self.tokenizer.sep_token)

        # Convert tokens to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        position_idx = list(range(len(code_tokens) + 2)) + [0] * dfg_token_count + [len(code_tokens) + 2]

        # Pad to max_length
        padding_len = self.max_length - len(input_ids)
        if padding_len < 0:
            raise ValueError(
                f"Sequence too long! {len(tokens)} tokens > {self.max_length} max_length. "
                f"Code tokens: {len(code_tokens)}, DFG nodes: {dfg_token_count}"
            )

        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
        position_idx.extend([0] * padding_len)

        # Build attention mask
        # According to GraphCodeBERT paper:
        # 1. Code tokens can attend to all code tokens (including code representation in DFG)
        # 2. DFG nodes attend to related code positions and other DFG nodes
        attn_mask = np.zeros((self.max_length, self.max_length), dtype=np.bool_)
        code_len = len(code_tokens) + 2

        # Code section attends to code section
        attn_mask[:code_len, :code_len] = True

        # Each token attends to itself
        for i in range(len(tokens)):
            attn_mask[i, i] = True

        # DFG nodes attend to their corresponding code positions
        for i, (_, code_pos) in enumerate(dfg_nodes):
            if code_pos + 1 < code_len:
                dfg_abs = dfg_start_pos + i
                code_abs = code_pos + 1
                attn_mask[dfg_abs, code_abs] = True
                attn_mask[code_abs, dfg_abs] = True

        # DFG edges: nodes attend to their dependencies
        for i, adjs in adj.items():
            for j in adjs:
                u, v = dfg_start_pos + i, dfg_start_pos + j
                attn_mask[u, v] = True
                attn_mask[v, u] = True

        # Validate shapes
        assert len(input_ids) == self.max_length, \
            f"Input IDs length {len(input_ids)} != max_length {self.max_length}"
        assert len(position_idx) == self.max_length, \
            f"Position indices length {len(position_idx)} != max_length {self.max_length}"
        assert attn_mask.shape == (self.max_length, self.max_length), \
            f"Attention mask shape {attn_mask.shape} != ({self.max_length}, {self.max_length})"

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, dtype=torch.bool),
            'position_idx': torch.tensor(position_idx, dtype=torch.long),
            'dfg_info': {
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

        # Add dropout for regularization
        self.roberta_mlm.config.hidden_dropout_prob = 0.2
        self.roberta_mlm.config.attention_probs_dropout_prob = 0.2

        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2),
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
    """Data collator for MLM and Edge Prediction"""
    tokenizer: RobertaTokenizer
    mlm_probability: float = 0.15
    edge_sample_ratio: float = 0.3

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)
        # Get max_length from first example's input_ids
        max_seq_length = examples[0]['input_ids'].shape[0]

        input_ids = torch.stack([ex['input_ids'] for ex in examples])
        attn_mask = torch.stack([ex['attention_mask'] for ex in examples])
        pos_idx = torch.stack([ex['position_idx'] for ex in examples])

        # Verify batch dimensions match actual max_length
        assert input_ids.shape == (batch_size, max_seq_length), \
            f"Input IDs batch shape error: {input_ids.shape} vs expected {(batch_size, max_seq_length)}"
        assert attn_mask.shape == (batch_size, max_seq_length, max_seq_length), \
            f"Attention mask batch shape error: {attn_mask.shape} vs expected {(batch_size, max_seq_length, max_seq_length)}"
        assert pos_idx.shape == (batch_size, max_seq_length), \
            f"Position indices batch shape error: {pos_idx.shape} vs expected {(batch_size, max_seq_length)}"

        # MLM masking
        labels, masked_ids = input_ids.clone(), input_ids.clone()
        for i in range(batch_size):
            code_indices = (pos_idx[i] > 1).nonzero(as_tuple=True)[0]
            if len(code_indices) > 1:
                code_indices = code_indices[:-1]
            if len(code_indices) == 0:
                continue
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
        for i in range(batch_size):
            if 'dfg_info' not in examples[i]:
                continue
            dfg_nodes = examples[i]['dfg_info']['nodes']
            dfg_edges = examples[i]['dfg_info']['edges']
            if len(dfg_nodes) < 2:
                continue

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
            edge_batch_idx = torch.tensor([], dtype=torch.long)
            edge_node1_pos = torch.tensor([], dtype=torch.long)
            edge_node2_pos = torch.tensor([], dtype=torch.long)
            edge_labels = torch.tensor([], dtype=torch.float)

        return {
            'input_ids': masked_ids,
            'attention_mask': attn_mask,
            'position_ids': pos_idx,
            'labels': labels,
            'edge_batch_idx': edge_batch_idx,
            'edge_node1_pos': edge_node1_pos,
            'edge_node2_pos': edge_node2_pos,
            'edge_labels': edge_labels
        }


def train_epoch(model, dataloader, optimizer, scheduler, device, tracker: PerformanceTracker, scaler, use_amp=False):
    """Training loop with memory management and loss tracking"""
    model.train()
    total_loss = total_mlm = total_edge = 0
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()

        try:
            if use_amp:
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    loss, mlm_loss, edge_loss = model(
                        input_ids=batch['input_ids'].to(device),
                        attention_mask=batch['attention_mask'].to(device),
                        position_ids=batch['position_idx'].to(device),
                        labels=batch['labels'].to(device),
                        edge_batch_idx=batch['edge_batch_idx'].to(device),
                        edge_node1_pos=batch['edge_node1_pos'].to(device),
                        edge_node2_pos=batch['edge_node2_pos'].to(device),
                        edge_labels=batch['edge_labels'].to(device)
                    )

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, mlm_loss, edge_loss = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    position_ids=batch['position_idx'].to(device),
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
            batch_count += 1

            tracker.log_batch('train', loss.item(),
                             mlm_loss.item() if mlm_loss else None,
                             edge_loss.item() if edge_loss else None)

            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = total_loss / batch_count
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}',
                'mlm': f'{mlm_loss.item() if mlm_loss else 0:.4f}',
                'edge': f'{edge_loss.item() if edge_loss else 0:.4f}',
                'lr': f'{current_lr:.2e}'
            })

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n⚠️ OUT OF MEMORY ERROR!")
                print(f"   Batch size: {batch['input_ids'].shape[0]}")
                print(f"   Sequence length: {batch['input_ids'].shape[1]}")
                print(f"   Estimated batch memory: ~{batch['input_ids'].shape[0] * batch['input_ids'].shape[1]**2 / 1e6:.0f}MB for attention alone")
                print(f"   Try reducing batch_size or max_length")
                raise
            else:
                raise

        finally:
            # Clear GPU cache after each batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()

            # Delete batch to free memory
            del batch

    return (total_loss / batch_count, total_mlm / batch_count, total_edge / batch_count)


def validate(model, dataloader, device, tracker: PerformanceTracker, use_amp=False):
    """Validation loop with loss tracking"""
    model.eval()
    total_loss = total_mlm = total_edge = 0
    batch_count = 0
    progress_bar = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for batch in progress_bar:
            if use_amp:
                with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
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
            else:
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
            batch_count += 1

            tracker.log_batch('val', loss.item(),
                             mlm_loss.item() if mlm_loss else None,
                             edge_loss.item() if edge_loss else None)

            avg_loss = total_loss / batch_count
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{avg_loss:.4f}',
                'mlm': f'{mlm_loss.item() if mlm_loss else 0:.4f}',
                'edge': f'{edge_loss.item() if edge_loss else 0:.4f}'
            })

    return (total_loss / batch_count, total_mlm / batch_count, total_edge / batch_count)


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
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')

    script_dir = Path(__file__).parent.absolute()

    config_path = script_dir.parent.parent / 'GraphCodeBert/config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f).get("train", {})
    parser.set_defaults(**config)
    args = parser.parse_args()
    if not args.data_file: parser.error("data_file must be specified.")

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

    output_path = script_dir.parent.parent / args.output_dir
    Path(output_path).mkdir(parents=True, exist_ok=True)

    tracker = PerformanceTracker(str(output_path), patience=args.early_stopping_patience)

    print("Loading GraphCodeBERT base...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = GraphCodeBERTWithEdgePrediction("microsoft/graphcodebert-base").to(device)
    print("✓ Loaded GraphCodeBERT with dropout regularization")

    data_dir = Path(__file__).parent.absolute()
    data_path = data_dir.parent.parent / args.data_file
    full_dataset = GraphCodeBERTDataset(data_path, tokenizer, args.max_length)
    val_size = int(args.validation_split * len(full_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [len(full_dataset) - val_size, val_size]
    )

    collator = MLMWithEdgePredictionCollator(tokenizer, mlm_probability=args.mlm_probability)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=collator, num_workers=min(4, os.cpu_count() or 1),
                         pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                       collate_fn=collator, num_workers=min(4, os.cpu_count() or 1),
                       pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dl) * args.epochs
    )

    scaler = GradScaler() if use_amp else None

    print("\n--- Training Configuration ---")
    for k, v in vars(args).items(): print(f"  {k}: {v}")
    print(f"  use_amp: {use_amp}")
    print(f"  device: {device}")
    print("------------------------------\n")

    # Enable gradient checkpointing to save memory
    if hasattr(model.roberta_mlm, 'gradient_checkpointing_enable'):
        model.roberta_mlm.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled (trades compute for memory)")

    for epoch in range(args.epochs):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 70}")

        # Clear cache before epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif device.type == 'mps':
            torch.mps.empty_cache()

        train_loss, train_mlm, train_edge = train_epoch(model, train_dl, optimizer, scheduler, device, tracker, scaler, use_amp)

        # Clear cache before validation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

        val_loss, val_mlm, val_edge = validate(model, val_dl, device, tracker, use_amp)

        current_lr = optimizer.param_groups[0]['lr']

        tracker.log_epoch(epoch, 'train', train_loss, train_mlm, train_edge, current_lr)
        tracker.log_epoch(epoch, 'val', val_loss, val_mlm, val_edge)

        # Memory stats
        if device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n  Peak GPU Memory: {peak_mem:.2f} GB")

        print(f"\n{'─' * 70}")
        print(f"Epoch {epoch + 1} Results:")
        print(f"  Train - Total: {train_loss:.6f}, MLM: {train_mlm:.6f}, Edge: {train_edge:.6f}")
        print(f"  Val   - Total: {val_loss:.6f}, MLM: {val_mlm:.6f}, Edge: {val_edge:.6f}")
        print(f"  Learning Rate: {current_lr:.6e}")
        print(f"  Best Val Loss: {tracker.best_val_loss:.6f} (Epoch {tracker.history['best_epoch'] + 1 if tracker.history['best_epoch'] is not None else 'N/A'})")
        print(f"  Patience:      {tracker.patience_counter}/{args.early_stopping_patience}")
        print(f"{'─' * 70}")

        if tracker.update_best(val_loss, epoch):
            checkpoint_path = Path(args.output_dir) / "best_model"
            print(f"\n✓ New best model! Saving to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
        else:
            print(f"\n⚠️  No improvement. Patience: {tracker.patience_counter}/{args.early_stopping_patience}")

        if tracker.should_stop_early():
            print(f"\n⚠️  Early stopping triggered!")
            print(f"   No improvement for {args.early_stopping_patience} epochs")
            print(f"   Best loss: {tracker.best_val_loss:.6f} at epoch {tracker.history['best_epoch'] + 1}")
            break

    print(f"\n{'=' * 70}")
    print(f"Training completed!")
    print(f"Best val loss: {tracker.best_val_loss:.6f} at epoch {tracker.history['best_epoch'] + 1}")
    print(f"{'=' * 70}\n")

    print("="*70)
    print("SAVING PERFORMANCE METRICS...")
    print("="*70)
    tracker.save()


if __name__ == "__main__":
    main()