"""
Evaluate all three C++ code models (GraphCodeBERT, UniXcoder, CodeBERT-cpp) on MLM task.
Loads evaluation data from JSONL files and saves aggregated metrics to .txt files.
"""

import torch
import numpy as np
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from transformers import RobertaTokenizer, RobertaForMaskedLM

try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp as tscpp
    TS_AVAILABLE = True
    CPP_LANGUAGE = Language(tscpp.language())
    ts_parser = Parser(CPP_LANGUAGE)
except ImportError:
    TS_AVAILABLE = False
    print("Warning: tree_sitter not available. DFG extraction will fail.")

random.seed(42)
torch.manual_seed(42)

# Model paths
repo_dir = Path(__file__).parent.parent.absolute()
print("repo dir: ", repo_dir)

GRAPH_PATH ="/home/mczap/code_comp/Code_Comprehension/BERTModels/GraphCodeBert/MLM/GraphCodeBert/Models/graphcodebert-cpp-mlm-from-config/best_model"
UNIX_PATH = "/home/mczap/code_comp/Code_Comprehension/BERTModels/UnixCoderCPP/unixcoder-cpp-mlm/best_model"
CODE_PATH = 'neulab/codebert-cpp'

# Configuration
MASK_RATIO = 0.2
TOP_K = 10
RESULTS_DIR = Path('results')
DATA_DIR = Path('data')

# JSONL file names
GRAPHCODEBERT_JSONL = DATA_DIR / 'eval/graphcodebert_evalset.jsonl'
UNIXCODER_JSONL = DATA_DIR / 'eval/unixcoder_evalset.jsonl'
CODEBERT_JSONL = DATA_DIR / 'eval/codebert_evalset.jsonl'


# ============================================================================
# Load JSONL data
# ============================================================================

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load a JSONL file into a list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"Loaded {len(data)} samples from {filepath}")
        return data
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        return []


# ============================================================================
# GraphCodeBERT Evaluator
# ============================================================================

class GraphCodeBERTEvaluator:
    def __init__(self, model_path: str, tokenizer: RobertaTokenizer, device: str = None):
        if not TS_AVAILABLE:
            raise RuntimeError("Tree-sitter is required for GraphCodeBERT.")
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Loading GraphCodeBERT from {model_path}...")
        self.tokenizer = tokenizer
        self.model = RobertaForMaskedLM.from_pretrained(model_path).to(self.device).eval()
        print("GraphCodeBERT loaded successfully!")

    def extract_dfg_for_snippet(self, code_bytes: bytes) -> List[Tuple]:
        """Extract DFG from C++ code."""
        tree = ts_parser.parse(code_bytes)
        root = tree.root_node
        defs, uses = defaultdict(list), defaultdict(list)
        tokens, node_map = [], {}

        def find_tokens(node):
            if node.type in ['identifier', 'field_identifier']:
                if id(node) not in node_map:
                    node_map[id(node)] = len(tokens)
                    tokens.append(node)
            for child in node.children:
                find_tokens(child)

        find_tokens(root)

        def is_def(node):
            p = node.parent
            return p and (p.type in ['declaration', 'init_declarator', 'parameter_declaration'] or
                          (p.type == 'assignment_expression' and node == p.child_by_field_name('left')))

        def find_vars(node):
            if node.type in ['identifier', 'field_identifier']:
                name = code_bytes[node.start_byte:node.end_byte].decode('utf8', 'ignore')
                pos = node_map.get(id(node), -1)
                if pos != -1:
                    (defs if is_def(node) else uses)[name].append(pos)
            for child in node.children:
                find_vars(child)

        find_vars(root)
        edges = []
        for name, use_positions in uses.items():
            def_positions = sorted(defs.get(name, []))
            for use_pos in use_positions:
                preds = [d for d in def_positions if d < use_pos]
                if preds:
                    edges.append((name, use_pos, "comesFrom", [name], [preds[-1]]))
        return edges

    def preprocess_for_graphcodebert(self, code: str, masked_code_tokens: List[str]):
        """Prepare input with DFG attention mask."""
        dfg = self.extract_dfg_for_snippet(code.encode('utf8'))
        adj, nodes, node_map = defaultdict(list), [], {}
        for var, use_pos, _, _, dep_list in dfg:
            if use_pos not in node_map:
                node_map[use_pos] = len(nodes)
                nodes.append((var, use_pos))
            for def_pos in dep_list:
                if def_pos not in node_map:
                    node_map[def_pos] = len(nodes)
                    nodes.append((var, def_pos))
                adj[node_map[use_pos]].append(node_map[def_pos])

        tokens = [self.tokenizer.cls_token] + masked_code_tokens + [self.tokenizer.sep_token]
        dfg_start = len(tokens)
        tokens.extend([self.tokenizer.unk_token] * len(nodes))
        tokens.append(self.tokenizer.sep_token)

        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        pos_ids = list(range(len(masked_code_tokens) + 2)) + [0] * len(nodes) + [len(masked_code_tokens) + 2]

        mask = np.zeros((len(ids), len(ids)), dtype=bool)
        code_len = len(masked_code_tokens) + 2
        mask[:code_len, :code_len] = True
        for i in range(len(ids)):
            mask[i, i] = True
        for i, (_, code_pos) in enumerate(nodes):
            if code_pos < len(masked_code_tokens):
                dfg_abs, code_abs = dfg_start + i, code_pos + 1
                mask[dfg_abs, code_abs] = mask[code_abs, dfg_abs] = True
        for i, adjs in adj.items():
            for j in adjs:
                u, v = dfg_start + i, dfg_start + j
                mask[u, v] = mask[v, u] = True

        return {
            'input_ids': torch.tensor([ids]),
            'attention_mask': torch.tensor([mask.tolist()]),
            'position_ids': torch.tensor([pos_ids])
        }

    def evaluate_dataset(self, data: List[Dict], mask_ratio: float, top_k: int) -> Dict:
        """Evaluate model on dataset."""
        total_top1, total_top5, total_top10 = 0, 0, 0
        total_masked = 0
        all_log_probs = []

        for idx, sample in enumerate(data):
            code = sample.get('code')
            code_tokens = sample.get('code_tokens', [])

            if not code_tokens:
                continue

            num_mask = max(1, int(len(code_tokens) * mask_ratio))
            mask_positions = sorted(random.sample(range(len(code_tokens)), num_mask))
            original_tokens = [code_tokens[i] for i in mask_positions]

            masked_tokens = code_tokens.copy()
            for pos in mask_positions:
                masked_tokens[pos] = self.tokenizer.mask_token

            try:
                inputs = self.preprocess_for_graphcodebert(code, masked_tokens)
                with torch.no_grad():
                    logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits

                for i, pos in enumerate(mask_positions):
                    probs = torch.softmax(logits[0, pos + 1], dim=-1)
                    top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

                    original_token = original_tokens[i]
                    top_predictions = self.tokenizer.convert_ids_to_tokens(top_indices)

                    correct_token_prob = 1e-9
                    for rank, (pred, prob) in enumerate(zip(top_predictions, top_probs), 1):
                        if pred == original_token:
                            correct_token_prob = prob.item()
                            if rank <= 1:
                                total_top1 += 1
                            if rank <= 5:
                                total_top5 += 1
                            if rank <= 10:
                                total_top10 += 1
                            break

                    all_log_probs.append(np.log(correct_token_prob))
                    total_masked += 1
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(data)} samples...")

        return {
            'total_top1': total_top1,
            'total_top5': total_top5,
            'total_top10': total_top10,
            'total_masked': total_masked,
            'log_probs': all_log_probs
        }


# ============================================================================
# UniXcoder Evaluator
# ============================================================================

class UniXcoderEvaluator:
    def __init__(self, model_path: str, tokenizer: RobertaTokenizer, device: str = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Loading UniXcoder from {model_path}...")
        self.tokenizer = tokenizer
        self.model = RobertaForMaskedLM.from_pretrained(model_path).to(self.device).eval()
        print("UniXcoder loaded successfully!")

    def evaluate_dataset(self, data: List[Dict], mask_ratio: float, top_k: int) -> Dict:
        """Evaluate model on dataset."""
        total_top1, total_top5, total_top10 = 0, 0, 0
        total_masked = 0
        all_log_probs = []

        for idx, sample in enumerate(data):
            code_tokens = sample.get('code_tokens', [])

            if not code_tokens:
                continue

            num_mask = max(1, int(len(code_tokens) * mask_ratio))
            mask_positions = sorted(random.sample(range(len(code_tokens)), num_mask))
            original_tokens = [code_tokens[i] for i in mask_positions]

            masked_tokens = code_tokens.copy()
            for pos in mask_positions:
                masked_tokens[pos] = self.tokenizer.mask_token

            input_tokens = [self.tokenizer.cls_token] + masked_tokens + [self.tokenizer.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            attention_mask = [1] * len(input_ids)

            try:
                input_ids_tensor = torch.tensor([input_ids]).to(self.device)
                attention_mask_tensor = torch.tensor([attention_mask]).to(self.device)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids_tensor,
                        attention_mask=attention_mask_tensor
                    )
                    logits = outputs.logits

                for i, pos in enumerate(mask_positions):
                    actual_pos = pos + 1
                    probs = torch.softmax(logits[0, actual_pos], dim=-1)
                    top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

                    original_token = original_tokens[i]
                    top_predictions = self.tokenizer.convert_ids_to_tokens(top_indices)

                    correct_token_prob = 1e-9
                    for rank, (pred, prob) in enumerate(zip(top_predictions, top_probs), 1):
                        if pred == original_token:
                            correct_token_prob = prob.item()
                            if rank <= 1:
                                total_top1 += 1
                            if rank <= 5:
                                total_top5 += 1
                            if rank <= 10:
                                total_top10 += 1
                            break

                    all_log_probs.append(np.log(correct_token_prob))
                    total_masked += 1
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(data)} samples...")

        return {
            'total_top1': total_top1,
            'total_top5': total_top5,
            'total_top10': total_top10,
            'total_masked': total_masked,
            'log_probs': all_log_probs
        }


# ============================================================================
# CodeBERT-cpp Evaluator
# ============================================================================

class CodeBERTcppEvaluator:
    def __init__(self, model_path: str, tokenizer: RobertaTokenizer, device: str = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Loading CodeBERT-cpp from {model_path}...")
        self.tokenizer = tokenizer
        self.model = RobertaForMaskedLM.from_pretrained(model_path).to(self.device).eval()
        print("CodeBERT-cpp loaded successfully!")

    def evaluate_dataset(self, data: List[Dict], mask_ratio: float, top_k: int) -> Dict:
        """Evaluate model on dataset."""
        total_top1, total_top5, total_top10 = 0, 0, 0
        total_masked = 0
        all_log_probs = []

        for idx, sample in enumerate(data):
            code_tokens = sample.get('code_tokens', [])

            if not code_tokens:
                continue

            num_mask = max(1, int(len(code_tokens) * mask_ratio))
            mask_positions = sorted(random.sample(range(len(code_tokens)), num_mask))
            original_tokens = [code_tokens[i] for i in mask_positions]

            masked_tokens = code_tokens.copy()
            for pos in mask_positions:
                masked_tokens[pos] = self.tokenizer.mask_token

            input_tokens = [self.tokenizer.cls_token] + masked_tokens + [self.tokenizer.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            attention_mask = [1] * len(input_ids)

            try:
                input_ids_tensor = torch.tensor([input_ids]).to(self.device)
                attention_mask_tensor = torch.tensor([attention_mask]).to(self.device)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids_tensor,
                        attention_mask=attention_mask_tensor
                    )
                    logits = outputs.logits

                for i, pos in enumerate(mask_positions):
                    actual_pos = pos + 1
                    probs = torch.softmax(logits[0, actual_pos], dim=-1)
                    top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

                    original_token = original_tokens[i]
                    top_predictions = self.tokenizer.convert_ids_to_tokens(top_indices)

                    correct_token_prob = 1e-9
                    for rank, (pred, prob) in enumerate(zip(top_predictions, top_probs), 1):
                        if pred == original_token:
                            correct_token_prob = prob.item()
                            if rank <= 1:
                                total_top1 += 1
                            if rank <= 5:
                                total_top5 += 1
                            if rank <= 10:
                                total_top10 += 1
                            break

                    all_log_probs.append(np.log(correct_token_prob))
                    total_masked += 1
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(data)} samples...")

        return {
            'total_top1': total_top1,
            'total_top5': total_top5,
            'total_top10': total_top10,
            'total_masked': total_masked,
            'log_probs': all_log_probs
        }


# ============================================================================
# Save Results
# ============================================================================

def save_results(model_name: str, results: Dict):
    """Save evaluation results to a text file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RESULTS_DIR / f'{model_name.lower()}_results.txt'

    if results['total_masked'] > 0:
        top1_acc = results['total_top1'] / results['total_masked']
        top5_acc = results['total_top5'] / results['total_masked']
        top10_acc = results['total_top10'] / results['total_masked']
        perplexity = np.exp(-np.mean(results['log_probs']))
    else:
        top1_acc = top5_acc = top10_acc = perplexity = 0.0

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"EVALUATION RESULTS: {model_name}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Total masked tokens: {results['total_masked']}\n")
        f.write(f"Mask ratio: {MASK_RATIO}\n\n")
        f.write(f"Top-1 Accuracy:  {top1_acc:.4f} ({results['total_top1']}/{results['total_masked']})\n")
        f.write(f"Top-5 Accuracy:  {top5_acc:.4f} ({results['total_top5']}/{results['total_masked']})\n")
        f.write(f"Top-10 Accuracy: {top10_acc:.4f} ({results['total_top10']}/{results['total_masked']})\n")
        f.write(f"Perplexity:      {perplexity:.4f}\n")
        f.write(f"{'=' * 70}\n")

    print(f"Results saved to {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("EVALUATING ALL THREE MODELS ON EVALUATION DATASETS")
    print("=" * 70 + "\n")

    # ========== GraphCodeBERT ==========
    print("\n--- EVALUATING GRAPHCODEBERT ---")
    gcb_data = load_jsonl(GRAPHCODEBERT_JSONL)
    if gcb_data:
        gcb_tokenizer = RobertaTokenizer.from_pretrained(str(GRAPH_PATH))
        gcb_evaluator = GraphCodeBERTEvaluator(str(GRAPH_PATH), gcb_tokenizer)
        gcb_results = gcb_evaluator.evaluate_dataset(gcb_data, MASK_RATIO, TOP_K)
        save_results('GraphCodeBERT', gcb_results)
    else:
        print("Skipping GraphCodeBERT evaluation (no data)")

    # ========== UniXcoder ==========
    print("\n--- EVALUATING UNIXCODER ---")
    uxc_data = load_jsonl(UNIXCODER_JSONL)
    if uxc_data:
        uxc_tokenizer = RobertaTokenizer.from_pretrained(str(UNIX_PATH))
        uxc_evaluator = UniXcoderEvaluator(str(UNIX_PATH), uxc_tokenizer)
        uxc_results = uxc_evaluator.evaluate_dataset(uxc_data, MASK_RATIO, TOP_K)
        save_results('UniXcoder', uxc_results)
    else:
        print("Skipping UniXcoder evaluation (no data)")

    # ========== CodeBERT-cpp ==========
    print("\n--- EVALUATING CODEBERT-CPP ---")
    cbt_data = load_jsonl(CODEBERT_JSONL)
    if cbt_data:
        cbt_tokenizer = RobertaTokenizer.from_pretrained(CODE_PATH)
        cbt_evaluator = CodeBERTcppEvaluator(CODE_PATH, cbt_tokenizer)
        cbt_results = cbt_evaluator.evaluate_dataset(cbt_data, MASK_RATIO, TOP_K)
        save_results('CodeBERT-cpp', cbt_results)
    else:
        print("Skipping CodeBERT-cpp evaluation (no data)")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
