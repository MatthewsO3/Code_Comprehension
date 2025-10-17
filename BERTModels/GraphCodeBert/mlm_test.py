"""
Evaluate a trained GraphCodeBERT model on MLM task with C++ code snippets.
MODIFIED to fetch test snippets directly from the codeparrot database,
and to aggregate metrics for a final summary report.
"""
import torch
import numpy as np
import random
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from transformers import RobertaTokenizer, RobertaForMaskedLM
# ADDED: Import for fetching data from Hugging Face Hub
from datasets import load_dataset

try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp as tscpp
    TS_AVAILABLE = True
    CPP_LANGUAGE = Language(tscpp.language())
    ts_parser = Parser(CPP_LANGUAGE)
except ImportError:
    TS_AVAILABLE = False
    print("Warning: tree_sitter/tree_sitter_cpp not found. DFG extraction will fail.")

random.seed(42)
torch.manual_seed(42)

# Helper function to filter snippets, copied from dfg_extract.py for consistency
def should_keep_code(code: str) -> bool:
    """Filter criteria for C++ code"""
    if not code: return False
    if len(code) < 100 or len(code) > 10000: return False
    lines = code.count('\n')
    if lines < 3 or lines > 500: return False
    if 'void ' not in code and 'int ' not in code and 'class ' not in code and 'std::' not in code: return False
    return True

# Function to fetch test snippets from the database
def fetch_test_snippets_from_db(skip_n: int, take_n: int, tokenizer: RobertaTokenizer) -> List[str]:
    """Fetches valid C++ snippets from the database, skipping the training data."""
    print(f"Fetching {take_n} test snippets from 'codeparrot/github-code-clean', skipping the first {skip_n}...")
    print("Applying filter: Snippets must have FEWER THAN 100 tokens.")

    try:
        dataset = load_dataset("codeparrot/github-code-clean", "C++-all", split="train", streaming=True)

        snippets = []

        # MODIFIED: Combined the two filter conditions into one generator expression
        # It now also checks for the token length.
        filtered_dataset = (
            ex for ex in dataset.skip(skip_n)
            if should_keep_code(ex.get('code')) and
               len(tokenizer.tokenize(ex.get('code'), add_prefix_space=True)) < 100
        )

        # Take the desired number of valid snippets
        for example in filtered_dataset:
            if len(snippets) >= take_n:
                break
            snippets.append(example['code'])

        if not snippets:
            print("\nWARNING: Could not fetch any valid snippets matching the criteria (including token length < 100).")
            print(f"This could be because the first {skip_n + take_n} snippets are low quality or too long.")
            print("Try reducing 'training_data_size' in config.json or checking your internet connection.\n")
        else:
            print(f"Successfully fetched {len(snippets)} snippets for evaluation.")

        return snippets
    except Exception as e:
        print(f"An error occurred while fetching data from the Hub: {e}")
        return []


class MLMEvaluator:
    # MODIFIED: The constructor now accepts a tokenizer instance to avoid loading it twice.
    def __init__(self, model_path: str, tokenizer: RobertaTokenizer, device: str = None):
        if not TS_AVAILABLE: raise RuntimeError("Tree-sitter is required.")
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")
        print(f"Loading model from {model_path}...")
        self.tokenizer = tokenizer # Use the passed tokenizer
        self.model = RobertaForMaskedLM.from_pretrained(model_path).to(self.device).eval()
        print("Model loaded successfully!")

    def extract_dfg_for_snippet(self, code_bytes: bytes) -> List[Tuple]:
        tree = ts_parser.parse(code_bytes)
        root = tree.root_node
        defs, uses = defaultdict(list), defaultdict(list)
        tokens, node_map = [], {}

        def find_tokens(node):
            if node.type in ['identifier', 'field_identifier']:
                if id(node) not in node_map:
                    node_map[id(node)] = len(tokens)
                    tokens.append(node)
            for child in node.children: find_tokens(child)

        find_tokens(root)

        def is_def(node):
            p = node.parent
            return p and (p.type in ['declaration', 'init_declarator', 'parameter_declaration'] or \
                   (p.type == 'assignment_expression' and node == p.child_by_field_name('left')))

        def find_vars(node):
            if node.type in ['identifier', 'field_identifier']:
                name = code_bytes[node.start_byte:node.end_byte].decode('utf8', 'ignore')
                pos = node_map.get(id(node), -1)
                if pos != -1: (defs if is_def(node) else uses)[name].append(pos)
            for child in node.children: find_vars(child)

        find_vars(root)
        edges = []
        for name, use_positions in uses.items():
            def_positions = sorted(defs.get(name, []))
            for use_pos in use_positions:
                preds = [d for d in def_positions if d < use_pos]
                if preds: edges.append((name, use_pos, "comesFrom", [name], [preds[-1]]))
        return edges

    def preprocess_for_graphcodebert(self, code: str, masked_code_tokens: List[str]):
        dfg = self.extract_dfg_for_snippet(code.encode('utf8'))
        adj, nodes, node_map = defaultdict(list), [], {}
        for var, use_pos, _, _, dep_list in dfg:
            if use_pos not in node_map: node_map[use_pos] = len(nodes); nodes.append((var, use_pos))
            for def_pos in dep_list:
                if def_pos not in node_map: node_map[def_pos] = len(nodes); nodes.append((var, def_pos))
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
        for i in range(len(ids)): mask[i, i] = True
        for i, (_, code_pos) in enumerate(nodes):
            if code_pos < len(masked_code_tokens):
                dfg_abs, code_abs = dfg_start + i, code_pos + 1
                mask[dfg_abs, code_abs] = mask[code_abs, dfg_abs] = True
        for i, adjs in adj.items():
            for j in adjs:
                u, v = dfg_start + i, dfg_start + j
                mask[u, v] = mask[v, u] = True

        return {
            'input_ids': torch.tensor([ids]), 'attention_mask': torch.tensor([mask.tolist()]),
            'position_ids': torch.tensor([pos_ids])
        }

    def evaluate_snippet(self, code: str, mask_ratio: float, top_k: int) -> Dict:
        """
        Evaluates a single snippet and returns raw numbers for aggregation.
        Still prints the detailed predictions for inspection.
        """
        print("\n" + "="*80 + "\nORIGINAL CODE:\n" + "-"*80 + f"\n{code.strip()}")
        # Use add_prefix_space=True for consistency with training data
        code_tokens = self.tokenizer.tokenize(code, add_prefix_space=True)
        if not code_tokens: return None

        num_mask = max(1, int(len(code_tokens) * mask_ratio))
        mask_pos = sorted(random.sample(range(len(code_tokens)), num_mask))
        orig_tokens = [code_tokens[i] for i in mask_pos]
        masked_tokens = code_tokens.copy()
        for pos in mask_pos: masked_tokens[pos] = self.tokenizer.mask_token
        print("\n" + "="*80 + "\nMASKED CODE:\n" + "-"*80 + f"\n{self.tokenizer.convert_tokens_to_string(masked_tokens)}")

        inputs = self.preprocess_for_graphcodebert(code, masked_tokens)
        with torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits

        print("\n" + "="*80 + "\nPREDICTIONS:\n" + "-"*80)
        top1_correct, top5_correct, log_probs = 0, 0, []
        for i, pos in enumerate(mask_pos):
            probs = torch.softmax(logits[0, pos + 1], dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)

            orig_token = orig_tokens[i]
            print(f"\nPosition {pos} (original: '{orig_token}'):")

            top_preds = self.tokenizer.convert_ids_to_tokens(top_indices)

            # Find the probability of the correct token for perplexity calculation
            correct_token_prob = 1e-9 # A small value for cases where the correct token is not in top_k
            found_top5 = False
            for rank, (pred, prob) in enumerate(zip(top_preds, top_probs), 1):
                marker = "✓" if pred == orig_token else " "
                if marker == "✓":
                    correct_token_prob = prob.item()
                    if not found_top5:
                        top5_correct += 1
                        found_top5 = True
                    if rank == 1:
                        top1_correct += 1
                print(f"    {rank}. {marker} '{pred}' (prob: {prob:.4f})")

            log_probs.append(np.log(correct_token_prob))

        # MODIFIED: Return raw numbers instead of calculated metrics
        return {
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'num_masked': num_mask,
            'log_probs': log_probs
        }

# This list is now a fallback if use_database_snippets is false
CPP_SNIPPETS_FALLBACK = [
    "int fibonacci(int n) {\n    if (n <= 1) return n;\n    int a = fibonacci(n - 1);\n    int b = fibonacci(n - 2);\n    return a + b;\n}"
]

def main():
    parser = argparse.ArgumentParser(description='Evaluate GraphCodeBERT MLM model on C++')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--use_database_snippets', action=argparse.BooleanOptionalAction)
    parser.add_argument('--training_data_size', type=int, default=None)
    parser.add_argument('--num_database_snippets', type=int, default=None)

    config_from_file = {}
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config_from_file = json.load(f).get("evaluate", {})
    parser.set_defaults(**config_from_file)
    args = parser.parse_args()

    if not args.model_path or not Path(args.model_path).exists():
        parser.error("A valid 'model_path' must be specified in config.json or via arguments.")

    print("\n--- Running evaluation with configuration ---")
    for k, v in vars(args).items(): print(f"  {k}: {v}")
    print("-------------------------------------------\n")

    # MODIFIED: Load tokenizer once here to pass it to helper functions
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)

    snippets_to_evaluate = []
    if args.use_database_snippets:
        if args.training_data_size is None or args.num_database_snippets is None:
            parser.error("'training_data_size' and 'num_database_snippets' must be set in config.json when using database snippets.")
        # MODIFIED: Pass the loaded tokenizer to the fetch function
        snippets_to_evaluate = fetch_test_snippets_from_db(
            skip_n=args.training_data_size,
            take_n=args.num_database_snippets,
            tokenizer=tokenizer
        )
    else:
        print("Using hardcoded C++ snippets for evaluation (fallback).")
        snippets_to_evaluate = CPP_SNIPPETS_FALLBACK

    if not snippets_to_evaluate:
        print("No snippets to evaluate. Exiting.")
        return

    # MODIFIED: Pass the loaded tokenizer to the evaluator
    evaluator = MLMEvaluator(args.model_path, tokenizer=tokenizer)

    # Aggregate results in a dictionary
    aggregated_results = {
        'total_top1_correct': 0,
        'total_top5_correct': 0,
        'total_masked': 0,
        'all_log_probs': []
    }

    for i, snippet in enumerate(snippets_to_evaluate, 1):
        print(f"\n\n{'#' * 35} SNIPPET {i}/{len(snippets_to_evaluate)} {'#' * 35}")
        results = evaluator.evaluate_snippet(snippet, args.mask_ratio, args.top_k)
        if results:
            aggregated_results['total_top1_correct'] += results['top1_correct']
            aggregated_results['total_top5_correct'] += results['top5_correct']
            aggregated_results['total_masked'] += results['num_masked']
            aggregated_results['all_log_probs'].extend(results['log_probs'])

    # Calculate and print final summary if any snippets were evaluated
    if aggregated_results['total_masked'] > 0:
        total_masked = aggregated_results['total_masked']

        # Calculate final metrics
        overall_top1_acc = aggregated_results['total_top1_correct'] / total_masked
        overall_top5_acc = aggregated_results['total_top5_correct'] / total_masked
        overall_perplexity = np.exp(-np.mean(aggregated_results['all_log_probs']))

        print("\n\n" + "#" * 80)
        print("#" + " " * 29 + "OVERALL STATISTICS" + " " * 29 + "#")
        print("#" * 80)
        print(f"  Snippets evaluated: {len(snippets_to_evaluate)}")
        print(f"  Total masked tokens: {total_masked}")
        print("-" * 80)
        print(f"  Overall Top-1 Accuracy: {overall_top1_acc:.2%} ({aggregated_results['total_top1_correct']}/{total_masked})")
        print(f"  Overall Top-5 Accuracy: {overall_top5_acc:.2%} ({aggregated_results['total_top5_correct']}/{total_masked})")
        print(f"  Overall Perplexity:     {overall_perplexity:.4f}")
        print("#" * 80)

if __name__ == "__main__":
    main()

