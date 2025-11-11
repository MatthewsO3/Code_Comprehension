"""
Evaluate UniXcoder MLM model on C++ code snippets.
Fetches test snippets from the codeparrot database and aggregates metrics.
Includes AST (Abstract Syntax Tree) extraction using tree-sitter.
"""
import torch
import numpy as np
import random
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from transformers import RobertaTokenizer, RobertaForMaskedLM
from datasets import load_dataset

try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp as tscpp
    TS_AVAILABLE = True
    CPP_LANGUAGE = Language(tscpp.language())
    ts_parser = Parser(CPP_LANGUAGE)
except ImportError:
    TS_AVAILABLE = False
    print("Warning: tree_sitter not available. AST extraction will fail.")

random.seed(42)
torch.manual_seed(42)


# ============================================================================
# AST Extraction
# ============================================================================

def extract_ast_sequence(tree) -> List[str]:
    """
    Extract AST node sequence from tree-sitter parse tree.
    Performs DFS traversal and collects node types.
    """
    if not TS_AVAILABLE:
        return []

    ast_nodes = []

    def traverse(node):
        ast_nodes.append(node.type)
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return ast_nodes


def extract_ast_for_code(code: str) -> List[str]:
    """Extract AST from code string."""
    if not TS_AVAILABLE:
        return []

    try:
        code_bytes = code.encode('utf8')
        tree = ts_parser.parse(code_bytes)
        return extract_ast_sequence(tree)
    except Exception:
        return []


# ============================================================================
# Filtering and Data Loading
# ============================================================================

def should_keep_code(code: str) -> bool:
    """Filter criteria for C++ code"""
    if not code:
        return False
    if len(code) < 100 or len(code) > 10000:
        return False
    lines = code.count('\n')
    if lines < 3 or lines > 500:
        return False
    if 'void ' not in code and 'int ' not in code and 'class ' not in code and 'std::' not in code:
        return False
    return True


def fetch_test_snippets_from_db(skip_n: int, take_n: int, tokenizer: RobertaTokenizer) -> List[Dict]:
    """
    Fetches valid C++ snippets from the database, skipping the training data.
    Returns list of dicts with 'code' and 'ast' fields.
    """
    print(f"Fetching {take_n} test snippets from 'codeparrot/github-code-clean', skipping the first {skip_n}...")
    print("Applying filter: Snippets must have FEWER THAN 100 tokens.")

    try:
        dataset = load_dataset("codeparrot/github-code-clean", "C++-all", split="train", streaming=True)
        snippets = []

        filtered_dataset = (
            ex for ex in dataset.skip(skip_n)
            if should_keep_code(ex.get('code')) and
               len(tokenizer.tokenize(ex.get('code'), add_prefix_space=True)) < 100
        )

        for example in filtered_dataset:
            if len(snippets) >= take_n:
                break
            code = example['code']
            ast = extract_ast_for_code(code)
            snippets.append({
                'code': code,
                'ast': ast
            })

        if not snippets:
            print("\nWARNING: Could not fetch any valid snippets.")
            print(f"Try reducing 'training_data_size' in config.json.\n")
        else:
            print(f"Successfully fetched {len(snippets)} snippets for evaluation.")

        return snippets
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return []


# ============================================================================
# UniXcoder Evaluator with AST
# ============================================================================

class UniXcoderMLMEvaluator:
    """Evaluator for UniXcoder MLM task with AST support."""

    def __init__(self, model_path: str, tokenizer: RobertaTokenizer, device: str = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")
        print(f"Loading model from {model_path}...")

        self.tokenizer = tokenizer
        self.model = RobertaForMaskedLM.from_pretrained(model_path).to(self.device).eval()
        print("Model loaded successfully!")

    def evaluate_snippet(self, code: str, ast: List[str], mask_ratio: float, top_k: int) -> Dict:
        """
        Evaluates a single snippet and returns raw numbers for aggregation.
        AST is included for logging/analysis but not used in the model forward pass.
        """
        # Tokenize with add_prefix_space=True for consistency
        code_tokens = self.tokenizer.tokenize(code, add_prefix_space=True)
        if not code_tokens:
            return None

        # Calculate number of tokens to mask
        num_mask = max(1, int(len(code_tokens) * mask_ratio))
        mask_positions = sorted(random.sample(range(len(code_tokens)), num_mask))

        # Store original tokens
        original_tokens = [code_tokens[i] for i in mask_positions]

        # Create masked version
        masked_tokens = code_tokens.copy()
        for pos in mask_positions:
            masked_tokens[pos] = self.tokenizer.mask_token

        # --- MODIFICATION START: Print Original and Masked Code ---
        masked_code_display = self.tokenizer.convert_tokens_to_string(masked_tokens)


       # print("\n" + "="*80 + "\nSNIPPET DETAILS:")
       # print(f"\nOriginal Code:\n{code}")
       # print(f"\nMasked Code:\n{masked_code_display}")
       # print("\n" + "-"*80)
        # --- MODIFICATION END ---

        # Prepare input: [CLS] tokens [SEP]
        input_tokens = [self.tokenizer.cls_token] + masked_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Convert to tensors
        input_ids_tensor = torch.tensor([input_ids]).to(self.device)
        attention_mask_tensor = torch.tensor([attention_mask]).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor
            )
            logits = outputs.logits

        # Evaluate predictions
        top1_correct, top5_correct, log_probs = 0, 0, []

        for i, pos in enumerate(mask_positions):
            # +1 because of [CLS] token at the beginning
            actual_pos = pos + 1

            # Get probabilities for this position
            probs = torch.softmax(logits[0, actual_pos], dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)

            original_token = original_tokens[i]
            top_predictions = self.tokenizer.convert_ids_to_tokens(top_indices)

            # Calculate metrics
            correct_token_prob = 1e-9  # Small value for cases where correct token not in top-k
            found_top5 = False

            for rank, (pred, prob) in enumerate(zip(top_predictions, top_probs), 1):
                if pred == original_token:
                    correct_token_prob = prob.item()
                    if not found_top5:
                        top5_correct += 1
                        found_top5 = True
                    if rank == 1:
                        top1_correct += 1
                    break


            log_probs.append(np.log(correct_token_prob))

        return {
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'num_masked': num_mask,
            'log_probs': log_probs
        }


# Fallback snippets
CPP_SNIPPETS_FALLBACK = [
    "int fibonacci(int n) {\n    if (n <= 1) return n;\n    int a = fibonacci(n - 1);\n    int b = fibonacci(n - 2);\n    return a + b;\n}"
]


def main():
    parser = argparse.ArgumentParser(description='Evaluate UniXcoder MLM model on C++')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--use_database_snippets', action=argparse.BooleanOptionalAction)
    parser.add_argument('--training_data_size', type=int, default=None)
    parser.add_argument('--num_database_snippets', type=int, default=None)

    # Load config from file
    config_from_file = {}
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config_from_file = json.load(f).get("evaluate", {})

    parser.set_defaults(**config_from_file)
    args = parser.parse_args()

    if not args.model_path or not Path(args.model_path).exists():
        parser.error("A valid 'model_path' must be specified in config.json or via arguments.")

    print("\n--- Running evaluation with configuration ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("-------------------------------------------\n")

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)

    # Get test snippets
    snippets_to_evaluate = []
    if args.use_database_snippets:
        if args.training_data_size is None or args.num_database_snippets is None:
            parser.error("'training_data_size' and 'num_database_snippets' must be set when using database snippets.")
        snippets_to_evaluate = fetch_test_snippets_from_db(
            skip_n=args.training_data_size,
            take_n=args.num_database_snippets,
            tokenizer=tokenizer
        )
    else:
        print("Using hardcoded C++ snippets for evaluation (fallback).")
        snippets_to_evaluate = [{'code': s, 'ast': extract_ast_for_code(s)} for s in CPP_SNIPPETS_FALLBACK]

    if not snippets_to_evaluate:
        print("No snippets to evaluate. Exiting.")
        return

    # Create evaluator
    evaluator = UniXcoderMLMEvaluator(args.model_path, tokenizer=tokenizer)

    # Aggregate results
    aggregated_results = {
        'total_top1_correct': 0,
        'total_top5_correct': 0,
        'total_masked': 0,
        'all_log_probs': []
    }

    for i, snippet_data in enumerate(snippets_to_evaluate, 1):
        code = snippet_data['code']
        ast = snippet_data.get('ast', [])
        results = evaluator.evaluate_snippet(code, ast, args.mask_ratio, args.top_k)
        if results:
            aggregated_results['total_top1_correct'] += results['top1_correct']
            aggregated_results['total_top5_correct'] += results['top5_correct']
            aggregated_results['total_masked'] += results['num_masked']
            aggregated_results['all_log_probs'].extend(results['log_probs'])

    # Calculate and print final summary
    if aggregated_results['total_masked'] > 0:
        total_masked = aggregated_results['total_masked']

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
