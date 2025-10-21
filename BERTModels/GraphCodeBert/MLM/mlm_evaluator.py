"""
Evaluate a trained GraphCodeBERT model on MLM task with custom C++ code snippets.
Supports 3 evaluation methods:
1. Random masking with specified ratio
2. Sequential token masking (one token at a time)
3. Masking all occurrences of a specific token
"""
import torch
import numpy as np
import random
import json
from pathlib import Path
from typing import List, Tuple, Dict
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
    print("Warning: tree_sitter/tree_sitter_cpp not found. DFG extraction will fail.")

random.seed(42)
torch.manual_seed(42)


class MLMEvaluator:
    def __init__(self, model_path: str, tokenizer: RobertaTokenizer, device: str = None):
        if not TS_AVAILABLE:
            raise RuntimeError("Tree-sitter is required.")
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")
        print(f"Loading model from {model_path}...")
        self.tokenizer = tokenizer
        self.model = RobertaForMaskedLM.from_pretrained(model_path).to(self.device).eval()
        print("Model loaded successfully!\n")

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
            for child in node.children:
                find_tokens(child)

        find_tokens(root)

        def is_def(node):
            p = node.parent
            return p and (p.type in ['declaration', 'init_declarator', 'parameter_declaration'] or \
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

    def _process_predictions(self, logits, mask_pos, orig_tokens, top_k):
        """Helper to process predictions and calculate metrics."""
        print("\n" + "=" * 80)
        print("PREDICTIONS:")
        print("-" * 80)

        top1_correct, top5_correct, log_probs = 0, 0, []

        for i, pos in enumerate(mask_pos):
            probs = torch.softmax(logits[0, pos + 1], dim=-1)
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

            orig_token = orig_tokens[i]
            print(f"\nPosition {pos} (original: '{orig_token}'):")

            top_preds = self.tokenizer.convert_ids_to_tokens(top_indices)

            correct_token_prob = 1e-9
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

        return {
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'num_masked': len(mask_pos),
            'log_probs': log_probs
        }

    def evaluate_random_mask(self, code: str, mask_ratio: float, top_k: int = 5) -> Dict:
        """Method 1: Mask random tokens with specified ratio."""
        print("\n" + "=" * 80)
        print("METHOD 1: RANDOM MASKING")
        print("=" * 80)
        print("\nORIGINAL CODE:")
        print("-" * 80)
        print(code.strip())

        code_tokens = self.tokenizer.tokenize(code, add_prefix_space=True)
        if not code_tokens:
            return None

        num_mask = max(1, int(len(code_tokens) * mask_ratio))
        mask_pos = sorted(random.sample(range(len(code_tokens)), num_mask))
        orig_tokens = [code_tokens[i] for i in mask_pos]
        masked_tokens = code_tokens.copy()
        for pos in mask_pos:
            masked_tokens[pos] = self.tokenizer.mask_token

        print("\nMASKED CODE:")
        print("-" * 80)
        print(self.tokenizer.convert_tokens_to_string(masked_tokens))

        inputs = self.preprocess_for_graphcodebert(code, masked_tokens)
        with torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits

        results = self._process_predictions(logits, mask_pos, orig_tokens, top_k)
        return results

    def evaluate_sequential_mask(self, code: str, top_k: int = 5) -> Dict:
        """Method 2: Mask each token one by one and predict it."""
        print("\n" + "=" * 80)
        print("METHOD 2: SEQUENTIAL MASKING (one token at a time)")
        print("=" * 80)
        print("\nORIGINAL CODE:")
        print("-" * 80)
        print(code.strip())

        code_tokens = self.tokenizer.tokenize(code, add_prefix_space=True)
        if not code_tokens:
            return None

        aggregated_top1, aggregated_top5, aggregated_log_probs = 0, 0, []

        for token_idx in range(len(code_tokens)):
            masked_tokens = code_tokens.copy()
            masked_tokens[token_idx] = self.tokenizer.mask_token

            print("\n" + "-" * 80)
            print(f"Masking token at position {token_idx}:")
            print(f"Masked code: {self.tokenizer.convert_tokens_to_string(masked_tokens)}")

            inputs = self.preprocess_for_graphcodebert(code, masked_tokens)
            with torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits

            results = self._process_predictions(logits, [token_idx], [code_tokens[token_idx]], top_k)
            aggregated_top1 += results['top1_correct']
            aggregated_top5 += results['top5_correct']
            aggregated_log_probs.extend(results['log_probs'])

        return {
            'top1_correct': aggregated_top1,
            'top5_correct': aggregated_top5,
            'num_masked': len(code_tokens),
            'log_probs': aggregated_log_probs
        }

    def evaluate_specific_token_mask(self, code: str, token_to_mask: str, top_k: int = 5) -> Dict:
        """Method 3: Mask all occurrences of a specific token and predict each."""
        print("\n" + "=" * 80)
        print(f"METHOD 3: MASKING ALL '{token_to_mask}' TOKENS")
        print("=" * 80)
        print("\nORIGINAL CODE:")
        print("-" * 80)
        print(code.strip())

        code_tokens = self.tokenizer.tokenize(code, add_prefix_space=True)
        if not code_tokens:
            return None

        # Find all positions of the token to mask
        mask_pos = [i for i, token in enumerate(code_tokens) if token == token_to_mask]

        if not mask_pos:
            print(f"\nWarning: Token '{token_to_mask}' not found in code.")
            return None

        print(f"\nFound {len(mask_pos)} occurrence(s) of '{token_to_mask}' at positions: {mask_pos}")

        masked_tokens = code_tokens.copy()
        for pos in mask_pos:
            masked_tokens[pos] = self.tokenizer.mask_token

        print("\nMASKED CODE:")
        print("-" * 80)
        print(self.tokenizer.convert_tokens_to_string(masked_tokens))

        inputs = self.preprocess_for_graphcodebert(code, masked_tokens)
        with torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits

        orig_tokens = [code_tokens[i] for i in mask_pos]
        results = self._process_predictions(logits, mask_pos, orig_tokens, top_k)
        return results

    def print_summary(self, results: Dict, method_name: str):
        """Print summary statistics."""
        if not results or results['num_masked'] == 0:
            return

        total_masked = results['num_masked']
        top1_acc = results['top1_correct'] / total_masked
        top5_acc = results['top5_correct'] / total_masked
        perplexity = np.exp(-np.mean(results['log_probs']))

        print("\n" + "#" * 80)
        print("#" + " " * 20 + f"SUMMARY: {method_name}" + " " * (39 - len(method_name)) + "#")
        print("#" * 80)
        print(f"  Total masked tokens: {total_masked}")
        print("-" * 80)
        print(f"  Top-1 Accuracy: {top1_acc:.2%} ({results['top1_correct']}/{total_masked})")
        print(f"  Top-5 Accuracy: {top5_acc:.2%} ({results['top5_correct']}/{total_masked})")
        print(f"  Perplexity:     {perplexity:.4f}")
        print("#" * 80 + "\n")


def main():
    # Load configuration
    config_path = '/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/GraphCodeBert/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f).get('mlm_eval', {})

    model_path = config.get('model_path', './model')
    device = config.get('device', None)

    # Load tokenizer and model
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    evaluator = MLMEvaluator(model_path, tokenizer, device)

    # Example C++ code snippet
    cpp_snippet = """
#include <type_traits>
#include <iostream>

template <typename T>
constexpr auto cpp_magic(T&& x) noexcept {
    if constexpr (std::is_integral_v<std::decay_t<T>>) {
        return x * x; // squares integers
    } else if constexpr (std::is_floating_point_v<std::decay_t<T>>) {
        return x / 2.0; // halves floats
    } else {
        static_assert(std::is_same_v<T, void>, "Unsupported type!");
    }
}

int main() {
    std::cout << cpp_magic(5) << '\n';     // 25
    std::cout << cpp_magic(3.14) << '\n';  // 1.57
}
"""

    # Method 1: Random masking
    """
    results1 = evaluator.evaluate_random_mask(cpp_snippet, mask_ratio=0.5, top_k=5)
    if results1:
        evaluator.print_summary(results1, "Random Masking (50%)")
    # Method 2: Sequential masking
    results2 = evaluator.evaluate_sequential_mask(cpp_snippet, top_k=5)
    if results2:
        evaluator.print_summary(results2, "Sequential Masking")
    """
    # Method 3: Mask specific token - Ġ
    results3 = evaluator.evaluate_specific_token_mask(cpp_snippet, token_to_mask='Ġx', top_k=5)
    if results3:
        evaluator.print_summary(results3, "Specific Token Masking (result)")


if __name__ == "__main__":
    main()