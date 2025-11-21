"""
Cross-Language MLM Evaluation for GraphCodeBERT
Evaluates the trained C++ GraphCodeBERT model on original training languages:
Java, Python, and JavaScript to measure transfer and degradation.

This helps understand:
1. How well does C++-trained model transfer to other languages?
2. Performance comparison: C++ vs Java vs Python vs JavaScript
"""

import torch
import numpy as np
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from transformers import RobertaTokenizer, RobertaForMaskedLM
from datasets import load_dataset
from tqdm import tqdm

try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp as tscpp
    import tree_sitter_java as tsjava
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjs

    TS_AVAILABLE = True
except ImportError:
    TS_AVAILABLE = False
    print("Warning: tree_sitter language packages not all available.")

random.seed(42)
torch.manual_seed(42)

# ============================================================================
# Language Setup
# ============================================================================

LANGUAGES = {
    'java': {
        'dataset_name': 'Java-all',
        'tree_sitter_module': tsjava if TS_AVAILABLE else None,
        'filter_keywords': ['public', 'private', 'class', 'void', 'return'],
    },
    'python': {
        'dataset_name': 'Python-all',
        'tree_sitter_module': tspython if TS_AVAILABLE else None,
        'filter_keywords': ['def', 'class', 'return', 'import', 'from'],
    },
    'javascript': {
        'dataset_name': 'JavaScript-all',
        'tree_sitter_module': tsjs if TS_AVAILABLE else None,
        'filter_keywords': ['function', 'class', 'const', 'let', 'return'],
    },
}
MODEL_PATH = "/home/mczap/code_comp/Code_Comprehension/BERTModels/GraphCodeBert/MLM/GraphCodeBert/Models/graphcodebert-cpp-mlm-from-config/best_model"
#MODEL_PATH = "/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/GraphCodeBert/GraphCodeBert/Models/graphcodebert-cpp-mlm-from-config/best_model"
RESULTS_DIR = Path('results/cross_language')
MASK_RATIO = 0.2
TOP_K = 10


# ============================================================================
# DFG Extraction (Language-Agnostic)
# ============================================================================

class DFGExtractor:
    """Extract DFG for different languages using tree-sitter."""

    def __init__(self, language_name: str):
        self.language_name = language_name
        lang_config = LANGUAGES[language_name]

        if not lang_config['tree_sitter_module']:
            raise RuntimeError(f"Tree-sitter module for {language_name} not available")

        self.parser = Parser(Language(lang_config['tree_sitter_module'].language()))

    def extract_dfg(self, code_bytes: bytes) -> List[Tuple]:
        """Extract DFG from code (language-agnostic approach)."""
        tree = self.parser.parse(code_bytes)
        root = tree.root_node

        defs, uses = defaultdict(list), defaultdict(list)
        tokens, node_map = [], {}

        def find_tokens(node):
            if node.type == 'identifier':
                if id(node) not in node_map:
                    node_map[id(node)] = len(tokens)
                    tokens.append(node)
            for child in node.children:
                find_tokens(child)

        find_tokens(root)

        def is_def(node):
            p = node.parent
            if not p:
                return False
            # Language-agnostic definition patterns
            def_patterns = ['declaration', 'init_declarator', 'parameter_declaration',
                            'function_declaration', 'variable_declarator', 'formal_parameter']
            if any(pat in p.type for pat in def_patterns):
                return True
            if 'assignment' in p.type and node == p.child_by_field_name('left'):
                return True
            return False

        def find_vars(node):
            if node.type == 'identifier':
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


# ============================================================================
# Dataset Loading
# ============================================================================

def fetch_language_snippets(language: str, num_samples: int = 1000,
                            tokenizer: RobertaTokenizer = None) -> List[str]:
    """Fetch code snippets for a specific language."""
    if language not in LANGUAGES:
        print(f"Error: Language '{language}' not supported")
        return []

    dataset_name = LANGUAGES[language]['dataset_name']
    print(f"\nFetching {num_samples} {language.upper()} snippets from '{dataset_name}'...")

    try:
        dataset = load_dataset("codeparrot/github-code-clean", dataset_name,
                               split="train", streaming=True)

        snippets = []
        for example in dataset:
            if len(snippets) >= num_samples:
                break

            code = example.get('code')
            if not code:
                continue

            # Length filters
            if len(code) < 100 or len(code) > 10000:
                continue

            lines = code.count('\n')
            if lines < 3 or lines > 500:
                continue

            # Language-specific filters
            keywords = LANGUAGES[language]['filter_keywords']
            if not any(kw in code for kw in keywords):
                continue

            # Token length filter
            if tokenizer:
                tokens = tokenizer.tokenize(code, add_prefix_space=True)
                if len(tokens) < 10 or len(tokens) > 450:
                    continue

            snippets.append(code)

        print(f"Successfully fetched {len(snippets)} valid {language} snippets")
        return snippets

    except Exception as e:
        print(f"Error fetching {language} snippets: {e}")
        return []


# ============================================================================
# GraphCodeBERT Cross-Language Evaluator
# ============================================================================

class CrossLanguageEvaluator:
    def __init__(self, model_path: str, tokenizer: RobertaTokenizer, device: str = None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Loading GraphCodeBERT from {model_path}...")
        self.tokenizer = tokenizer
        self.model = RobertaForMaskedLM.from_pretrained(model_path).to(self.device).eval()
        print("Model loaded successfully!")

        self.dfg_extractors = {}

    def get_dfg_extractor(self, language: str) -> DFGExtractor:
        """Get or create DFG extractor for language."""
        if language not in self.dfg_extractors:
            self.dfg_extractors[language] = DFGExtractor(language)
        return self.dfg_extractors[language]

    def preprocess_for_graphcodebert(self, code: str, masked_code_tokens: List[str],
                                     language: str):
        """Prepare input with DFG attention mask."""
        dfg = []
        if TS_AVAILABLE:
            try:
                extractor = self.get_dfg_extractor(language)
                dfg = extractor.extract_dfg(code.encode('utf8'))
            except Exception as e:
                # Fallback: no DFG if extraction fails
                pass

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

    def evaluate_language(self, language: str, snippets: List[str],
                          mask_ratio: float, top_k: int) -> Dict:
        """Evaluate model on language-specific snippets."""
        total_top1, total_top5, total_top10 = 0, 0, 0
        total_masked = 0
        all_log_probs = []
        failed_samples = 0

        print(f"\nEvaluating on {len(snippets)} {language.upper()} snippets...")

        for idx, code in enumerate(tqdm(snippets, desc=f"Evaluating {language}")):
            code_tokens = self.tokenizer.tokenize(code, add_prefix_space=True)

            if not code_tokens or len(code_tokens) < 5:
                continue

            num_mask = max(1, int(len(code_tokens) * mask_ratio))
            mask_positions = sorted(random.sample(range(len(code_tokens)),
                                                  min(num_mask, len(code_tokens))))
            original_tokens = [code_tokens[i] for i in mask_positions]

            masked_tokens = code_tokens.copy()
            for pos in mask_positions:
                masked_tokens[pos] = self.tokenizer.mask_token

            try:
                inputs = self.preprocess_for_graphcodebert(code, masked_tokens, language)
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
                failed_samples += 1
                continue

        results = {
            'language': language,
            'total_top1': total_top1,
            'total_top5': total_top5,
            'total_top10': total_top10,
            'total_masked': total_masked,
            'log_probs': all_log_probs,
            'failed_samples': failed_samples
        }

        return results


# ============================================================================
# Results Saving
# ============================================================================

def save_results(all_results: Dict[str, Dict]):
    """Save evaluation results across all languages."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Summary file
    summary_file = RESULTS_DIR / 'cross_language_summary.txt'

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CROSS-LANGUAGE MLM EVALUATION: C++-Trained GraphCodeBERT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Mask Ratio: {MASK_RATIO}\n")
        f.write(f"Top-K: {TOP_K}\n\n")

        f.write("=" * 80 + "\n")
        f.write("RESULTS BY LANGUAGE\n")
        f.write("=" * 80 + "\n\n")

        for language, results in all_results.items():
            if results['total_masked'] > 0:
                top1_acc = results['total_top1'] / results['total_masked']
                top5_acc = results['total_top5'] / results['total_masked']
                top10_acc = results['total_top10'] / results['total_masked']
                perplexity = np.exp(-np.mean(results['log_probs']))
            else:
                top1_acc = top5_acc = top10_acc = perplexity = 0.0

            f.write(f"\n{language.upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total masked tokens: {results['total_masked']}\n")
            f.write(f"  Failed samples: {results['failed_samples']}\n")
            f.write(f"  Top-1 Accuracy:  {top1_acc:.4f} ({results['total_top1']}/{results['total_masked']})\n")
            f.write(f"  Top-5 Accuracy:  {top5_acc:.4f} ({results['total_top5']}/{results['total_masked']})\n")
            f.write(f"  Top-10 Accuracy: {top10_acc:.4f} ({results['total_top10']}/{results['total_masked']})\n")
            f.write(f"  Perplexity:      {perplexity:.4f}\n")

        # Comparison table
        f.write("\n" + "=" * 80 + "\n")
        f.write("COMPARISON TABLE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Language':<15} {'Top-1':<12} {'Top-5':<12} {'Top-10':<12} {'Perplexity':<12}\n")
        f.write("-" * 80 + "\n")

        for language, results in all_results.items():
            if results['total_masked'] > 0:
                top1_acc = results['total_top1'] / results['total_masked']
                top5_acc = results['total_top5'] / results['total_masked']
                top10_acc = results['total_top10'] / results['total_masked']
                perplexity = np.exp(-np.mean(results['log_probs']))
            else:
                top1_acc = top5_acc = top10_acc = perplexity = 0.0

            f.write(
                f"{language.upper():<15} {top1_acc:<12.4f} {top5_acc:<12.4f} {top10_acc:<12.4f} {perplexity:<12.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nResults saved to {summary_file}")

    # JSON results for further analysis
    json_file = RESULTS_DIR / 'cross_language_detailed.json'
    json_results = {}
    for lang, res in all_results.items():
        json_results[lang] = {
            'total_top1': res['total_top1'],
            'total_top5': res['total_top5'],
            'total_top10': res['total_top10'],
            'total_masked': res['total_masked'],
            'failed_samples': res['failed_samples'],
            'top1_acc': res['total_top1'] / res['total_masked'] if res['total_masked'] > 0 else 0,
            'top5_acc': res['total_top5'] / res['total_masked'] if res['total_masked'] > 0 else 0,
            'top10_acc': res['total_top10'] / res['total_masked'] if res['total_masked'] > 0 else 0,
            'perplexity': float(np.exp(-np.mean(res['log_probs']))) if res['log_probs'] else 0.0,
        }

    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Detailed results saved to {json_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("CROSS-LANGUAGE MLM EVALUATION: C++-TRAINED GRAPHCODEBERT")
    print("=" * 80)

    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_PATH}...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    print("✓ Tokenizer loaded")

    # Initialize evaluator
    evaluator = CrossLanguageEvaluator(MODEL_PATH, tokenizer)

    all_results = {}

    # Evaluate on each language
    for language in LANGUAGES.keys():
        print(f"\n{'=' * 80}")
        print(f"EVALUATING {language.upper()}")
        print(f"{'=' * 80}")

        try:
            snippets = fetch_language_snippets(language, num_samples=100,
                                               tokenizer=tokenizer)

            if not snippets:
                print(f"Skipping {language} - no valid snippets")
                continue

            results = evaluator.evaluate_language(language, snippets,
                                                  MASK_RATIO, TOP_K)
            all_results[language] = results

        except Exception as e:
            print(f"Error evaluating {language}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    if all_results:
        save_results(all_results)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
