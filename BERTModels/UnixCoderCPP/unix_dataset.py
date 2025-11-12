"""
UniXcoder Dataset Preparation for C++ Code
Streams the-stack dataset and prepares data for UniXcoder training
UniXcoder doesn't use DFG - it uses simpler tokenization with attention masks
Includes AST (Abstract Syntax Tree) extraction using tree-sitter
"""

import json
from pathlib import Path
from typing import List
from datasets import load_dataset
from transformers import RobertaTokenizer
from tqdm import tqdm

try:
    from tree_sitter import Language, Parser
    import tree_sitter_cpp as tscpp
    TS_AVAILABLE = True
    CPP_LANGUAGE = Language(tscpp.language())
    ts_parser = Parser(CPP_LANGUAGE)
except ImportError:
    TS_AVAILABLE = False
    print("Warning: tree_sitter not available. AST extraction will fail.")

# Load UniXcoder tokenizer - use unixcoder-base-nine which is trained on C++
tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
print("✓ UniXcoder tokenizer loaded")


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
# Preprocessing
# ============================================================================

def preprocess_code(code: str, idx: int) -> dict:
    """
    Preprocess C++ code for UniXcoder.
    UniXcoder is simpler than GraphCodeBERT - no DFG needed.
    Includes AST extraction.
    """
    try:
        # UniXcoder uses standard tokenization with add_prefix_space=True
        tokens = tokenizer.tokenize(code, add_prefix_space=True)

        # Filter by token length (leave room for special tokens)
        if len(tokens) < 10 or len(tokens) > 450:
            return None

        # Extract AST
        ast = extract_ast_for_code(code)

        return {
            'idx': f'cpp::{idx}',
            'code': code,
            'code_tokens': tokens,
            'ast': ast,
            'docstring': '',  # Keep for compatibility
            'docstring_tokens': []
        }
    except Exception:
        return None


def should_keep_code(code: str) -> bool:
    """Filter criteria for C++ code"""
    if len(code) < 100 or len(code) > 10000:
        return False
    lines = code.count('\n')
    if lines < 3 or lines > 500:
        return False
    # Basic C++ indicators
    if 'void ' not in code and 'int ' not in code and 'class ' not in code and 'std::' not in code:
        return False
    return True


# ============================================================================
# Main Processing Function
# ============================================================================

def stream_and_process_dataset(output_file: str, max_samples: int = 10000):
    """
    Stream dataset and save in JSONL format for UniXcoder.
    Each sample includes code, tokens, and AST.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset in streaming mode...")
    print(f"Processing up to {max_samples} samples with AST extraction...\n")
    dataset = load_dataset("codeparrot/github-code-clean", "C++-all", split="train", streaming=True)

    processed_count = 0
    with open(output_path, 'w', encoding='utf-8') as f, tqdm(total=max_samples, desc="Processing C++ files") as pbar:
        for example in dataset:
            if processed_count >= max_samples:
                break

            code = example.get('code')
            if not code or not should_keep_code(code):
                continue

            processed = preprocess_code(code, processed_count)
            if processed:
                f.write(json.dumps(processed, ensure_ascii=False) + '\n')
                processed_count += 1
                pbar.update(1)

    print(f"\n{'=' * 50}")
    print(f"Processing complete!")
    print(f"Total samples processed and saved: {processed_count}")
    print(f"Data saved to: {output_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prepare dataset for UniXcoder with AST')
    parser.add_argument('--output_file', type=str, default='data/unixcoder_cpp.jsonl',
                        help='Output JSONL file for processed data')
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Maximum number of samples to process')
    args = parser.parse_args()
    stream_and_process_dataset(args.output_file, args.max_samples)
