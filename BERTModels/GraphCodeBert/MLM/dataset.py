"""
Data Flow Graph (DFG) Extraction for C++ Code
Streams the-stack dataset and extracts DFG using tree-sitter
Compatible with GraphCodeBERT dataset format
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from tree_sitter import Language, Parser
from datasets import load_dataset
from transformers import RobertaTokenizer
from tqdm import tqdm

import tree_sitter_cpp as tscpp

CPP_LANGUAGE = Language(tscpp.language())
ts_parser = Parser(CPP_LANGUAGE)
print("✓ Tree-sitter initialized")

# Load tokenizer with explicit settings for consistency
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
print("✓ Tokenizer loaded")

"""
Extract data flow graph from C++ code following GraphCodeBERT format
Maps tree-sitter nodes to their sequential token index (0, 1, 2...)
Creates edges from variable definitions to uses
"""
def extract_dataflow_graph(code_bytes: bytes, tree) -> List[Tuple]:
    root_node = tree.root_node
    var_definitions = defaultdict(list)
    var_uses = defaultdict(list)
    tokens = []


    node_to_token_pos = {}

    def extract_tokens_recursive(node):
        if node.type in ['identifier', 'field_identifier']:
            if id(node) not in node_to_token_pos:
                node_to_token_pos[id(node)] = len(tokens)
                tokens.append(node)

        for child in node.children:
            extract_tokens_recursive(child)

    extract_tokens_recursive(root_node)

    def is_definition(node):
        parent = node.parent
        if not parent: return False
        if parent.type in ['declaration', 'init_declarator', 'parameter_declaration']: return True
        if parent.type == 'assignment_expression' and node == parent.child_by_field_name('left'): return True
        return False

    def traverse_for_vars(node):
        if node.type in ['identifier', 'field_identifier']:
            var_name = code_bytes[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
            token_pos = node_to_token_pos.get(id(node), -1)
            if token_pos != -1:
                (var_definitions if is_definition(node) else var_uses)[var_name].append(token_pos)

        for child in node.children:
            traverse_for_vars(child)

    traverse_for_vars(root_node)

    dfg_edges = []
    for var_name, uses in var_uses.items():
        defs = sorted(var_definitions.get(var_name, []))
        for use_pos in uses:
            preceding_defs = [d for d in defs if d < use_pos]
            if preceding_defs:
                def_pos = preceding_defs[-1]
                dfg_edges.append((var_name, use_pos, "comesFrom", [var_name], [def_pos]))
    return dfg_edges

"""
Preprocess C++ code and extract DFG in GraphCodeBERT format
Filters out code snippets that are too short/long or have insufficient DFG
Returns a dictionary with code, tokens, DFG, and metadata
"""
def preprocess_code(code: str, idx: int) -> Dict:
    try:
        code_bytes = code.encode('utf8')
        tree = ts_parser.parse(code_bytes)
        tokens = tokenizer.tokenize(code, add_prefix_space=True)

        if len(tokens) < 10 or len(tokens) > 450: #
            return None

        dfg = extract_dataflow_graph(code_bytes, tree)

        if not dfg or len(dfg) < 2:
            return None

        return {
            'idx': f'cpp::{idx}',
            'code': code,
            'code_tokens': tokens,
            'dataflow_graph': dfg,
            'docstring': '',
            'docstring_tokens': []
        }
    except Exception:
        return None

"""
Basic filtering to determine if code snippet should be processed
Based on length and presence of C++ constructs
"""
def should_keep_code(code: str) -> bool:
    if len(code) < 100 or len(code) > 10000: return False
    lines = code.count('\n')
    if lines < 3 or lines > 500: return False
    if 'void ' not in code and 'int ' not in code and 'class ' not in code and 'std::' not in code: return False
    return True

"""
Stream dataset, extract DFG, and save in JSONL format.
"""
def stream_and_process_dataset(output_file: str, max_samples: int = 10000):

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset in streaming mode...")
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

    print(f"\n{'=' * 50}\nProcessing complete!")
    print(f"Total samples processed and saved: {processed_count}")
    print(f"Data saved to: {output_path}\n{'=' * 50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract DFG from C++ code')
    parser.add_argument('--output_file', type=str, default='data/cpp_functions.jsonl',
                        help='Output JSONL file for processed data')
    parser.add_argument('--max_samples', type=int, default=10000,
                        help='Maximum number of samples to process')
    args = parser.parse_args()
    stream_and_process_dataset(args.output_file, args.max_samples)

