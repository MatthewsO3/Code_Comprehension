"""
Data Flow Graph (DFG) Extraction for C++ Code
Streams the github-code-clean dataset and extracts DFG using tree-sitter
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
from tree_sitter import Language, Parser
from datasets import load_dataset
from transformers import RobertaTokenizer
from tqdm import tqdm
#
from tree_sitter_cpp import language

CPP_LANGUAGE = Language(language)
parser = Parser()
parser.set_language(CPP_LANGUAGE)
print("Tree-sitter works!")

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")


def get_variable_nodes(node, variables=None):
    """Extract all variable nodes from AST"""
    if variables is None:
        variables = []

    if node.type in ['identifier', 'field_identifier']:
        variables.append((node, node.start_byte, node.end_byte))

    for child in node.children:
        get_variable_nodes(child, variables)

    return variables


def extract_dataflow(code_bytes, tree):
    """
    Extract data flow graph from C++ code
    Returns: List of (variable_name, line_number, edges)
    """
    root_node = tree.root_node

    # Track variable definitions and uses
    definitions = defaultdict(list)  # var_name -> [(line, node)]
    uses = defaultdict(list)  # var_name -> [(line, node)]

    def traverse(node, parent_scope="global"):
        """Traverse AST and track variable def/use"""

        # Variable declarations
        if node.type in ['declaration', 'init_declarator']:
            for child in node.children:
                if child.type == 'identifier':
                    var_name = code_bytes[child.start_byte:child.end_byte].decode('utf8', errors='ignore')
                    definitions[var_name].append((child.start_point[0], child))

        # Assignment expressions
        elif node.type in ['assignment_expression', 'compound_assignment_expression']:
            left = node.child_by_field_name('left')
            if left and left.type == 'identifier':
                var_name = code_bytes[left.start_byte:left.end_byte].decode('utf8', errors='ignore')
                definitions[var_name].append((left.start_point[0], left))

        # Variable uses (reading)
        elif node.type == 'identifier':
            var_name = code_bytes[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
            # Check if this is a use (not a definition)
            if node.parent and node.parent.type not in ['declaration', 'init_declarator']:
                uses[var_name].append((node.start_point[0], node))

        # Recursively process children
        for child in node.children:
            traverse(child, parent_scope)

    traverse(root_node)

    # Build DFG edges
    dfg = []
    for var_name in definitions.keys():
        defs = definitions[var_name]
        var_uses = uses.get(var_name, [])

        for def_line, def_node in defs:
            for use_line, use_node in var_uses:
                if use_line >= def_line:  # Use comes after definition
                    dfg.append((var_name, def_line, use_line))

    return dfg


def preprocess_code(code: str) -> Dict:
    """
    Preprocess C++ code and extract DFG
    Returns: Dictionary with tokens, DFG, and metadata
    """
    try:
        # Parse code
        code_bytes = code.encode('utf8')
        tree = parser.parse(code_bytes)

        # Tokenize
        tokens = tokenizer.tokenize(code)

        # Check token length
        if len(tokens) < 10 or len(tokens) > 512:
            return None

        # Extract DFG
        dfg = extract_dataflow(code_bytes, tree)

        # Convert tokens to IDs
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Create position indices (for each token)
        position_idx = list(range(len(tokens)))

        # Map DFG to token positions
        # Simplified version: map line numbers to approximate token positions
        lines = code.split('\n')
        line_to_token_pos = {}
        current_pos = 0
        for i, line in enumerate(lines):
            line_tokens = tokenizer.tokenize(line)
            line_to_token_pos[i] = current_pos
            current_pos += len(line_tokens)

        # Create DFG edges in token space
        dfg_to_code = []
        dfg_to_dfg = []

        for var_name, def_line, use_line in dfg:
            def_pos = line_to_token_pos.get(def_line, 0)
            use_pos = line_to_token_pos.get(use_line, 0)

            if def_pos < len(tokens) and use_pos < len(tokens):
                dfg_to_code.append((def_pos, use_pos))

        return {
            'input_ids': token_ids,
            'position_idx': position_idx,
            'dfg_to_code': dfg_to_code,
            'dfg_to_dfg': dfg_to_dfg,
            'code': code
        }

    except Exception as e:
        print(f"Error processing code: {e}")
        return None


def should_keep_file(example):
    """Filter criteria for C++ files"""
    code = example['code']
    path = example['path']

    # Check if it's C++
    if not any(path.endswith(ext) for ext in ['.cpp', '.cc', '.cxx', '.h', '.hpp', '.c++']):
        return False

    # Length checks
    if len(code) < 100 or len(code) > 50000:
        return False

    # Line count check
    if code.count('\n') < 10:
        return False

    return True


def stream_and_process_dataset(output_dir: str, max_samples: int = 10):
    """
    Stream dataset, extract DFG, and save processed samples
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading dataset in streaming mode...")
    dataset = load_dataset(
        "codeparrot/github-code-clean",
        split="train",
        streaming=True
    )

    processed_samples = []
    skipped = 0

    print("Processing C++ files...")
    with tqdm(total=max_samples) as pbar:
        for example in dataset:
            # Filter for C++ files
            if not should_keep_file(example):
                continue

            # Process code and extract DFG
            processed = preprocess_code(example['code'])

            if processed is not None:
                processed_samples.append(processed)
                pbar.update(1)

                # Save periodically
                if len(processed_samples) % 1000 == 0:
                    chunk_num = len(processed_samples) // 1000
                    chunk_path = output_path / f"processed_chunk_{chunk_num}.pkl"
                    with open(chunk_path, 'wb') as f:
                        pickle.dump(processed_samples[-1000:], f)
                    print(f"\nSaved chunk {chunk_num} ({len(processed_samples)} total samples)")

                if len(processed_samples) >= max_samples:
                    break
            else:
                skipped += 1

    # Save final chunk
    remaining = len(processed_samples) % 1000
    if remaining > 0:
        chunk_num = len(processed_samples) // 1000 + 1
        chunk_path = output_path / f"processed_chunk_{chunk_num}.pkl"
        with open(chunk_path, 'wb') as f:
            pickle.dump(processed_samples[-remaining:], f)

    # Save metadata
    metadata = {
        'total_samples': len(processed_samples),
        'skipped': skipped,
        'max_samples': max_samples
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Processing complete!")
    print(f"Total samples processed: {len(processed_samples)}")
    print(f"Samples skipped: {skipped}")
    print(f"Data saved to: {output_path}")
    print(f"{'=' * 50}")

    return processed_samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract DFG from C++ code')
    parser.add_argument('--output_dir', type=str, default='./processed_data',
                        help='Output directory for processed data')
    parser.add_argument('--max_samples', type=int, default=100000,
                        help='Maximum number of samples to process')

    args = parser.parse_args()

    # Check if tree-sitter language is built
    if not Path('build/my-languages.so').exists():
        print("Error: Tree-sitter C++ language not built!")
        print("Please run setup script first: bash 1_setup_environment.sh")
        exit(1)

    stream_and_process_dataset(args.output_dir, args.max_samples)