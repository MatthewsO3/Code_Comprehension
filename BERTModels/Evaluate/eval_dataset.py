"""
Evaluation Dataset Generation for Multiple Code Models
Streams C++ code from the-stack dataset once and generates evaluation sets for:
- GraphCodeBERT (with DFG extraction)
- UniXcoder (simplified tokenization)
- CodeBERT-cpp (RoBERTa-based tokenization)

All three models process the SAME examples - a record is only saved if it passes
validation for ALL three models. This ensures identical evaluation sets.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from tree_sitter import Language, Parser
from datasets import load_dataset
from transformers import RobertaTokenizer
from tqdm import tqdm

import tree_sitter_cpp as tscpp

# ============================================================================
# Initialize models and tokenizers
# ============================================================================

CPP_LANGUAGE = Language(tscpp.language())
ts_parser = Parser(CPP_LANGUAGE)
print("✓ Tree-sitter initialized")

repo_dir = Path(__file__).parent.parent.absolute()
print("repo dir: ", repo_dir)
# Navigate up to repo root, then to config
graph_path = repo_dir / 'GraphCodeBert/Models/graphcodebert-cpp-mlm-from-config/best_model'
unix_path = repo_dir / 'UnixCoderCPP/unixcoder-cpp-mlm/best_model'
code_path = 'neulab/codebert-cpp'

# Load tokenizers for each model
graphcodebert_tokenizer = RobertaTokenizer.from_pretrained(graph_path)
print("✓ GraphCodeBERT tokenizer loaded")

unixcoder_tokenizer = RobertaTokenizer.from_pretrained(unix_path)
print("✓ UniXcoder tokenizer loaded")

codebert_tokenizer = RobertaTokenizer.from_pretrained(code_path)
print("✓ CodeBERT-cpp tokenizer loaded")


# ============================================================================
# GraphCodeBERT: DFG Extraction
# ============================================================================

def extract_dataflow_graph(code_bytes: bytes, tree) -> List[Tuple]:
    """
    Extract data flow graph from C++ code following GraphCodeBERT format.
    Maps tree-sitter nodes to their sequential token index (0, 1, 2...)
    Creates edges from variable definitions to uses.
    """
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
        if not parent:
            return False
        if parent.type in ['declaration', 'init_declarator', 'parameter_declaration']:
            return True
        if parent.type == 'assignment_expression' and node == parent.child_by_field_name('left'):
            return True
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


def preprocess_graphcodebert(code: str, idx: int) -> Optional[Dict]:
    """
    Preprocess C++ code for GraphCodeBERT.
    Extracts DFG and validates minimum dataflow complexity.
    """
    try:
        code_bytes = code.encode('utf8')
        tree = ts_parser.parse(code_bytes)
        tokens = graphcodebert_tokenizer.tokenize(code, add_prefix_space=True)

        if len(tokens) < 10 or len(tokens) > 450:
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


# ============================================================================
# UniXcoder: Simple Tokenization (no DFG)
# ============================================================================

def preprocess_unixcoder(code: str, idx: int) -> Optional[Dict]:
    """
    Preprocess C++ code for UniXcoder.
    UniXcoder is simpler than GraphCodeBERT - no DFG needed.
    """
    try:
        tokens = unixcoder_tokenizer.tokenize(code, add_prefix_space=True)

        if len(tokens) < 10 or len(tokens) > 450:
            return None

        return {
            'idx': f'cpp::{idx}',
            'code': code,
            'code_tokens': tokens,
            'docstring': '',
            'docstring_tokens': []
        }
    except Exception:
        return None


# ============================================================================
# CodeBERT-cpp: RoBERTa Tokenization (no DFG)
# ============================================================================

def preprocess_codebert_cpp(code: str, idx: int) -> Optional[Dict]:
    """
    Preprocess C++ code for CodeBERT-cpp.
    CodeBERT-cpp uses standard RoBERTa tokenization without DFG.
    """
    try:
        tokens = codebert_tokenizer.tokenize(code, add_prefix_space=True)

        if len(tokens) < 10 or len(tokens) > 450:
            return None

        return {
            'idx': f'cpp::{idx}',
            'code': code,
            'code_tokens': tokens,
            'docstring': '',
            'docstring_tokens': []
        }
    except Exception:
        return None


# ============================================================================
# Shared Utilities
# ============================================================================

def should_keep_code(code: str) -> bool:
    """Filter criteria for C++ code snippets."""
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
# Main Streaming Function
# ============================================================================

def stream_and_process_dataset(
        output_dir: str,
        skip_records: int = 0,
        num_samples: int = 10000
):
    """
    Stream dataset once and process records for all three models.

    IMPORTANT: A record is ONLY saved if it passes validation for ALL three models.
    This ensures all three JSONL files contain identical examples.

    Args:
        output_dir: Directory to save the three JSONL files
        skip_records: Number of records to skip at the start
        num_samples: Exact number of samples to save in each file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    graphcodebert_file = output_path / 'graphcodebert_evalset.jsonl'
    unixcoder_file = output_path / 'unixcoder_evalset.jsonl'
    codebert_file = output_path / 'codebert_evalset.jsonl'

    print(f"Loading dataset in streaming mode...")
    print(f"Skipping {skip_records} records, processing {num_samples} records")
    print(f"Note: A record is only saved if it passes ALL THREE models' validation\n")
    dataset = load_dataset("codeparrot/github-code-clean", "C++-gpl-2.0", split="train", streaming=True)

    skipped = 0
    processed_count = 0
    total_evaluated = 0
    failed_gcb = 0
    failed_uxc = 0
    failed_cbt = 0

    # Open all three output files
    with open(graphcodebert_file, 'w', encoding='utf-8') as gcb_f, \
            open(unixcoder_file, 'w', encoding='utf-8') as uxc_f, \
            open(codebert_file, 'w', encoding='utf-8') as cbt_f, \
            tqdm(total=num_samples, desc="Processing C++ files (all 3 models)") as pbar:

        for example in dataset:
            # Stop when we've processed enough
            if processed_count >= num_samples:
                break

            # Skip records if needed
            if skipped < skip_records:
                skipped += 1
                continue

            code = example.get('code')
            if not code or not should_keep_code(code):
                continue

            total_evaluated += 1
            idx = skip_records + processed_count

            # Process code for ALL THREE models in single pass
            gcb_result = preprocess_graphcodebert(code, idx)
            uxc_result = preprocess_unixcoder(code, idx)
            cbt_result = preprocess_codebert_cpp(code, idx)

            # Track which models failed
            if not gcb_result:
                failed_gcb += 1
            if not uxc_result:
                failed_uxc += 1
            if not cbt_result:
                failed_cbt += 1

            # Only write if ALL three models succeeded
            if gcb_result and uxc_result and cbt_result:
                gcb_f.write(json.dumps(gcb_result, ensure_ascii=False) + '\n')
                uxc_f.write(json.dumps(uxc_result, ensure_ascii=False) + '\n')
                cbt_f.write(json.dumps(cbt_result, ensure_ascii=False) + '\n')

                processed_count += 1
                pbar.update(1)

            if total_evaluated % 5000 == 0 and total_evaluated > 0:
                print(f"\nEvaluated {total_evaluated} records, accepted {processed_count}/{num_samples}...")

    print(f"\n{'=' * 70}")
    print(f"Processing complete!")
    print(f"\nStatistics:")
    print(f"  Total records evaluated: {total_evaluated}")
    print(f"  Total samples saved: {processed_count}")
    print(f"  Acceptance rate: {100 * processed_count / total_evaluated:.2f}%" if total_evaluated > 0 else "  No records processed")
    print(f"\nFailures by model (records that failed for that model):")
    print(f"  GraphCodeBERT failed: {failed_gcb}")
    print(f"  UniXcoder failed: {failed_uxc}")
    print(f"  CodeBERT-cpp failed: {failed_cbt}")
    print(f"\nOutput files (all with {processed_count} identical records):")
    print(f"  {graphcodebert_file}")
    print(f"  {unixcoder_file}")
    print(f"  {codebert_file}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate evaluation dataset for multiple code models'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/eval',
        help='Output directory for JSONL files'
    )
    parser.add_argument(
        '--skip_records',
        type=int,
        default=0,
        help='Number of records to skip from the start'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10000,
        help='Number of samples to process (each file will have exactly this many records)'
    )

    args = parser.parse_args()
    stream_and_process_dataset(
        output_dir=args.output_dir,
        skip_records=args.skip_records,
        num_samples=args.num_samples
    )