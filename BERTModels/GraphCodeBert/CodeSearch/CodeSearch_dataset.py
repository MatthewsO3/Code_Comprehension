# ============================================================================
# CodeSearch_dataset.py - OPTIMIZED with pre-computed corpus encoding
# ============================================================================

import json
import os
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import torch
from pathlib import Path
import numpy as np

# Initialize Tree-sitter and Tokenizer
CPP_LANGUAGE = Language(tscpp.language())
ts_parser = Parser(CPP_LANGUAGE)

try:
    token_dir = Path(__file__).parent.absolute()

    # Navigate up to repo root, then to config
    token_path = token_dir.parent.parent / 'GraphCodeBert/graphcodebert-cpp-mlm-from-config/best_model'
    tokenizer = RobertaTokenizer.from_pretrained(token_path)
except:
    print("Warning: Could not load tokenizer. Using default RobertaTokenizer.")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def extract_dataflow_graph(code_bytes, tree):
    """Extract data flow graph from code using tree-sitter."""
    try:
        root_node = tree.root_node
        tokens = []
        node_to_token_pos = {}

        def get_tokens_recursive(node):
            if not node.children:
                tokens.append(node)
                node_to_token_pos[node.id] = len(tokens) - 1
            for child in node.children:
                get_tokens_recursive(child)

        get_tokens_recursive(root_node)

        var_definitions = defaultdict(list)
        var_uses = defaultdict(list)

        def is_definition(node):
            parent = node.parent
            if not parent:
                return False
            if parent.type in ['declaration', 'init_declarator', 'parameter_declaration']:
                return True
            if parent.type == 'assignment_expression' and node.id == parent.child_by_field_name('left').id:
                return True
            return False

        queue = [root_node]
        while queue:
            node = queue.pop(0)
            if node.type in ['identifier', 'field_identifier']:
                var_name = code_bytes[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
                token_pos = node_to_token_pos.get(node.id)
                if token_pos is not None:
                    if is_definition(node):
                        var_definitions[var_name].append(token_pos)
                    else:
                        var_uses[var_name].append(token_pos)
            queue.extend(node.children)

        dfg_edges = []
        for var_name, uses in var_uses.items():
            defs = sorted(var_definitions.get(var_name, []))
            for use_pos in uses:
                preceding_defs = [d for d in defs if d < use_pos]
                if preceding_defs:
                    def_pos = preceding_defs[-1]
                    dfg_edges.append([var_name, use_pos, "comesFrom", [var_name], [def_pos]])

        return dfg_edges
    except Exception as e:
        return []


def process_sample(sample):
    """Process a single sample and extract DFG."""
    try:
        code = sample.get('code', '').strip()
        docstring = sample.get('docstring', '').strip()

        if not code or len(code) < 10:
            return None

        if not docstring or len(docstring) < 5:
            return None

        code_bytes = code.encode('utf8')
        tree = ts_parser.parse(code_bytes)

        if tree.root_node.child_count == 0:
            return None

        dfg = extract_dataflow_graph(code_bytes, tree)

        return {
            'code': code,
            'docstring': docstring,
            'dfg': dfg if dfg is not None else []
        }
    except Exception as e:
        return None


def preprocess_dataset(output_path, num_samples=None):
    """Stream dataset, extract DFG, and save to JSONL."""
    print(f"Loading streaming dataset from codeparrot/xlcost-text-to-code...")
    try:
        ds = load_dataset(
            "codeparrot/xlcost-text-to-code",
            "C++-program-level",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load with config: {str(e)}")
        print("Trying without config...")
        try:
            ds = load_dataset(
                "codeparrot/xlcost-text-to-code",
                split="train",
                streaming=True,
                trust_remote_code=True
            )
        except Exception as e2:
            print(f"Failed to load dataset: {str(e2)}")
            print("\nTroubleshooting:")
            print("1. Check your internet connection")
            print("2. Try: huggingface-cli repo download codeparrot/xlcost-text-to-code --repo-type dataset")
            print("3. Or manually download from: https://huggingface.co/datasets/codeparrot/xlcost-text-to-code")
            raise

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0

    print(f"Processing dataset and saving to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        iterator = ds.iter(batch_size=1)
        pbar = tqdm(iterator, desc="Processing samples")

        for idx, batch in enumerate(pbar):
            if num_samples is not None and processed_count >= num_samples:
                break

            try:
                if 'code' not in batch or 'text' not in batch:
                    if skipped_count == 0:
                        print(f"Available batch keys: {list(batch.keys())}")
                    skipped_count += 1
                    continue

                code = batch['code']
                text = batch['text']

                if isinstance(code, (list, tuple)):
                    code = code[0] if code else None
                if isinstance(text, (list, tuple)):
                    text = text[0] if text else None

                if code is None or text is None:
                    skipped_count += 1
                    continue

                sample = {'code': code, 'docstring': text}
                processed_sample = process_sample(sample)

                if processed_sample is not None:
                    f_out.write(json.dumps(processed_sample, ensure_ascii=False) + '\n')
                    processed_count += 1
                    pbar.set_postfix({'processed': processed_count, 'skipped': skipped_count})
                else:
                    skipped_count += 1

            except Exception as e:
                skipped_count += 1
                continue

    print(f"\n✓ Preprocessing complete!")
    print(f"  Processed samples: {processed_count}")
    print(f"  Skipped samples: {skipped_count}")
    print(f"  Output saved to: {output_path}")
    print(f"  Success rate: {100 * processed_count / (processed_count + skipped_count):.2f}%")

    return processed_count, skipped_count


def encode_corpus(model, tokenizer, collator, input_jsonl, output_jsonl, device, batch_size=32):
    """
    Pre-compute and cache encoded corpus vectors.

    Args:
        model: RoBERTa model for encoding
        tokenizer: RoBERTa tokenizer
        collator: CodeSearchCollator for processing
        input_jsonl: Path to preprocessed dataset
        output_jsonl: Path to save encoded corpus
        device: torch device
        batch_size: Encoding batch size
    """
    print(f"\nEncoding corpus from {input_jsonl}...")

    # Load samples
    samples = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples for encoding")

    model.eval()

    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        with torch.no_grad():
            for i in tqdm(range(0, len(samples), batch_size), desc="Encoding corpus"):
                batch_samples = samples[i:i + batch_size]

                # Process batch with collator
                processed_codes = []
                for sample in batch_samples:
                    code = sample['code']
                    dfg = sample['dfg']
                    code_processed = collator._process_item(
                        code, dfg, code, collator.max_code_len, is_code=True
                    )
                    processed_codes.append(code_processed)

                # Stack batch
                code_ids = torch.stack([c[0] for c in processed_codes]).to(device)
                code_mask = torch.stack([c[1] for c in processed_codes]).to(device)
                code_pos = torch.stack([c[2] for c in processed_codes]).to(device)

                # Encode
                code_vecs = model(
                    input_ids=code_ids,
                    attention_mask=code_mask,
                    position_ids=code_pos
                ).pooler_output

                # Save with embeddings
                for j, sample in enumerate(batch_samples):
                    embedding = code_vecs[j].cpu().numpy().tolist()
                    output_sample = {
                        'code': sample['code'],
                        'docstring': sample['docstring'],
                        'dfg': sample['dfg'],
                        'embedding': embedding
                    }
                    f_out.write(json.dumps(output_sample, ensure_ascii=False) + '\n')

    print(f"✓ Corpus encoded and saved to {output_jsonl}")


if __name__ == '__main__':
    import json as cfg_json


    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()

    # Navigate up to repo root, then to config
    config_path = script_dir.parent.parent / 'GraphCodeBert/config.json'
    try:
        with open(config_path) as f:
            config = cfg_json.load(f).get('codesearch')
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        print("Using default settings...")
        config = {
            'processed_data_file': './data/code_docstring_dataset.jsonl',
            'encoded_corpus_file': './data/encoded_corpus.jsonl',
            'num_samples': None
        }
    output_file = config.get('processed_data_file')
    script_dir = Path(__file__).parent.absolute()

    # Navigate up to repo root, then to config
    output_file = script_dir.parent.parent / output_file


    encoded_corpus_file = config.get('encoded_corpus_file', './data/encoded_corpus.jsonl')
    script_dir = Path(__file__).parent.absolute()

    # Navigate up to repo root, then to config
    encoded_corpus_file = script_dir.parent.parent / encoded_corpus_file
    num_samples = config.get('num_samples')

    if not output_file:
        raise ValueError("'processed_data_file' must be specified in config")

    print(f"Config loaded:")
    print(f"  Output file: {output_file}")
    print(f"  Encoded corpus file: {encoded_corpus_file}")
    print(f"  Num samples: {num_samples if num_samples else 'all available'}")
    print()

    # Step 1: Preprocess dataset
    processed, skipped = preprocess_dataset(output_file, num_samples=num_samples)
    print(f"\nDataset preprocessing finished!")

    # Step 2: Encode corpus
    print("\n" + "=" * 60)
    print("Now encoding corpus for efficient retrieval...")
    print("=" * 60)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    from CodeSearch_train import CodeSearchCollator

    model_path = config['mlm_model_path']
    model_dir = Path(__file__).parent.absolute()

    # Navigate up to repo root, then to config
    mod_path = model_dir.parent.parent / model_path

    tokenizer = RobertaTokenizer.from_pretrained(mod_path)
    model = RobertaModel.from_pretrained(mod_path).to(device)
    collator = CodeSearchCollator(tokenizer, config['max_code_len'], config['max_query_len'])

    encode_corpus(
        model, tokenizer, collator,
        output_file, encoded_corpus_file,
        device, batch_size=config.get('batch_size', 32)
    )

    print(f"\n✓ Complete! Ready for training and evaluation.")