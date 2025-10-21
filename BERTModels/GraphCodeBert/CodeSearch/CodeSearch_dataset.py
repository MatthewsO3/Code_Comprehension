import json
import os
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
from transformers import RobertaTokenizer
from tqdm import tqdm

# Initialize Tree-sitter and Tokenizer
CPP_LANGUAGE = Language(tscpp.language())
ts_parser = Parser(CPP_LANGUAGE)

# Use a public model instead of local path
try:
    tokenizer = RobertaTokenizer.from_pretrained("/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/GraphCodeBert/graphcodebert-cpp-mlm-from-config/best_model")
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

        # Return empty list instead of None if no edges found (still valid)
        return dfg_edges
    except Exception as e:
        # Log the error for debugging
        # print(f"DFG extraction error: {str(e)}")
        return []  # Return empty list instead of None


def process_sample(sample):
    """Process a single sample and extract DFG."""
    try:
        code = sample.get('code', '').strip()
        docstring = sample.get('docstring', '').strip()

        # Check if code and docstring are valid
        if not code or len(code) < 10:  # Minimum code length
            return None

        if not docstring or len(docstring) < 5:  # Minimum docstring length
            return None

        # Extract DFG
        code_bytes = code.encode('utf8')
        tree = ts_parser.parse(code_bytes)

        # Check if parse tree is valid
        if tree.root_node.child_count == 0:
            return None

        dfg = extract_dataflow_graph(code_bytes, tree)

        # Accept samples even with empty DFG (they still have valid code structure)
        return {
            'code': code,
            'docstring': docstring,
            'dfg': dfg if dfg is not None else []
        }
    except Exception as e:
        # print(f"Sample processing error: {str(e)}")
        return None


def preprocess_dataset(output_path, num_samples=None):
    """Stream dataset, extract DFG, and save to JSONL."""
    print(f"Loading streaming dataset from codeparrot/xlcost-text-to-code...")
    try:
        # Try to load with specific config first
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
            # Fallback: load without specific config
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
                # Extract single sample from batch
                # The dataset has 'code' and 'text' columns
                if 'code' not in batch or 'text' not in batch:
                    # Print available keys on first miss
                    if skipped_count == 0:
                        print(f"Available batch keys: {list(batch.keys())}")
                    skipped_count += 1
                    continue

                code = batch['code']
                text = batch['text']

                # Extract first element if it's a list
                if isinstance(code, (list, tuple)):
                    code = code[0] if code else None
                if isinstance(text, (list, tuple)):
                    text = text[0] if text else None

                if code is None or text is None:
                    skipped_count += 1
                    continue

                sample = {
                    'code': code,
                    'docstring': text
                }

                processed_sample = process_sample(sample)

                if processed_sample is not None:
                    # Write to JSONL
                    f_out.write(json.dumps(processed_sample, ensure_ascii=False) + '\n')
                    processed_count += 1
                    pbar.set_postfix({'processed': processed_count, 'skipped': skipped_count})
                else:
                    skipped_count += 1

            except Exception as e:
                skipped_count += 1
                # Uncomment to debug specific errors
                # print(f"Error processing batch {idx}: {str(e)}")
                continue

    print(f"\n✓ Preprocessing complete!")
    print(f"  Processed samples: {processed_count}")
    print(f"  Skipped samples: {skipped_count}")
    print(f"  Output saved to: {output_path}")
    print(f"  Success rate: {100 * processed_count / (processed_count + skipped_count):.2f}%")

    return processed_count, skipped_count


if __name__ == '__main__':
    import json as cfg_json

    # Load config
    config_path = '/Users/czapmate/Desktop/szakdoga/GraphCodeBert_CPP/BERTModels/GraphCodeBert/config.json'
    try:
        with open(config_path) as f:
            config = cfg_json.load(f).get('codesearch')
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        print("Using default settings...")
        config = {
            'processed_data_file': './data/code_docstring_dataset.jsonl',
            'num_samples': None
        }

    # Get output path and num_samples from config
    output_file = config.get('processed_data_file')
    num_samples = config.get('num_samples')

    if not output_file:
        raise ValueError("'processed_data_file' must be specified in config")

    print(f"Config loaded:")
    print(f"  Output file: {output_file}")
    print(f"  Num samples: {num_samples if num_samples else 'all available'}")
    print()

    processed, skipped = preprocess_dataset(output_file, num_samples=num_samples)
    print(f"\nDataset preprocessing finished. Ready to train!")