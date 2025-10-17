# codesearch_dataset.py

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Reuse the DFG extraction and preprocessing logic from your original dataset.py
from dataset import preprocess_code, ts_parser, tokenizer


def create_codesearch_jsonl(csv_file: str, output_file: str):
    """
    Reads a CSV with 'docstring' and 'code' columns, processes them for GraphCodeBERT,
    and saves the output to a JSONL file.
    """
    print(f"Reading source CSV file from: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        # ⚠️ IMPORTANT: Update these column names if your CSV has different ones!
        required_columns = {'docstring', 'code'}
        if not required_columns.issubset(df.columns):
            print(f"Error: CSV must contain 'docstring' and 'code' columns.")
            print(f"Found columns: {list(df.columns)}")
            return
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    print(f"Processing {len(df)} rows and saving to {output_file}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            docstring = str(row['docstring'])
            code = str(row['code'])

            if not docstring or not code:
                continue

            # Preprocess the code to get tokens and DFG
            processed_code_obj = preprocess_code(code, idx)

            if processed_code_obj:
                # Tokenize the docstring (query)
                docstring_tokens = tokenizer.tokenize(docstring, add_prefix_space=True)

                # Use the processed data and add the docstring
                processed_code_obj['docstring'] = docstring
                processed_code_obj['docstring_tokens'] = docstring_tokens

                f.write(json.dumps(processed_code_obj, ensure_ascii=False) + '\n')
                processed_count += 1

    print(f"\nProcessing complete. Total samples saved: {processed_count}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create dataset for code search training.')

    # Load config from file
    config = {}
    if Path('config.json').exists():
        with open('config.json', 'r') as f:
            config = json.load(f).get("codesearch", {})

    parser.add_argument('--csv_file', type=str, default=config.get('source_csv_file'),
                        help='Path to the source CSV file.')
    parser.add_argument('--output_file', type=str, default=config.get('processed_data_file'),
                        help='Path to the output JSONL file.')

    args = parser.parse_args()
    if not args.csv_file or not args.output_file:
        parser.error("Both --csv_file and --output_file must be specified.")

    create_codesearch_jsonl(args.csv_file, args.output_file)