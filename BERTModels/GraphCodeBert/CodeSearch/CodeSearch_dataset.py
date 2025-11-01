import json
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from typing import List, Dict
import random
from datasets import load_dataset
import re
from itertools import islice


class CodeSearchDataset:
    def __init__(self, jsonl_path: str, seed: int = 42):
        """
        Initialize dataset with code-docstring pairs.

        Args:
            jsonl_path: Path to the JSONL file with 'code' and 'positive' columns
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Load data
        self.records = self._load_jsonl(jsonl_path)
        self.docstrings = [record['positive'] for record in self.records]

        # Build BM25 index for hard negatives
        self.bm25 = self._build_bm25_index()
        print(f"Loaded {len(self.records)} records")
        print(f"Built BM25 index with {len(self.docstrings)} docstrings")

    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load JSONL file."""
        records = []
        with open(path, 'r') as f:
            for line in f:
                records.append(json.loads(line.strip()))
        return records

    def _build_bm25_index(self) -> BM25Okapi:
        """Build BM25 index from docstrings for hard negative mining."""
        # Tokenize docstrings
        tokenized_docs = [doc.lower().split() for doc in self.docstrings]
        bm25 = BM25Okapi(tokenized_docs)
        return bm25

    def _get_bm25_hard_negative(self, positive_docstring: str, top_k: int = 10) -> str:
        """
        Get a hard negative using BM25 ranking.
        Returns a docstring that's lexically similar but irrelevant.

        Args:
            positive_docstring: The positive docstring to find negatives for
            top_k: Number of top results to sample from

        Returns:
            A hard negative docstring
        """
        # Tokenize the query
        query_tokens = positive_docstring.lower().split()

        # Get top-k similar docstrings
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(-scores)[:top_k]

        # Filter out the positive itself and randomly select from top-k
        candidates = [i for i in top_indices if self.docstrings[i] != positive_docstring]

        if not candidates:
            # Fallback: random docstring if no candidates
            candidates = list(range(len(self.docstrings)))

        selected_idx = random.choice(candidates)
        return self.docstrings[selected_idx]

    def _get_random_negative(self, positive_docstring: str) -> str:
        """Get a random negative docstring."""
        while True:
            idx = random.randint(0, len(self.docstrings) - 1)
            if self.docstrings[idx] != positive_docstring:
                return self.docstrings[idx]

    def create_training_data(self, output_path: str, hard_negative_ratio: float = 0.7):
        """
        Create training dataset with hard negatives.

        Args:
            output_path: Path to save the output JSONL file
            hard_negative_ratio: Fraction of negatives to be hard negatives (BM25-based)
        """
        training_data = []

        for i, record in enumerate(self.records):
            code = record['code']
            positive = record['positive']

            # Hard negative from BM25
            if random.random() < hard_negative_ratio:
                bad1 = self._get_bm25_hard_negative(positive, top_k=20)
            else:
                bad1 = self._get_random_negative(positive)

            # Second negative (mix of hard and random)
            if random.random() < hard_negative_ratio:
                bad2 = self._get_bm25_hard_negative(positive, top_k=20)
            else:
                bad2 = self._get_random_negative(positive)

            # Ensure negatives are different
            while bad2 == bad1 or bad2 == positive:
                bad2 = self._get_random_negative(positive)

            training_record = {
                'code': code,
                'good_docstring': positive,
                'bad1_docstring': bad1,
                'bad2_docstring': bad2
            }
            training_data.append(training_record)

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(self.records)} records")

        # Save to JSONL
        with open(output_path, 'w') as f:
            for record in training_data:
                f.write(json.dumps(record) + '\n')

        print(f"Saved training data to {output_path}")
        return training_data

    def create_training_data_with_batch_negatives(self, output_path: str, batch_size: int = 32):
        """
        Create training dataset using in-batch negatives + BM25 hard negatives.
        Simulates batch-based negative sampling.

        Args:
            output_path: Path to save the output JSONL file
            batch_size: Simulated batch size for in-batch negatives
        """
        training_data = []

        for i, record in enumerate(self.records):
            code = record['code']
            positive = record['positive']

            # Hard negative from BM25
            bad1 = self._get_bm25_hard_negative(positive, top_k=10)

            # Batch negative: sample from nearby records in the dataset
            # This simulates what would happen in actual batching
            if batch_size > 1:
                batch_start = max(0, i - batch_size // 2)
                batch_end = min(len(self.records), i + batch_size // 2)
                batch_indices = [j for j in range(batch_start, batch_end) if j != i]

                if batch_indices:
                    batch_idx = random.choice(batch_indices)
                    bad2 = self.records[batch_idx]['positive']
                else:
                    bad2 = self._get_random_negative(positive)
            else:
                bad2 = self._get_random_negative(positive)

            # Ensure negatives are different
            while bad2 == bad1 or bad2 == positive:
                bad2 = self._get_random_negative(positive)

            training_record = {
                'code': code,
                'good_docstring': positive,
                'bad1_docstring': bad1,
                'bad2_docstring': bad2
            }
            training_data.append(training_record)

            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(self.records)} records")

        # Save to JSONL
        with open(output_path, 'w') as f:
            for record in training_data:
                f.write(json.dumps(record) + '\n')

        print(f"Saved training data with batch negatives to {output_path}")
        return training_data

    @staticmethod
    def extract_items_from_text(text: str):
        """Extract docstring items from raw text."""
        parts = text.split("|", 1)
        after = parts[1] if len(parts) > 1 else parts[0]

        pieces = [p.strip() for p in after.split(";") if p.strip()]

        # drop leading "C ++ Program to implement..." (case-insensitive)
        if pieces:
            pieces.pop(0)

        # truncate at the first occurrence of "Driver code" (case-insensitive),
        # removing it and everything after
        for i, p in enumerate(pieces):
            if re.search(r'(?i)driver\s*code', p):
                pieces = pieces[:i]
                break

        return pieces

    @staticmethod
    def create_dataset_from_source(
        output_path: str,
        num_records: int = 7500,
        start_idx: int = 0,
        dataset_name: str = "codeparrot/xlcost-text-to-code",
        config: str = "C++-program-level"
    ):
        """
        Create a dataset JSONL file from the HuggingFace dataset source.

        Args:
            output_path: Path to save the output JSONL file
            num_records: Number of records to extract
            start_idx: Starting index in the dataset
            dataset_name: Name of the HuggingFace dataset
            config: Configuration of the dataset
        """
        print(f"Loading dataset {dataset_name} with config {config}...")
        ds = load_dataset(
            dataset_name,
            config,
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        with open(output_path, "w", encoding="utf-8") as fout:
            for record in islice(ds, start_idx, start_idx + num_records):
                code_text = record.get("code", "")
                text = record.get("text", "")
                pieces = CodeSearchDataset.extract_items_from_text(text)
                positive = " ; ".join(pieces)
                fout.write(json.dumps({"code": code_text, "positive": positive}, ensure_ascii=False) + "\n")

        print(f"Created dataset with {num_records} records at {output_path}")

    @staticmethod
    def create_distractors(
        output_path: str,
        num_distractors: int = 1147,
        start_idx: int = 8650,
        dataset_name: str = "codeparrot/xlcost-text-to-code",
        config: str = "C++-program-level"
    ):
        """
        Create a distractors JSONL file for evaluation.

        Args:
            output_path: Path to save the output JSONL file
            num_distractors: Number of distractors to create
            start_idx: Starting index in the dataset (should not overlap with training/eval sets)
            dataset_name: Name of the HuggingFace dataset
            config: Configuration of the dataset
        """
        print(f"Loading dataset {dataset_name} with config {config}...")
        ds = load_dataset(
            dataset_name,
            config,
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        print(f"Creating distractor file at: {output_path}")
        with open(output_path, "w", encoding="utf-8") as fout:
            for idx, record in enumerate(islice(ds, start_idx, start_idx + num_distractors)):
                code_text = record.get("code", "")
                distractor_url = f"distractor_{idx}"
                fout.write(json.dumps({"code": code_text, "url": distractor_url}, ensure_ascii=False) + "\n")

                if (idx + 1) % 1000 == 0:
                    print(f"Processed {idx + 1} distractors...")

        print(f"Finished creating {output_path} with {num_distractors} distractors.")


if __name__ == "__main__":
    script_dir = Path(__file__).parent.parent.absolute()
    data_dir = script_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Step 1: Create dataset from source (replaces dataset_test.py)
    print("\n=== Creating dataset from source ===")
    ds_path = data_dir / "eval.jsonl"
    CodeSearchDataset.create_dataset_from_source(
        str(ds_path),
        num_records=1150,
        start_idx=7500
    )

    # Step 2: Initialize dataset for training
    print("\n=== Initializing dataset ===")
    dataset = CodeSearchDataset(str(ds_path))

    # Step 3: Create training data with hard negatives
    print("\n=== Creating training data with hard negatives ===")
    out_path = data_dir / "training_data.jsonl"
    training_data = dataset.create_training_data(
        str(out_path),
        hard_negative_ratio=0.6
    )

    # Step 4: Create distractors (replaces create_distractors.py)
    print("\n=== Creating distractors ===")
    distractors_path = data_dir / "distractors.jsonl"
    CodeSearchDataset.create_distractors(
        str(distractors_path),
        num_distractors=1147,
        start_idx=8650
    )

    # Step 5: Print sample
    print("\n=== Sample training record ===")
    sample = training_data[0]
    print(f"Code:\n{sample['code'][:100]}...\n")
    print(f"Good: {sample['good_docstring'][:80]}...")
    print(f"Bad1: {sample['bad1_docstring'][:80]}...")
    print(f"Bad2: {sample['bad2_docstring'][:80]}...")

    # Optional: Create dataset with batch negatives
    # print("\n=== Creating training data with batch negatives ===")
    # batch_out_path = data_dir / "training_data_batch.jsonl"
    # training_data_batch = dataset.create_training_data_with_batch_negatives(
    #     str(batch_out_path),
    #     batch_size=32
    # )