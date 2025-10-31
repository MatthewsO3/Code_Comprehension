import json
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from typing import List, Dict
import random

from GraphCodeBert.CodeSearch.dataset_test import out_path


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

            if (i + 1) % 20 == 0:
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


if __name__ == "__main__":
    # Initialize dataset
    script_dir = Path(__file__).parent.parent.absolute()
    ds_path = script_dir / "data/first1000.jsonl"
    dataset = CodeSearchDataset(ds_path)

    out_path = script_dir / "data/training_data.jsonl"
    # Option 1: Create dataset with hard negatives + random negatives (recommended)
    print("\n=== Creating training data with hard negatives ===")
    training_data = dataset.create_training_data(
        out_path,
        hard_negative_ratio=0.6
    )

    # Option 2: Create dataset with batch negatives (alternative)
    # print("\n=== Creating training data with batch negatives ===")
    # training_data = dataset.create_training_data_with_batch_negatives(
    #     'training_data_batch.jsonl',
    #     batch_size=32
    # )

    # Print sample
    print("\n=== Sample record ===")
    sample = training_data[0]
    print(f"Code:\n{sample['code'][:100]}...\n")
    print(f"Good: {sample['good_docstring'][:80]}...")
    print(f"Bad1: {sample['bad1_docstring'][:80]}...")
    print(f"Bad2: {sample['bad2_docstring'][:80]}...")