import argparse
import logging
import os
import json
from pathlib import Path

import torch
import numpy as np
from model import Model
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

try:
    import pytrec_eval

    HAS_PYTREC = True
except ImportError:
    HAS_PYTREC = False
    print("Warning: pytrec_eval not installed. Install with: pip install pytrec-eval-terrier")

logger = logging.getLogger(__name__)


class CodeSearchEvalDataset(Dataset):
    """Dataset for code search evaluation."""

    def __init__(self, tokenizer, args, file_path=None):
        """
        Args:
            tokenizer: Tokenizer for encoding
            args: Arguments containing code_length
            file_path: Path to JSONL file with code
        """
        self.args = args
        self.tokenizer = tokenizer
        self.examples = []
        self.urls = []

        with open(file_path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                js = json.loads(line)
                self.examples.append(js)
                self.urls.append(js.get('url', f'example_{idx}'))

        logger.info(f"Loaded {len(self.examples)} code examples from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        """
        Returns:
            code_ids: Tokenized code
            code_mask: Attention mask for code
            url: URL or identifier for the example
        """
        example = self.examples[item]
        code = example['code']

        # Tokenize
        tokens = self.tokenizer.tokenize(code)[:self.args.code_length - 2]
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        code_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        code_mask = [1] * len(code_ids)

        # Pad
        padding_length = self.args.code_length - len(code_ids)
        code_ids += [self.tokenizer.pad_token_id] * padding_length
        code_mask += [0] * padding_length

        return (
            torch.tensor(code_ids),
            torch.tensor(code_mask),
            self.urls[item]
        )


class DocstringDataset(Dataset):
    """Dataset for docstrings during evaluation."""

    def __init__(self, tokenizer, args, file_path=None):
        """
        Args:
            tokenizer: Tokenizer for encoding
            args: Arguments containing nl_length
            file_path: Path to JSONL file with docstrings
        """
        self.args = args
        self.tokenizer = tokenizer
        self.examples = []
        self.urls = []

        with open(file_path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                js = json.loads(line)
                self.examples.append(js)
                self.urls.append(js.get('url', f'example_{idx}'))

        logger.info(f"Loaded {len(self.examples)} docstrings from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        """
        Returns:
            doc_ids: Tokenized docstring
            doc_mask: Attention mask for docstring
            url: URL or identifier for the example
        """
        example = self.examples[item]

        # Handle both possible docstring keys
        doc_key = 'positive' if 'positive' in example else 'good_docstring'
        doc = example[doc_key]

        # Tokenize
        tokens = self.tokenizer.tokenize(doc)[:self.args.nl_length - 2]
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        doc_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        doc_mask = [1] * len(doc_ids)

        # Pad
        padding_length = self.args.nl_length - len(doc_ids)
        doc_ids += [self.tokenizer.pad_token_id] * padding_length
        doc_mask += [0] * padding_length

        return (
            torch.tensor(doc_ids),
            torch.tensor(doc_mask),
            self.urls[item]
        )


def collate_fn_code(batch):
    """Collate function for code batch."""
    code_ids = torch.stack([x[0] for x in batch])
    code_mask = torch.stack([x[1] for x in batch])
    urls = [x[2] for x in batch]
    return (code_ids, code_mask, urls)


def collate_fn_docstring(batch):
    """Collate function for docstring batch."""
    doc_ids = torch.stack([x[0] for x in batch])
    doc_mask = torch.stack([x[1] for x in batch])
    urls = [x[2] for x in batch]
    return (doc_ids, doc_mask, urls)


def compute_metrics_pytrec(scores, query_urls, db_urls):
    """
    Compute evaluation metrics using pytrec_eval.

    Args:
        scores: Similarity scores [num_queries, num_docs]
        query_urls: URLs of queries (docstrings)
        db_urls: URLs of database items (code snippets)

    Returns:
        Dictionary with metrics
    """
    if not HAS_PYTREC:
        logger.warning("pytrec_eval not available, using simple metrics instead")
        return compute_metrics_simple(scores, query_urls, db_urls)

    # Create qrel (ground truth): query_url matches code with same URL
    qrel = {}
    db_url_to_idx = {url: i for i, url in enumerate(db_urls)}

    for query_url in query_urls:
        relevance_scores = {}
        # The matching doc is the one with the same URL
        if query_url in db_url_to_idx:
            for db_url in db_urls:
                relevance_scores[db_url] = 1 if db_url == query_url else 0
        qrel[query_url] = relevance_scores

    # Create run (model results)
    run = {}
    for i, query_url in enumerate(query_urls):
        doc_scores = {}
        for j, db_url in enumerate(db_urls):
            doc_scores[db_url] = float(scores[i, j])
        run[query_url] = doc_scores

    # Evaluate
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel,
        {'map', 'recip_rank', 'recall.1', 'recall.5', 'recall.10', 'recall.100', 'recall.1000'}
    )
    results = evaluator.evaluate(run)

    # Aggregate results
    metrics = {
        'mrr': 0.0,
        'map': 0.0,
        'recall@1': 0.0,
        'recall@5': 0.0,
        'recall@10': 0.0,
        'recall@100': 0.0,
        'recall@1000': 0.0,
    }

    num_queries = len(results)
    if num_queries == 0:
        return metrics

    for query_id in results:
        metrics['mrr'] += results[query_id]['recip_rank']
        metrics['map'] += results[query_id]['map']
        metrics['recall@1'] += results[query_id]['recall_1']
        metrics['recall@5'] += results[query_id]['recall_5']
        metrics['recall@10'] += results[query_id]['recall_10']
        metrics['recall@100'] += results[query_id]['recall_100']
        metrics['recall@1000'] += results[query_id]['recall_1000']

    for key in metrics:
        metrics[key] /= num_queries

    return metrics


def compute_metrics_simple(scores, query_urls, db_urls):
    """
    Compute evaluation metrics without pytrec_eval (fallback).

    Args:
        scores: Similarity scores [num_queries, num_docs]
        query_urls: URLs of queries
        db_urls: URLs of database items

    Returns:
        Dictionary with metrics
    """
    sort_ids = np.argsort(scores, axis=-1)[:, ::-1]

    mrr_list = []
    recall_at_k = {1: 0, 5: 0, 10: 0, 100: 0, 1000: 0}

    for i, query_url in enumerate(query_urls):
        rank = None
        for k, db_idx in enumerate(sort_ids[i]):
            if db_urls[db_idx] == query_url:
                rank = k + 1
                break

        if rank is not None:
            mrr_list.append(1.0 / rank)
            for top_k in recall_at_k.keys():
                if rank <= top_k:
                    recall_at_k[top_k] += 1
        else:
            mrr_list.append(0)

    num_queries = len(query_urls)
    metrics = {
        'mrr': float(np.mean(mrr_list)),
        'map': float(np.mean(mrr_list)),  # Approximate as MRR
        'recall@1': float(recall_at_k[1] / num_queries),
        'recall@5': float(recall_at_k[5] / num_queries),
        'recall@10': float(recall_at_k[10] / num_queries),
        'recall@100': float(recall_at_k[100] / num_queries),
        'recall@1000': float(recall_at_k[1000] / num_queries),
    }

    return metrics


def evaluate(args, model, tokenizer):
    """Evaluate the model on code search task with distractors."""

    logger.info("Loading datasets...")

    # Load query dataset (docstrings)
    query_dataset = DocstringDataset(tokenizer, args, args.eval_data_file)

    # Load ground truth code dataset
    gt_code_dataset = CodeSearchEvalDataset(tokenizer, args, args.eval_data_file)

    # Load distractor dataset
    if not os.path.exists(args.distractor_data_file):
        logger.warning(f"Distractor file not found: {args.distractor_data_file}")
        logger.warning("Evaluation will only use ground truth code (no distractors)")
        distractor_dataset = None
    else:
        distractor_dataset = CodeSearchEvalDataset(tokenizer, args, args.distractor_data_file)

    num_workers = 0 if str(args.device) == "mps" else 4

    query_dataloader = DataLoader(
        query_dataset,
        sampler=SequentialSampler(query_dataset),
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn_docstring,
        num_workers=num_workers
    )

    gt_code_dataloader = DataLoader(
        gt_code_dataset,
        sampler=SequentialSampler(gt_code_dataset),
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn_code,
        num_workers=num_workers
    )

    if distractor_dataset:
        distractor_dataloader = DataLoader(
            distractor_dataset,
            sampler=SequentialSampler(distractor_dataset),
            batch_size=args.eval_batch_size,
            collate_fn=collate_fn_code,
            num_workers=num_workers
        )

    # Multi-GPU evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    logger.info(f"Encoding {len(query_dataset)} docstrings (queries)...")
    doc_vecs = []
    doc_urls_list = []

    with torch.no_grad():
        for batch in tqdm(query_dataloader, desc="Docstring embeddings"):
            doc_ids = batch[0].to(args.device)
            doc_mask = batch[1].to(args.device)
            urls = batch[2]

            doc_vec = model(nl_inputs=doc_ids, attention_mask=doc_mask)
            doc_vecs.append(doc_vec.cpu().numpy())
            doc_urls_list.extend(urls)

    doc_vecs = np.concatenate(doc_vecs, axis=0)

    logger.info(f"Encoding {len(gt_code_dataset)} ground truth code snippets...")
    code_vecs_gt = []
    code_urls_gt = []

    with torch.no_grad():
        for batch in tqdm(gt_code_dataloader, desc="Ground truth code"):
            code_ids = batch[0].to(args.device)
            code_mask = batch[1].to(args.device)
            urls = batch[2]

            code_vec = model(code_inputs=code_ids, attention_mask=code_mask)
            code_vecs_gt.append(code_vec.cpu().numpy())
            code_urls_gt.extend(urls)

    code_vecs_list = [np.concatenate(code_vecs_gt, axis=0)]
    code_urls_all = code_urls_gt.copy()

    # Add distractors if available
    if distractor_dataset:
        logger.info(f"Encoding {len(distractor_dataset)} distractor code snippets...")
        code_vecs_distractor = []
        code_urls_distractor = []

        with torch.no_grad():
            for batch in tqdm(distractor_dataloader, desc="Distractor code"):
                code_ids = batch[0].to(args.device)
                code_mask = batch[1].to(args.device)
                urls = batch[2]

                code_vec = model(code_inputs=code_ids, attention_mask=code_mask)
                code_vecs_distractor.append(code_vec.cpu().numpy())
                code_urls_distractor.extend(urls)

        code_vecs_list.append(np.concatenate(code_vecs_distractor, axis=0))
        code_urls_all.extend(code_urls_distractor)

    # Combine all code vectors
    code_vecs = np.concatenate(code_vecs_list, axis=0)

    logger.info(f"  Total queries: {len(doc_urls_list)}")
    logger.info(f"  Total database size: {len(code_urls_all)}")
    logger.info(f"  (GT code: {len(code_urls_gt)}, Distractors: {len(code_urls_all) - len(code_urls_gt)})")

    # Compute similarity scores
    logger.info("Computing similarity scores...")
    scores = np.matmul(doc_vecs, code_vecs.T)

    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_metrics_pytrec(scores, doc_urls_list, code_urls_all)

    model.train()

    return metrics


def set_seed(seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate Code Search Model')

    script_dir = Path(__file__).parent.parent.absolute()
    config_path = script_dir / 'config.json'
    parser.add_argument("--config", default=config_path, type=str,
                        help="Path to config JSON file")

    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="Evaluation data file")
    parser.add_argument("--distractor_data_file", default=None, type=str,
                        help="Distractor code snippets file (optional)")
    parser.add_argument("--model_path", default=None, type=str,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_file", default=None, type=str,
                        help="Output file for evaluation results")
    parser.add_argument("--eval_batch_size", default=None, type=int,
                        help="Evaluation batch size")
    parser.add_argument("--code_length", default=None, type=int,
                        help="Code length")
    parser.add_argument("--nl_length", default=None, type=int,
                        help="Docstring length")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")

    cli_args = parser.parse_args()

    config = load_config(cli_args.config)
    codesearch_config = config.get('codesearch', {})
    eval_config = codesearch_config.get('evaluation', {})

    class Args:
        pass

    args = Args()

    script_dir = Path(__file__).parent.parent.absolute()

    model_path = script_dir / (cli_args.model_path or codesearch_config.get('output_dir', '') + '/best_model')
    eval_data_path = script_dir / (cli_args.eval_data_file or eval_config.get('eval_data_file'))
    distractor_path = script_dir / (
                cli_args.distractor_data_file or eval_config.get('distractor_data_file', 'data/distractors.jsonl'))
    output_file = cli_args.output_file or eval_config.get('output_file', 'eval_results.json')

    args.model_path = model_path
    args.eval_data_file = eval_data_path
    args.distractor_data_file = distractor_path
    args.output_file = output_file
    args.eval_batch_size = cli_args.eval_batch_size or codesearch_config.get('eval_batch_size', 32)
    args.code_length = cli_args.code_length or codesearch_config.get('code_length', 256)
    args.nl_length = cli_args.nl_length or codesearch_config.get('nl_length', 128)
    args.seed = cli_args.seed or codesearch_config.get('seed', 42)

    # Validate paths
    if not args.eval_data_file or not os.path.exists(args.eval_data_file):
        raise ValueError(f"Evaluation data file not found: {args.eval_data_file}")

    if not args.model_path or not os.path.exists(args.model_path):
        raise ValueError(f"Model path not found: {args.model_path}")

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info(f"Device: {device}, n_gpu: {args.n_gpu}")

    set_seed(args.seed)

    logger.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    logger.info(f"Loading model from {args.model_path}")
    base_model = AutoModel.from_pretrained(args.model_path)
    model = Model(base_model)
    model.to(args.device)

    logger.info(f"Model loaded successfully. Hidden size: {model.encoder.config.hidden_size}")

    print("\n" + "=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print(f"  Model path: {args.model_path}")
    print(f"  Eval data: {args.eval_data_file}")
    print(f"  Distractor data: {args.distractor_data_file}")
    print(f"  Code length: {args.code_length}")
    print(f"  NL length: {args.nl_length}")
    print(f"  Batch size: {args.eval_batch_size}")
    print("=" * 60 + "\n")

    logger.info("***** Running evaluation *****")
    metrics = evaluate(args, model, tokenizer)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 60 + "\n")

    results_dir = os.path.dirname(args.output_file)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()