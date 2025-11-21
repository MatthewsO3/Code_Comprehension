import argparse
import logging
import os
import random
from pathlib import Path
import csv  # Importálva a CSV mentéshez

import torch
import json
import numpy as np
from model import Model
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, random_split
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.optim import AdamW

logger = logging.getLogger(__name__)


class Args:
    """Configuration arguments class that can be pickled for multiprocessing."""
    pass


class PerformanceTracker:
    """Tracks and saves all performance metrics during training (Train + Validation)."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'epoch': [],
            # Training metrics
            'train_total_loss': [],
            'train_ce_loss': [],
            'train_neg_loss': [],
            'train_batch_losses': [],
            'train_ce_batch_losses': [],
            'train_neg_batch_losses': [],
            # Validation metrics (ÚJ)
            'val_total_loss': [],
            'val_ce_loss': [],
            'val_neg_loss': [],
            # Misc
            'learning_rate': [],
            'best_loss': None,
            'best_epoch': None,
        }

    def log_batch(self, total_loss, ce_loss, neg_loss):
        """Log individual batch metrics."""
        self.history['train_batch_losses'].append(total_loss)
        self.history['train_ce_batch_losses'].append(ce_loss)
        self.history['train_neg_batch_losses'].append(neg_loss)

    def log_epoch(self, epoch: int, train_metrics, val_metrics, lr=None):
        """Log epoch-level metrics for both train and validation."""
        self.history['epoch'].append(epoch)

        # Train log
        self.history['train_total_loss'].append(train_metrics[0])
        self.history['train_ce_loss'].append(train_metrics[1])
        self.history['train_neg_loss'].append(train_metrics[2])

        # Val log (ÚJ)
        self.history['val_total_loss'].append(val_metrics[0])
        self.history['val_ce_loss'].append(val_metrics[1])
        self.history['val_neg_loss'].append(val_metrics[2])

        if lr is not None:
            self.history['learning_rate'].append(lr)

    def update_best(self, loss, epoch):
        """Update best loss (based on validation)."""
        if self.history['best_loss'] is None or loss < self.history['best_loss']:
            self.history['best_loss'] = loss
            self.history['best_epoch'] = epoch
            return True
        return False

    def save(self):
        """Save all metrics to JSON files."""
        # Save detailed history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Saved training history to {history_path}")

        # Save summary statistics
        summary = self._compute_summary()
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved training summary to {summary_path}")

        # Save CSV for easy plotting
        self._save_csv()

    def _compute_summary(self) -> dict:
        """Compute summary statistics."""
        return {
            'total_epochs': len(self.history['epoch']),
            'best_epoch': self.history['best_epoch'],
            'best_val_loss': self.history['best_loss'],  # Renamed for clarity

            'final_train_loss': self.history['train_total_loss'][-1] if self.history['train_total_loss'] else None,
            'final_val_loss': self.history['val_total_loss'][-1] if self.history['val_total_loss'] else None,

            'min_train_loss': min(self.history['train_total_loss']) if self.history['train_total_loss'] else None,
            'min_val_loss': min(self.history['val_total_loss']) if self.history['val_total_loss'] else None,

            'avg_batch_loss': np.mean(self.history['train_batch_losses']) if self.history[
                'train_batch_losses'] else None,
        }

    def _save_csv(self):
        """Save epoch-level metrics as CSV."""
        try:
            csv_path = self.output_dir / 'training_metrics.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Frissített fejléc validációs adatokkal
                writer.writerow([
                    'Epoch',
                    'Train Total Loss', 'Train CE Loss', 'Train Neg Loss',
                    'Val Total Loss', 'Val CE Loss', 'Val Neg Loss',
                    'Learning Rate'
                ])
                for i in range(len(self.history['epoch'])):
                    writer.writerow([
                        self.history['epoch'][i],
                        self.history['train_total_loss'][i],
                        self.history['train_ce_loss'][i],
                        self.history['train_neg_loss'][i],
                        self.history['val_total_loss'][i],
                        self.history['val_ce_loss'][i],
                        self.history['val_neg_loss'][i],
                        self.history['learning_rate'][i] if i < len(self.history['learning_rate']) else '',
                    ])
            print(f"✓ Saved metrics CSV to {csv_path}")
        except Exception as e:
            print(f"⚠️ Could not save CSV: {e}")


class CodeSearchDataset(Dataset):
    """Dataset for code search with triplet loss using hard negatives."""

    def __init__(self, tokenizer, args, file_path=None):
        self.args = args
        self.tokenizer = tokenizer
        self.examples = []

        with open(file_path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                js = json.loads(line)
                self.examples.append(js)

        logger.info(f"Loaded {len(self.examples)} examples from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]

        def encode(text, max_len):
            encoded = self.tokenizer(
                text,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

        code_ids, code_mask = encode(example['code'], self.args.code_length)
        good_ids, good_mask = encode(example['good_docstring'], self.args.nl_length)
        bad1_ids, bad1_mask = encode(example['bad1_docstring'], self.args.nl_length)
        bad2_ids, bad2_mask = encode(example['bad2_docstring'], self.args.nl_length)

        return (
            code_ids, code_mask,
            good_ids, good_mask,
            bad1_ids, bad1_mask,
            bad2_ids, bad2_mask,
        )


def collate_fn(batch):
    """Custom collate function for batching."""
    code_ids = torch.stack([x[0] for x in batch])
    code_mask = torch.stack([x[1] for x in batch])
    good_ids = torch.stack([x[2] for x in batch])
    good_mask = torch.stack([x[3] for x in batch])
    bad1_ids = torch.stack([x[4] for x in batch])
    bad1_mask = torch.stack([x[5] for x in batch])
    bad2_ids = torch.stack([x[6] for x in batch])
    bad2_mask = torch.stack([x[7] for x in batch])

    return (code_ids, code_mask, good_ids, good_mask, bad1_ids, bad1_mask, bad2_ids, bad2_mask)


def evaluate(model, eval_dataloader, device, args):
    """
    Evaluate the model on the validation set.
    No gradient calculation, strictly for metrics.
    """
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_neg_loss = 0.0
    num_batches = 0

    # Nem számolunk gradienst validáció alatt -> gyorsabb és kevesebb memória
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Validating", unit="batch"):
            code_ids = batch[0].to(device)
            code_mask = batch[1].to(device)
            good_ids = batch[2].to(device)
            good_mask = batch[3].to(device)
            bad1_ids = batch[4].to(device)
            bad1_mask = batch[5].to(device)
            bad2_ids = batch[6].to(device)
            bad2_mask = batch[7].to(device)

            code_vec = model(code_inputs=code_ids, attention_mask=code_mask)
            good_vec = model(nl_inputs=good_ids, attention_mask=good_mask)
            bad1_vec = model(nl_inputs=bad1_ids, attention_mask=bad1_mask)
            bad2_vec = model(nl_inputs=bad2_ids, attention_mask=bad2_mask)

            scores = torch.einsum("ab,cb->ac", good_vec, code_vec)
            loss_fct = CrossEntropyLoss()
            ce_loss = loss_fct(scores, torch.arange(code_ids.size(0), device=scores.device))

            bad_scores_1 = torch.einsum("ab,cb->ac", bad1_vec, code_vec)
            bad_scores_2 = torch.einsum("ab,cb->ac", bad2_vec, code_vec)

            neg_loss = 0
            margin = args.margin
            batch_size = code_ids.size(0)

            for i in range(batch_size):
                pos_score = scores[i, i]
                for j in range(batch_size):
                    neg_loss += torch.clamp(margin + bad_scores_1[i, j] - pos_score, min=0)
                    neg_loss += torch.clamp(margin + bad_scores_2[i, j] - pos_score, min=0)

            neg_loss = neg_loss / (batch_size ** 2)
            total_loss_batch = ce_loss + args.neg_weight * neg_loss

            total_loss += total_loss_batch.item()
            total_ce_loss += ce_loss.item()
            total_neg_loss += neg_loss.item()
            num_batches += 1

    return total_loss / num_batches, total_ce_loss / num_batches, total_neg_loss / num_batches


def train_epoch(model, train_dataloader, optimizer, scheduler, device, args, tracker: PerformanceTracker):
    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_neg_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_dataloader, desc="Training", unit="batch")

    for step, batch in enumerate(progress_bar):
        code_ids = batch[0].to(args.device)
        code_mask = batch[1].to(args.device)
        good_ids = batch[2].to(args.device)
        good_mask = batch[3].to(args.device)
        bad1_ids = batch[4].to(args.device)
        bad1_mask = batch[5].to(args.device)
        bad2_ids = batch[6].to(args.device)
        bad2_mask = batch[7].to(args.device)

        optimizer.zero_grad()

        code_vec = model(code_inputs=code_ids, attention_mask=code_mask)
        good_vec = model(nl_inputs=good_ids, attention_mask=good_mask)
        bad1_vec = model(nl_inputs=bad1_ids, attention_mask=bad1_mask)
        bad2_vec = model(nl_inputs=bad2_ids, attention_mask=bad2_mask)

        scores = torch.einsum("ab,cb->ac", good_vec, code_vec)
        loss_fct = CrossEntropyLoss()
        ce_loss = loss_fct(scores, torch.arange(code_ids.size(0), device=scores.device))

        bad_scores_1 = torch.einsum("ab,cb->ac", bad1_vec, code_vec)
        bad_scores_2 = torch.einsum("ab,cb->ac", bad2_vec, code_vec)

        neg_loss = 0
        margin = args.margin
        batch_size = code_ids.size(0)

        for i in range(batch_size):
            pos_score = scores[i, i]
            for j in range(batch_size):
                neg_loss += torch.clamp(margin + bad_scores_1[i, j] - pos_score, min=0)
                neg_loss += torch.clamp(margin + bad_scores_2[i, j] - pos_score, min=0)

        neg_loss = neg_loss / (batch_size ** 2)
        total_loss_batch = ce_loss + args.neg_weight * neg_loss

        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += total_loss_batch.item()
        total_ce_loss += ce_loss.item()
        total_neg_loss += neg_loss.item()
        num_batches += 1

        tracker.log_batch(total_loss_batch.item(), ce_loss.item(), neg_loss.item())

        progress_bar.set_postfix({
            'Loss': f'{total_loss / num_batches:.4f}',
            'CE': f'{total_ce_loss / num_batches:.4f}',
            'Neg': f'{total_neg_loss / num_batches:.4f}'
        })

    return total_loss / num_batches, total_ce_loss / num_batches, total_neg_loss / num_batches


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    parser = argparse.ArgumentParser()

    script_dir = Path(__file__).parent.parent.absolute()
    config_path = script_dir / 'config.json'
    parser.add_argument("--config", default=config_path, type=str, help="Path to config JSON file")

    # CLI argument overrides
    parser.add_argument("--train_data_file", default=None, type=str, help="Override config: training data file")
    parser.add_argument("--output_dir", default=None, type=str, help="Override config: output directory")
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Override config: model path")
    parser.add_argument("--tokenizer_name", default=None, type=str, help="Override config: tokenizer name")
    parser.add_argument("--config_name", default=None, type=str, help="Override config: config name")
    parser.add_argument("--nl_length", default=None, type=int, help="Override config: docstring length")
    parser.add_argument("--code_length", default=None, type=int, help="Override config: code length")
    parser.add_argument("--train_batch_size", default=None, type=int, help="Override config: training batch size")
    parser.add_argument("--learning_rate", default=None, type=float, help="Override config: learning rate")
    parser.add_argument("--max_grad_norm", default=None, type=float, help="Override config: max gradient norm")
    parser.add_argument("--num_train_epochs", default=None, type=int, help="Override config: number of epochs")
    parser.add_argument("--margin", default=None, type=float, help="Override config: margin for negative loss")
    parser.add_argument("--neg_weight", default=None, type=float, help="Override config: weight for negative loss")
    parser.add_argument('--seed', type=int, default=None, help="Override config: random seed")

    cli_args = parser.parse_args()

    config = load_config(cli_args.config)
    codesearch_config = config.get('codesearch', {})

    args = Args()
    script_dir = Path(__file__).parent.parent.absolute()
    model_path = script_dir / (cli_args.model_name_or_path or codesearch_config.get('model_name_or_path'))
    output_path = script_dir / (cli_args.output_dir or codesearch_config.get('output_dir'))
    train_data_path = script_dir / (cli_args.train_data_file or codesearch_config.get('train_data_file'))

    args.train_data_file = train_data_path
    args.output_dir = output_path
    args.model_name_or_path = model_path
    args.tokenizer_name = model_path
    args.config_name = model_path
    args.nl_length = cli_args.nl_length or codesearch_config.get('nl_length', 128)
    args.code_length = cli_args.code_length or codesearch_config.get('code_length', 256)
    args.train_batch_size = cli_args.train_batch_size or codesearch_config.get('train_batch_size', 8)
    args.learning_rate = cli_args.learning_rate or codesearch_config.get('learning_rate', 5e-5)
    args.max_grad_norm = cli_args.max_grad_norm or codesearch_config.get('max_grad_norm', 1.0)
    args.num_train_epochs = cli_args.num_train_epochs or codesearch_config.get('num_train_epochs', 3)
    args.margin = cli_args.margin or codesearch_config.get('margin', 0.5)
    args.neg_weight = cli_args.neg_weight or codesearch_config.get('neg_weight', 0.5)
    args.seed = cli_args.seed or codesearch_config.get('seed', 42)
    args.early_stopping_patience = codesearch_config.get('early_stopping_patience', 3)

    if not args.train_data_file or not args.output_dir or not args.model_name_or_path:
        raise ValueError("Missing required paths in config or CLI.")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tracker = PerformanceTracker(str(args.output_dir))

    logger.info(f"Loading tokenizer and model from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path)
    base_model = AutoModel.from_pretrained(args.model_name_or_path)
    model = Model(base_model)
    model.to(args.device)

    # --- DATA LOADING AND SPLITTING (ÚJ RÉSZ) ---
    logger.info("Loading full dataset...")
    full_dataset = CodeSearchDataset(tokenizer, args, args.train_data_file)

    # 90% Train, 10% Validation split
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info(f"Data Split: Train={len(train_dataset)}, Val={len(val_dataset)}")

    num_workers = 0 if str(args.device) == "mps" else 4

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    # Validation DataLoader (Sequential sampler is fine for eval)
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * args.num_train_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {total_steps}")

    patience_counter = 0

    for epoch in range(args.num_train_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        print(f"{'=' * 60}")

        # 1. Train
        train_loss, train_ce, train_neg = train_epoch(
            model, train_dataloader, optimizer, scheduler, device, args, tracker
        )

        # 2. Evaluate (Validáció)
        val_loss, val_ce, val_neg = evaluate(
            model, val_dataloader, device, args
        )

        current_lr = optimizer.param_groups[0]['lr']

        # 3. Log Metrics (Train + Val)
        tracker.log_epoch(
            epoch,
            train_metrics=(train_loss, train_ce, train_neg),
            val_metrics=(val_loss, val_ce, val_neg),
            lr=current_lr
        )

        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, Neg: {train_neg:.4f})")
        print(f"  Val Loss:   {val_loss:.4f}   (CE: {val_ce:.4f},   Neg: {val_neg:.4f})")
        print(f"  LR: {current_lr:.2e}")

        # 4. Save Best Model (Based on Validation Loss now!)
        if tracker.update_best(val_loss, epoch):
            patience_counter = 0
            checkpoint_path = Path(args.output_dir) / "best_model"
            print(f"\nNew best model found (Val Loss: {val_loss:.4f})! Saving to {checkpoint_path}")

            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.encoder.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
        else:
            patience_counter += 1
            print(f"\nNo improvement in Val Loss. Patience: {patience_counter}/{args.early_stopping_patience}")
            if patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    print(f"\n{'=' * 60}")
    print(f"Training completed!")
    print(f"Best Val loss: {tracker.history['best_loss']:.4f} at epoch {tracker.history['best_epoch']}")
    print(f"{'=' * 60}\n")

    print("\n" + "=" * 60)
    print("SAVING PERFORMANCE METRICS...")
    print("=" * 60)
    tracker.save()


if __name__ == "__main__":
    main()