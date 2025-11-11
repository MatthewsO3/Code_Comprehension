import argparse
import subprocess
import sys
from pathlib import Path


def run_command(script_name, description, extra_args=None):
    """Run a Python script and handle errors."""
    print(f"\n{'=' * 70}")
    print(f"Running: {description}")
    print(f"{'=' * 70}\n")

    try:
        cmd = [sys.executable, script_name]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Error running {description}: {e}")
        return False


def run_mlm_pipeline(max_samples=10):
    """Run the MLM pre-training pipeline with optional sample limit."""
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "MLM PRE-TRAINING PIPELINE" + " " * 22 + "#")
    print("#" * 70)

    if max_samples:
        print(f"# Max samples: {max_samples}")
        print("#" * 70)

    script_dir = Path(__file__).parent.absolute()

    # Step 1: Extract DFG from C++ code
    dataset_script = str(script_dir / "MLM/dataset.py")
    dataset_args = []
    if max_samples:
        dataset_args = ["--max_samples", str(max_samples)]

    if not Path(dataset_script).exists():
        print(f"\n✗ Script not found: {dataset_script}")
        return False

    if not run_command(dataset_script, "Step 1: Extract DFG from C++ code", dataset_args):
        print("\n" + "#" * 70)
        print("# MLM Pipeline FAILED at: Step 1 (Dataset extraction)")
        print("#" * 70 + "\n")
        return False

    # Step 2: Train GraphCodeBERT with MLM + Edge Prediction
    train_script = str(script_dir / "MLM/train_mlm.py")
    if not Path(train_script).exists():
        print(f"\n✗ Script not found: {train_script}")
        return False

    if not run_command(train_script, "Step 2: Train GraphCodeBERT with MLM + Edge Prediction"):
        print("\n" + "#" * 70)
        print("# MLM Pipeline FAILED at: Step 2 (Training)")
        print("#" * 70 + "\n")
        return False

    # Step 3: Evaluate MLM model
    eval_script = str(script_dir / "MLM/mlm_evaluator.py")
    if not Path(eval_script).exists():
        print(f"\n✗ Script not found: {eval_script}")
        return False

    if not run_command(eval_script, "Step 3: Evaluate MLM model"):
        print("\n" + "#" * 70)
        print("# MLM Pipeline FAILED at: Step 3 (Evaluation)")
        print("#" * 70 + "\n")
        return False

    print("\n" + "#" * 70)
    print("# MLM Pipeline COMPLETED successfully!")
    print("#" * 70 + "\n")
    return True


def run_codesearch_pipeline():
    """Run the CodeSearch training pipeline."""
    print("\n" + "#" * 70)
    print("#" + " " * 18 + "CODE SEARCH TRAINING PIPELINE" + " " * 19 + "#")
    print("#" * 70)

    script_dir = Path(__file__).parent.absolute()

    steps = [
        (str(script_dir / "CodeSearch/CodeSearch_dataset.py"), "Step 1: Prepare CodeSearch dataset and distractors"),
        (str(script_dir / "CodeSearch/CodeSearch_train.py"), "Step 2: Train CodeSearch model"),
        (str(script_dir / "CodeSearch/CodeSearch_eval.py"), "Step 3: Evaluate CodeSearch model"),
    ]

    failed_step = None
    for script, description in steps:
        if not Path(script).exists():
            print(f"\n✗ Script not found: {script}")
            failed_step = description
            break

        if not run_command(script, description):
            failed_step = description
            break

    print("\n" + "#" * 70)
    if failed_step:
        print(f"# CodeSearch Pipeline FAILED at: {failed_step}")
    else:
        print("# CodeSearch Pipeline COMPLETED successfully!")
    print("#" * 70 + "\n")

    return failed_step is None


def main():
    parser = argparse.ArgumentParser(
        description="Run complete ML pipelines for code understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mlm                    # Run MLM pre-training pipeline
  python run.py --mlm --max_samples 1000 # Run MLM with 1000 samples
  python run.py --codesearch             # Run CodeSearch training pipeline
        """
    )

    parser.add_argument(
        "--mlm",
        action="store_true",
        help="Run MLM pre-training pipeline (dataset.py → train_mlm.py → mlm_evaluator.py)"
    )

    parser.add_argument(
        "--codesearch",
        action="store_true",
        help="Run CodeSearch training pipeline (CodeSearch_dataset.py → CodeSearch_train.py → CodeSearch_eval.py)"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=50000,
        help="Maximum number of samples to process in MLM dataset extraction (default: None = all)"
    )

    args = parser.parse_args()

    # Check that exactly one flag is provided
    if not args.mlm and not args.codesearch:
        parser.print_help()
        print("\n✗ Error: Please specify either --mlm or --codesearch flag")
        sys.exit(1)

    if args.mlm and args.codesearch:
        print("\n✗ Error: Please specify only one pipeline (--mlm or --codesearch)")
        sys.exit(1)

    # Warn if max_samples is specified but not using MLM
    if args.max_samples and not args.mlm:
        print("\n⚠️  Warning: --max_samples is only used with --mlm flag")

    # Run the appropriate pipeline
    success = False
    if args.mlm:
        success = run_mlm_pipeline(args.max_samples)
    elif args.codesearch:
        success = run_codesearch_pipeline()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()