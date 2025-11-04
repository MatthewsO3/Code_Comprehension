import argparse
import subprocess
import sys
from pathlib import Path


def run_command(script_name, description):
    """Run a Python script and handle errors."""
    print(f"\n{'=' * 70}")
    print(f"Running: {description}")
    print(f"{'=' * 70}\n")

    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Error running {description}: {e}")
        return False


def run_mlm_pipeline():
    """Run the MLM pre-training pipeline."""
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "MLM PRE-TRAINING PIPELINE" + " " * 22 + "#")
    print("#" * 70)

    script_dir = Path(__file__).parent.absolute()

    steps = [
        (str(script_dir / "MLM/dataset.py"), "Step 1: Extract DFG from C++ code"),
        (str(script_dir / "MLM/train_mlm.py"), "Step 2: Train GraphCodeBERT with MLM + Edge Prediction"),
        (str(script_dir / "MLM/mlm_evaluator.py"), "Step 3: Evaluate MLM model"),
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
        print(f"# MLM Pipeline FAILED at: {failed_step}")
    else:
        print("# MLM Pipeline COMPLETED successfully!")
    print("#" * 70 + "\n")

    return failed_step is None


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
  python run.py --mlm              # Run MLM pre-training pipeline
  python run.py --codesearch       # Run CodeSearch training pipeline
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

    args = parser.parse_args()

    # Check that exactly one flag is provided
    if not args.mlm and not args.codesearch:
        parser.print_help()
        print("\n✗ Error: Please specify either --mlm or --codesearch flag")
        sys.exit(1)

    if args.mlm and args.codesearch:
        print("\n✗ Error: Please specify only one pipeline (--mlm or --codesearch)")
        sys.exit(1)

    # Run the appropriate pipeline
    success = False
    if args.mlm:
        success = run_mlm_pipeline()
    elif args.codesearch:
        success = run_codesearch_pipeline()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()