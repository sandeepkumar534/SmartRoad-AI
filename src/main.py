"""
Command-line interface for running pothole detection and benchmarking.
"""

import argparse
import sys
from pathlib import Path

from src.model_manager import get_available_models
from src.benchmark import run_single, save_results, print_summary


# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_INPUT = "input/sample.mp4"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_CONF = 0.25
DEFAULT_CSV = "output/benchmark_results.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="RoadVision AI",
        description=(
            "RoadVision AI — Intelligent Pothole Detection & Road Damage Analysis\n"
            "Phase 1: Pretrained Model Benchmark"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8",
        choices=["yolov8"],
        help="Model to run (default: 'yolov8').",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Path to input video (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for annotated videos (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Confidence threshold for detections (default: {DEFAULT_CONF}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cpu', 'cuda', 'cuda:0', etc. (default: auto-detect).",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable custom CMC temporal association tracking (Phase 2).",
    )

    return parser.parse_args()


def _print_banner():
    """Print the application banner."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          ROADVISION AI — Pothole Detection              ║")
    print("║      Phase 1: Pretrained Model Benchmark                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def _print_settings(args: argparse.Namespace):
    """Print the current run settings."""
    print("Run Settings")
    print("─" * 40)
    print(f"  Model(s)   : {args.model}")
    print(f"  Input      : {args.input}")
    print(f"  Output dir : {args.output}")
    print(f"  Confidence : {args.conf}")
    print(f"  Device     : {args.device or 'auto-detect'}")
    print(f"  Tracking   : {'Enabled (Phase 2)' if args.track else 'Disabled'}")
    print("─" * 40)


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    _print_banner()
    _print_settings(args)

    # Validate confidence threshold
    if not 0.0 < args.conf < 1.0:
        print(f"\n✗ Invalid confidence threshold: {args.conf}")
        print(f"  Must be between 0.0 and 1.0 (exclusive).")
        sys.exit(1)

    # Validate input video path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n✗ Input video not found: {input_path.resolve()}")
        print(f"\n  Please place your road video at:")
        print(f"    {input_path.resolve()}")
        print(f"\n  Supported formats: .mp4, .avi, .mov, .mkv")
        sys.exit(1)

    # Determine which models to run
    model_keys = [args.model]

    # Run inference pipeline
    try:
        # Single model run
        stats = run_single(
            model_key=model_keys[0],
            input_path=args.input,
            output_dir=args.output,
            conf_threshold=args.conf,
            device=args.device,
            track=args.track,
        )
        results = [stats]

        # Save CSV results
        csv_path = Path(args.output) / "benchmark_results.csv"
        save_results(results, csv_path)

        # Print comparison summary
        print_summary(results)

    except FileNotFoundError as exc:
        print(f"\n✗ File not found:\n  {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n✗ Runtime error:\n  {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n⚠ Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n✗ Unexpected error: {type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
