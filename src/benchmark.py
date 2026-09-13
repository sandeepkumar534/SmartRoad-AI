"""
Runs identical inference on both YOLO models and compares results.
"""

from pathlib import Path

import pandas as pd

from src.model_manager import download_model, get_model_info
from src.detector import process_video
from src.video_processor import process_video_with_tracking


# ─── Default output file mapping ────────────────────────────────────────────
OUTPUT_FILENAMES = {
    "yolov8": "yolov8_result.mp4",
}


def run_single(
    model_key: str,
    input_path: str | Path,
    output_dir: str | Path,
    conf_threshold: float = 0.25,
    device: str | None = None,
    track: bool = False,
) -> dict:
    """Run benchmark for a single model.

    Args:
        model_key: Model identifier ('yolov8' or 'yolo26').
        input_path: Path to input video.
        output_dir: Directory for output video.
        conf_threshold: Detection confidence threshold.
        device: Inference device (None for auto-detect).

    Returns:
        Dictionary of benchmark statistics.
    """
    info = get_model_info(model_key)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = OUTPUT_FILENAMES.get(model_key, f"{model_key}_result.mp4")
    output_path = output_dir / output_filename

    print()
    print("=" * 60)
    print(f"  BENCHMARKING: {info['description']}")
    print("=" * 60)
    print()

    # Download / verify model weights
    model_path = download_model(model_key)

    # Run inference
    if track:
        stats = process_video_with_tracking(
            model_path=model_path,
            input_path=input_path,
            output_path=output_path,
            conf_threshold=conf_threshold,
            device=device,
        )
    else:
        stats = process_video(
            model_path=model_path,
            input_path=input_path,
            output_path=output_path,
            conf_threshold=conf_threshold,
            device=device,
        )

    # Add model identifier to stats
    stats["model"] = model_key

    return stats



def save_results(results: list[dict], csv_path: str | Path) -> Path:
    """Save benchmark results to CSV.

    Args:
        results: List of stats dictionaries from benchmark runs.
        csv_path: Output CSV file path.

    Returns:
        Path to the saved CSV file.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Select and order the columns for the CSV
    columns = [
        "model",
        "input_video",
        "frames_processed",
        "resolution",
        "source_fps",
        "video_duration_sec",
        "total_detections",
        "avg_detections_per_frame",
        "max_detections_per_frame",
        "min_detections_per_frame",
        "frames_with_detection",
        "avg_confidence",
        "min_confidence",
        "max_confidence",
        "processing_time_seconds",
        "processing_fps",
        "conf_threshold",
        "device",
        "output_video",
        "total_unique_potholes",
        "total_minor",
        "total_moderate",
        "total_severe",
        "severe_percent",
        "average_severity_score",
    ]

    df = pd.DataFrame(results)
    # Only include columns that exist in the data
    available_cols = [c for c in columns if c in df.columns]
    df = df[available_cols]

    df.to_csv(csv_path, index=False)
    print(f"✓ Benchmark results saved to: {csv_path}")
    return csv_path


def print_summary(results: list[dict]):
    """Print a human-readable comparison summary to the terminal.

    Args:
        results: List of stats dictionaries from benchmark runs.
    """
    print()
    print("=" * 60)
    print("  ROADVISION AI — MODEL BENCHMARK RESULTS")
    print("=" * 60)
    print()
    print(f"  NOTE: 'Processing FPS' includes video decoding,")
    print(f"        inference, annotation, and encoding time.")
    print()

    for i, stats in enumerate(results):
        model_name = stats.get("model", "Unknown")
        info = get_model_info(model_name)

        print(f"  Model               : {info['description']}")
        print(f"  Frames              : {stats['frames_processed']}")
        print(f"  Total detections    : {stats['total_detections']}")
        print(f"  Avg detections/frame: {stats['avg_detections_per_frame']}")
        print(f"  Max detections/frame: {stats['max_detections_per_frame']}")
        print(f"  Min detections/frame: {stats['min_detections_per_frame']}")
        print(f"  Frames w/ detection : {stats['frames_with_detection']}")
        print(f"  Avg confidence      : {stats['avg_confidence']:.4f}")
        print(f"  Min confidence      : {stats['min_confidence']:.4f}")
        print(f"  Max confidence      : {stats['max_confidence']:.4f}")
        print(f"  Processing time     : {stats['processing_time_seconds']:.1f} sec")
        print(f"  Processing FPS      : {stats['processing_fps']}")
        print(f"  Device              : {stats['device']}")
        print(f"  Conf threshold      : {stats['conf_threshold']}")
        
        
        if 'total_unique_potholes' in stats:
            print(f"  Unique Potholes     : {stats['total_unique_potholes']}")
        if 'total_minor' in stats:
            print(f"  Minor / Mod / Severe: {stats['total_minor']} / {stats['total_moderate']} / {stats['total_severe']}")
            print(f"  Severe %            : {stats['severe_percent']}%")
            print(f"  Avg Severity Score  : {stats['average_severity_score']:.4f}")
            print(f"  Relative PSI        : {stats.get('relative_pothole_severity_index', 0):.4f}")
            print(f"  Worst Segment       : {stats.get('worst_segment', 'N/A')}")

        if i < len(results) - 1:
            print()
            print("  " + "-" * 56)
            print()

    print()
    print("=" * 60)
    print()
