"""
Downloads and manages pretrained YOLO pothole detection model weights.
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


# ─── Model Registry ─────────────────────────────────────────────────────────
# Maps friendly model names to their HuggingFace repo details and local paths.
MODEL_REGISTRY = {
    "yolov8": {
        "repo_id": "Samdutse/pothole-yolov8",
        "filename": "best.pt",
        "local_name": "pothole_yolov8.pt",
        "description": "YOLOv8 Pothole Detector (Samdutse)",
    },
}

# Default directory for storing model weights
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def get_available_models() -> list[str]:
    """Return list of available model keys."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_key: str) -> dict:
    """Return registry info for a model key.

    Raises:
        ValueError: If model_key is not in the registry.
    """
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(get_available_models())
        raise ValueError(
            f"Unknown model '{model_key}'. Available models: {available}"
        )
    return MODEL_REGISTRY[model_key]


def download_model(model_key: str, models_dir: Path | None = None) -> Path:
    """Download model weights from HuggingFace if not already cached locally.

    Args:
        model_key: Friendly name of the model (e.g. 'yolov8', 'yolo26').
        models_dir: Directory to store weights in. Defaults to project's models/.

    Returns:
        Path to the local .pt weight file.

    Raises:
        ValueError: If model_key is not recognized.
        RuntimeError: If the download fails.
    """
    info = get_model_info(model_key)  # raises ValueError if unknown
    if models_dir is None:
        models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    local_path = models_dir / info["local_name"]

    # ── Check cache ──────────────────────────────────────────────────────
    if local_path.exists():
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model already cached: {local_path}  ({size_mb:.1f} MB)")
        return local_path

    # ── Download from HuggingFace ────────────────────────────────────────
    print(f"⬇ Downloading {info['description']}...")
    print(f"  Repository : {info['repo_id']}")
    print(f"  File       : {info['filename']}")
    print(f"  Destination: {local_path}")
    print()

    try:
        downloaded_path = hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            local_dir=None,  # use HF cache
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download model '{model_key}' from HuggingFace.\n"
            f"  Repository: {info['repo_id']}\n"
            f"  Error: {exc}\n\n"
            f"Possible fixes:\n"
            f"  1. Check your internet connection.\n"
            f"  2. Verify the repository exists: https://huggingface.co/{info['repo_id']}\n"
            f"  3. If the repo is private, run: huggingface-cli login"
        ) from exc

    # Copy from HF cache to our models/ directory for a clean project layout
    try:
        shutil.copy2(downloaded_path, local_path)
    except Exception as exc:
        raise RuntimeError(
            f"Downloaded model but failed to copy to {local_path}.\n"
            f"  Source: {downloaded_path}\n"
            f"  Error: {exc}"
        ) from exc

    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"✓ Model saved: {local_path}  ({size_mb:.1f} MB)")
    print()

    return local_path


def ensure_models(model_keys: list[str], models_dir: Path | None = None) -> dict[str, Path]:
    """Download/verify multiple models and return a mapping of key -> path.

    Args:
        model_keys: List of model keys to ensure are available.
        models_dir: Directory to store weights in.

    Returns:
        Dict mapping model key to local weight file path.
    """
    paths = {}
    for key in model_keys:
        paths[key] = download_model(key, models_dir)
    return paths
