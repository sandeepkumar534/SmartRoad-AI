import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO


# ─── Visual Settings ────────────────────────────────────────────────────────
BOX_COLOR = (0, 255, 0)       # Green bounding boxes (BGR)
TEXT_COLOR = (255, 255, 255)   # White text
BG_COLOR = (0, 0, 0)          # Black text background
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 1


class VideoInfo:
    """Container for input video metadata."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Input video not found: {path}\n\n"
                f"Please place your road video at:\n"
                f"  {path.resolve()}\n\n"
                f"Supported formats: .mp4, .avi, .mov, .mkv"
            )

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Cannot open video: {path}\n\n"
                f"The file may be corrupted or use an unsupported codec.\n"
                f"Try re-encoding with FFmpeg:\n"
                f'  ffmpeg -i "{path}" -c:v libx264 output.mp4'
            )

        self.path = path
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0.0

        # Validate we can actually read frames
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(
                f"Video appears empty or unreadable: {path}\n"
                f"Could not decode the first frame."
            )

    def print_info(self):
        """Print a formatted video info table."""
        print()
        print("Input Video")
        print("─" * 40)
        print(f"  File       : {self.path.name}")
        print(f"  Path       : {self.path.resolve()}")
        print(f"  Resolution : {self.width} × {self.height}")
        print(f"  FPS        : {self.fps:.2f}")
        print(f"  Frames     : {self.frame_count}")
        print(f"  Duration   : {self.duration:.1f} sec")
        print("─" * 40)
        print()


def _select_device(device: str | None) -> str:
    
    if device is not None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            print(f"⚠ CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return device

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ CUDA available: {gpu_name}")
        return "cuda"
    else:
        print("ℹ CUDA not available. Using CPU.")
        return "cpu"


def _draw_detections(frame, results, conf_threshold: float):
   
    detections = 0
    confidences = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < conf_threshold:
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

            # Class name
            cls_id = int(boxes.cls[i])
            class_name = result.names.get(cls_id, f"class_{cls_id}")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

            # Draw label with background
            label = f"{class_name} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, FONT, FONT_SCALE, FONT_THICKNESS
            )
            label_y = max(y1, text_h + 10)
            cv2.rectangle(
                frame,
                (x1, label_y - text_h - 8),
                (x1 + text_w + 6, label_y + 4),
                BG_COLOR,
                cv2.FILLED,
            )
            cv2.putText(
                frame, label, (x1 + 3, label_y - 3),
                FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA,
            )

            detections += 1
            confidences.append(conf)

    return frame, detections, confidences


def process_video(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    conf_threshold: float = 0.25,
    device: str | None = None,
) -> dict:
  
    model_path = Path(model_path)
    input_path = Path(input_path)
    output_path = Path(output_path)

    # ── Validate model ───────────────────────────────────────────────────
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            f"Run the model download step first."
        )

    # ── Validate & report video info ─────────────────────────────────────
    video_info = VideoInfo(input_path)
    video_info.print_info()

    # ── Select device ────────────────────────────────────────────────────
    selected_device = _select_device(device)
    print(f"  Device     : {selected_device}")
    print(f"  Confidence : {conf_threshold}")
    print()

    # ── Load model ───────────────────────────────────────────────────────
    print(f"Loading model: {model_path.name}...")
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load YOLO model from {model_path}.\n"
            f"The weight file may be corrupted or incompatible.\n"
            f"Error: {exc}"
        ) from exc
    print(f"✓ Model loaded successfully.")
    print()

    # ── Open video ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for processing: {input_path}")

    # ── Setup writer ─────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc,
        video_info.fps,
        (video_info.width, video_info.height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot create output video: {output_path}\n"
            f"The 'mp4v' codec may not be available on your system.\n"
            f"Try installing OpenCV with FFmpeg support."
        )

    # ── Frame-by-frame inference ─────────────────────────────────────────
    all_detections_per_frame = []
    all_confidences = []
    frames_processed = 0

    print(f"Running inference...")
    start_time = time.perf_counter()

    pbar = tqdm(total=video_info.frame_count, desc="Processing", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model.predict(
            frame,
            conf=conf_threshold,
            device=selected_device,
            verbose=False,
        )

        # Draw detections and collect stats
        annotated_frame, det_count, confs = _draw_detections(
            frame, results, conf_threshold
        )

        all_detections_per_frame.append(det_count)
        all_confidences.extend(confs)
        frames_processed += 1

        writer.write(annotated_frame)
        pbar.update(1)

    pbar.close()
    end_time = time.perf_counter()

    # ── Cleanup ──────────────────────────────────────────────────────────
    cap.release()
    writer.release()

    total_time = end_time - start_time
    total_detections = sum(all_detections_per_frame)
    frames_with_detection = sum(1 for d in all_detections_per_frame if d > 0)

    stats = {
        # Video info
        "input_video": str(input_path),
        "resolution": f"{video_info.width}x{video_info.height}",
        "source_fps": round(video_info.fps, 2),
        "video_duration_sec": round(video_info.duration, 2),
        "frames_processed": frames_processed,
        # Detection stats
        "total_detections": total_detections,
        "avg_detections_per_frame": round(
            total_detections / max(frames_processed, 1), 2
        ),
        "max_detections_per_frame": max(all_detections_per_frame) if all_detections_per_frame else 0,
        "min_detections_per_frame": min(all_detections_per_frame) if all_detections_per_frame else 0,
        "frames_with_detection": frames_with_detection,
        # Confidence stats
        "avg_confidence": round(
            sum(all_confidences) / max(len(all_confidences), 1), 4
        ) if all_confidences else 0.0,
        "max_confidence": round(max(all_confidences), 4) if all_confidences else 0.0,
        "min_confidence": round(min(all_confidences), 4) if all_confidences else 0.0,
        # Performance stats
        "processing_time_seconds": round(total_time, 2),
        "processing_fps": round(
            frames_processed / max(total_time, 0.001), 2
        ),
        # Settings used
        "conf_threshold": conf_threshold,
        "device": selected_device,
        "output_video": str(output_path),
    }

 
    print()
    print(f"✓ Inference complete!")
    print(f"  Frames processed  : {frames_processed}")
    print(f"  Total detections  : {total_detections}")
    print(f"  Processing time   : {total_time:.1f} sec")
    print(f"  Processing FPS    : {stats['processing_fps']}")
    print(f"  Output saved to   : {output_path}")
    print()

    return stats
