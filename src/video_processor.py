import time
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO

from src.detector import VideoInfo, _select_device
from src.tracking.tracker import PotholeTracker
from src.severity.estimator import SeverityEstimator, SeverityClass
from src.analytics import RoadAnalyzer, Reporter
import csv

# Colors
COLOR_MINOR = (0, 255, 0)       # Green
COLOR_MODERATE = (0, 255, 255)  # Yellow (BGR format)
COLOR_SEVERE = (0, 0, 255)      # Red
COLOR_TENTATIVE = (0, 165, 255) # Orange
COLOR_TEXT = (255, 255, 255)    # White

def get_severity_color(sev_class):
    if sev_class == SeverityClass.MINOR:
        return COLOR_MINOR
    elif sev_class == SeverityClass.MODERATE:
        return COLOR_MODERATE
    else:
        return COLOR_SEVERE

def _draw_hud(frame, tracker, fps, estimator, total_frames):
    """Draw the Global HUD Overlay."""
    minor, mod, severe = 0, 0, 0
    for t in tracker.get_confirmed_tracks():
        sev_class = estimator.estimate(t)['severity_class']
        if sev_class == SeverityClass.MINOR: minor += 1
        elif sev_class == SeverityClass.MODERATE: mod += 1
        elif sev_class == SeverityClass.SEVERE: severe += 1

    hud_text = [
        f"Unique Potholes : {tracker.total_unique_potholes}",
        f"Minor           : {minor}",
        f"Moderate        : {mod}",
        f"Severe          : {severe}",
        "",
        f"Frame           : {tracker.frame_count} / {total_frames}",
        f"Processing FPS  : {fps:.1f}"
    ]
    
    y0, dy = 60, 30
    for i, line in enumerate(hud_text):
        if not line: continue
        y = y0 + i * dy
        # Shadow/Outline
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        # Text
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)

def _draw_tracks(frame, tracks, estimator):
    """Draw bounding boxes and IDs for active tracks."""
    for t in tracks:
        # Hide TENTATIVE tracks for clean presentation video
        if t.state != 2: # NOT CONFIRMED
            continue
            
        x1, y1, x2, y2 = map(int, t.current_bbox)
        
        features = estimator.estimate(t)
        sev_class = features['severity_class']
        score = features['severity_score']
        color = get_severity_color(sev_class)
        label = f"ID:{t.track_id} | {sev_class} | {score:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)

def process_video_with_tracking(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    conf_threshold: float = 0.25,
    device: str | None = None,
) -> dict:
    """Run YOLO inference with the custom CMC temporal association tracker."""
    model_path = Path(model_path)
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    video_info = VideoInfo(input_path)
    video_info.print_info()
    
    selected_device = _select_device(device)
    
    print(f"Loading model for tracking: {model_path.name}...")
    model = YOLO(str(model_path))
    
    cap = cv2.VideoCapture(str(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc,
        video_info.fps,
        (video_info.width, video_info.height),
    )
    
    # Initialize Tracker & Estimator
    tracker = PotholeTracker(fps=video_info.fps, max_missed_seconds=0.5, iou_threshold=0.3)
    estimator = SeverityEstimator(video_info.width, video_info.height)
    
    all_detections_per_frame = []
    all_confidences = []
    
    frames_processed = 0
    start_time = time.perf_counter()
    pbar = tqdm(total=video_info.frame_count, desc="Tracking", unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        loop_start = time.perf_counter()
        
        # YOLO inference
        results = model.predict(frame, conf=conf_threshold, device=selected_device, verbose=False)
        
        # Extract raw detections
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                if conf >= conf_threshold:
                    x1, y1, x2, y2 = map(float, boxes.xyxy[i].tolist())
                    detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf})
                    
        all_detections_per_frame.append(len(detections))
        all_confidences.extend([d['conf'] for d in detections])
                    
        # Update custom tracker
        tracker.update(frame, detections)
        
        # Calculate current FPS
        loop_time = time.perf_counter() - loop_start
        current_fps = 1.0 / max(loop_time, 0.001)
        
        # Visualizations
        _draw_tracks(frame, tracker.get_active_tracks(), estimator)
        _draw_hud(frame, tracker, current_fps, estimator, video_info.frame_count)
        
        writer.write(frame)
        frames_processed += 1
        pbar.update(1)
        
    pbar.close()
    cap.release()
    writer.release()
    
    total_time = time.perf_counter() - start_time
    total_detections = sum(all_detections_per_frame)
    frames_with_detection = sum(1 for d in all_detections_per_frame if d > 0)
    
    # Final pass to calculate severities and export to CSV
    confirmed_tracks = tracker.get_confirmed_tracks()
    all_features = []
    
    csv_path = output_path.parent / "pothole_measurements.csv"
    with open(csv_path, mode='w', newline='') as f:
        fieldnames = ['track_id', 'peak_frame', 'severity_score', 'severity_class', 'max_bbox_area', 'max_width', 'max_height', 'median_area', 'track_duration', 'detection_count']
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        for t in confirmed_tracks:
            features = estimator.estimate(t)
            all_features.append(features)
            csv_writer.writerow(features)
            
    # Phase 4: Road Analytics & Reporting
    analyzer = RoadAnalyzer(fps=video_info.fps, total_frames=frames_processed)
    report_data = analyzer.analyze(all_features)
    
    Reporter.save_json(report_data, output_path.parent / "road_report.json")
    Reporter.save_segment_csv(report_data, output_path.parent / "road_summary.csv")
    Reporter.generate_chart(report_data, output_path.parent / "segment_analysis.png")
    
    global_stats = report_data['global_stats']
    
    stats = {
        "input_video": str(input_path),
        "resolution": f"{video_info.width}x{video_info.height}",
        "source_fps": round(video_info.fps, 2),
        "video_duration_sec": round(video_info.duration, 2),
        "total_unique_potholes": tracker.total_unique_potholes,
        "total_minor": global_stats['minor_count'],
        "total_moderate": global_stats['moderate_count'],
        "total_severe": global_stats['severe_count'],
        "severe_percent": global_stats['severe_percent'],
        "average_severity_score": global_stats['average_severity'],
        "relative_pothole_severity_index": global_stats['relative_pothole_severity_index'],
        "worst_segment": report_data['worst_segment_id'],
        "total_detections": total_detections,
        "avg_detections_per_frame": round(total_detections / max(frames_processed, 1), 2),
        "max_detections_per_frame": max(all_detections_per_frame) if all_detections_per_frame else 0,
        "min_detections_per_frame": min(all_detections_per_frame) if all_detections_per_frame else 0,
        "frames_with_detection": frames_with_detection,
        "avg_confidence": round(sum(all_confidences) / max(len(all_confidences), 1), 4) if all_confidences else 0.0,
        "max_confidence": round(max(all_confidences), 4) if all_confidences else 0.0,
        "min_confidence": round(min(all_confidences), 4) if all_confidences else 0.0,
        "frames_processed": frames_processed,
        "processing_time_seconds": round(total_time, 2),
        "processing_fps": round(frames_processed / max(total_time, 0.001), 2),
        "conf_threshold": conf_threshold,
        "device": selected_device,
        "output_video": str(output_path),
    }
    
    print(f"\n✓ Tracking complete!")
    print(f"  Unique Potholes : {tracker.total_unique_potholes}")
    print(f"  Processing FPS  : {stats['processing_fps']}")
    print(f"  Output saved to : {output_path}\n")
    
    return stats
