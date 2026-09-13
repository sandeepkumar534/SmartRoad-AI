import numpy as np
from .cmc import CameraMotionCompensator
from .matcher import compute_iou_distance, linear_assignment

class TrackState:
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    TERMINATED = 4

class Track:
    _id_counter = 1
    
    def __init__(self, bbox, confidence, frame_idx):
        self.track_id = Track._id_counter
        Track._id_counter += 1
        
        self.current_bbox = bbox
        self.state = TrackState.TENTATIVE
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.hits = 1
        self.missed_frames = 0
        self.age = 1
        
        self.max_confidence = confidence
        self.bbox_history = [bbox]
        self.frame_history = [frame_idx]
        
    def predict(self, affine_matrix):
        """Update bounding box location using camera motion."""
        if affine_matrix is not None:
            self.current_bbox = CameraMotionCompensator.apply_affine_to_bbox(self.current_bbox, affine_matrix)
        self.age += 1
        self.missed_frames += 1
        
        # If we missed it, it goes to LOST (if it was CONFIRMED) or stays TENTATIVE
        if self.state == TrackState.CONFIRMED and self.missed_frames > 0:
            self.state = TrackState.LOST
            
    def update(self, bbox, confidence, frame_idx):
        """Update track with new matched detection."""
        self.current_bbox = bbox
        self.last_frame = frame_idx
        self.hits += 1
        self.missed_frames = 0
        self.max_confidence = max(self.max_confidence, confidence)
        self.bbox_history.append(bbox)
        self.frame_history.append(frame_idx)
        
        # State transitions
        if self.state == TrackState.TENTATIVE and self.hits >= 3:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

class PotholeTracker:
    def __init__(self, fps, max_missed_seconds=0.5, iou_threshold=0.3):
        # Calculate max_missed_frames dynamically from FPS
        self.max_missed_frames = int(fps * max_missed_seconds)
        
        self.max_cost = 1.0 - iou_threshold 
        
        self.tracks = []
        self.cmc = CameraMotionCompensator()
        self.frame_count = 0
        
        # Metrics
        self.total_unique_potholes = 0

    def get_active_tracks(self):
        """Return tracks that are currently visible (CONFIRMED or TENTATIVE)."""
        return [t for t in self.tracks if t.state in [TrackState.CONFIRMED, TrackState.TENTATIVE]]

    def get_confirmed_tracks(self):
        """Return tracks that are definitively potholes (hits >= 3)."""
        return [t for t in self.tracks if t.hits >= 3]

    def update(self, frame, detections):
        """
        Update the tracker with the current frame and YOLO detections.
        detections: list of dicts {'bbox': [x1, y1, x2, y2], 'conf': float}
        """
        self.frame_count += 1
        
        # 1. Estimate Camera Motion and predict existing tracks
        affine_matrix = self.cmc.compute_affine_matrix(frame)
        for track in self.tracks:
            track.predict(affine_matrix)
            
        # Filter active tracks for matching
        active_tracks = [t for t in self.tracks if t.state != TrackState.TERMINATED]
        
        if len(active_tracks) == 0:
            # Create new tracks for all detections
            for det in detections:
                self.tracks.append(Track(det['bbox'], det['conf'], self.frame_count))
            return
            
        if len(detections) == 0:
            # No detections, just handle terminations
            self._handle_terminations(active_tracks)
            return

        # 2. Association (IoU Distance)
        track_bboxes = [t.current_bbox for t in active_tracks]
        det_bboxes = [d['bbox'] for d in detections]
        
        cost_matrix = compute_iou_distance(track_bboxes, det_bboxes)
        
        matches, unmatched_tracks, unmatched_dets = linear_assignment(cost_matrix, max_cost=self.max_cost)
        
        # 3. Update Matched Tracks
        for r, c in matches:
            track = active_tracks[r]
            det = detections[c]
            
            was_tentative = (track.state == TrackState.TENTATIVE)
            track.update(det['bbox'], det['conf'], self.frame_count)
            
            # Metric update
            if was_tentative and track.state == TrackState.CONFIRMED:
                self.total_unique_potholes += 1
                
        # 4. Create New Tracks
        for c in unmatched_dets:
            det = detections[c]
            self.tracks.append(Track(det['bbox'], det['conf'], self.frame_count))
            
        # 5. Handle Terminations
        self._handle_terminations(active_tracks)

    def _handle_terminations(self, active_tracks):
        for track in active_tracks:
            # Terminate if lost for too long
            if track.missed_frames > self.max_missed_frames:
                track.state = TrackState.TERMINATED
                continue
                
            # Note: We do NOT terminate if it touches the bottom edge, per user request.
            # We rely on missed_frames and potentially future ROI checks.
