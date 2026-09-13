import numpy as np

class SeverityClass:
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"

class SeverityEstimator:
    """
    Estimates the Relative Visual Severity of a pothole track based on geometric
    and temporal features, independent of detection confidence.
    """
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height

    def extract_features(self, track):
        """
        Extracts geometric and temporal features from a Track's history.
        """
        areas = []
        widths = []
        heights = []
        
        for bbox in track.bbox_history:
            x1, y1, x2, y2 = bbox
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            areas.append(w * h)
            widths.append(w)
            heights.append(h)
            
        peak_idx = int(np.argmax(areas)) if areas else 0
        peak_frame = track.frame_history[peak_idx] if hasattr(track, 'frame_history') and track.frame_history else track.first_frame
            
        features = {
            'track_id': track.track_id,
            'peak_frame': peak_frame,
            'max_bbox_area': float(max(areas)) if areas else 0.0,
            'max_width': float(max(widths)) if widths else 0.0,
            'max_height': float(max(heights)) if heights else 0.0,
            'median_area': float(np.median(areas)) if areas else 0.0,
            'track_duration': len(track.bbox_history),
            'detection_count': track.hits
        }
        
        return features

    def calculate_severity(self, features):
        """
        Calculates a relative severity score (0.0 to 1.0) based on geometric features.
        Weights:
            50% Size (Area)
            30% Span (Width)
            20% Persistence (Duration)
        """
        # Normalize features relative to frame dimensions
        # These normalization denominators are rough heuristics based on typical dashcam perspective
        # A pothole taking up 5% of the frame is considered extremely large (SEVERE).
        norm_area = min(features['max_bbox_area'] / (self.frame_area * 0.05), 1.0)
        
        # A pothole spanning 30% of the frame width is huge
        norm_width = min(features['max_width'] / (self.frame_width * 0.3), 1.0)
        
        # A track lasting 30 frames (approx 1 second) is a strong persistent signal
        norm_duration = min(features['track_duration'] / 30.0, 1.0)
        
        # Calculate Weighted Score
        score = (0.50 * norm_area) + (0.30 * norm_width) + (0.20 * norm_duration)
        score = min(max(score, 0.0), 1.0) # Clamp between 0 and 1
        
        # Classify
        if score < 0.35:
            severity_class = SeverityClass.MINOR
        elif score < 0.70:
            severity_class = SeverityClass.MODERATE
        else:
            severity_class = SeverityClass.SEVERE
            
        return score, severity_class

    def estimate(self, track):
        """
        Main entry point for estimating severity of a track.
        Returns a dict of all features + score + class.
        """
        features = self.extract_features(track)
        score, sev_class = self.calculate_severity(features)
        
        features['severity_score'] = round(score, 4)
        features['severity_class'] = sev_class
        
        return features
