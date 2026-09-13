from collections import defaultdict
from src.severity.estimator import SeverityClass

class RoadAnalyzer:
    """
    Computes road-level statistics and temporal segmentation from a list of
    pothole severity features.
    """
    def __init__(self, fps, total_frames, segment_length_frames=100):
        self.fps = fps
        self.total_frames = total_frames
        self.segment_length_frames = segment_length_frames
        self.video_duration_sec = total_frames / max(fps, 1.0)
        
    def analyze(self, all_pothole_features):
        """
        Analyzes the list of pothole feature dictionaries and returns a complete report.
        """
        total_potholes = len(all_pothole_features)
        minor_count = sum(1 for p in all_pothole_features if p['severity_class'] == SeverityClass.MINOR)
        moderate_count = sum(1 for p in all_pothole_features if p['severity_class'] == SeverityClass.MODERATE)
        severe_count = sum(1 for p in all_pothole_features if p['severity_class'] == SeverityClass.SEVERE)
        
        total_severity_score = sum(p['severity_score'] for p in all_pothole_features)
        
        # Pothole Severity Index (PSI): Severity load per 10 seconds of driving
        # This normalizes the severity score over the length of the video.
        time_windows_10s = max(self.video_duration_sec / 10.0, 1.0)
        global_psi = total_severity_score / time_windows_10s
        
        avg_severity = total_severity_score / max(total_potholes, 1)
        max_severity = max([p['severity_score'] for p in all_pothole_features]) if all_pothole_features else 0.0
        avg_duration = sum(p['track_duration'] for p in all_pothole_features) / max(total_potholes, 1)
        
        # Segment analysis
        segments = defaultdict(lambda: {'minor': 0, 'moderate': 0, 'severe': 0, 'total_score': 0.0, 'potholes': 0})
        
        for p in all_pothole_features:
            # Assign to segment based on the frame where it reached peak severity
            peak_frame = p.get('peak_frame', 0)
            seg_idx = peak_frame // self.segment_length_frames
            
            seg = segments[seg_idx]
            seg['potholes'] += 1
            seg['total_score'] += p['severity_score']
            
            if p['severity_class'] == SeverityClass.MINOR:
                seg['minor'] += 1
            elif p['severity_class'] == SeverityClass.MODERATE:
                seg['moderate'] += 1
            else:
                seg['severe'] += 1
                
        # Format segments for output
        segment_list = []
        worst_segment_idx = -1
        max_seg_psi = -1.0
        
        num_segments = (self.total_frames // self.segment_length_frames) + 1
        for i in range(num_segments):
            start_f = i * self.segment_length_frames
            end_f = min(start_f + self.segment_length_frames - 1, self.total_frames)
            
            s_data = segments.get(i, {'minor': 0, 'moderate': 0, 'severe': 0, 'total_score': 0.0, 'potholes': 0})
            
            # Segment PSI (normalized to 10 seconds for consistency, even if segment is shorter)
            seg_duration_sec = (end_f - start_f + 1) / max(self.fps, 1.0)
            seg_time_windows = max(seg_duration_sec / 10.0, 0.1)
            seg_psi = s_data['total_score'] / seg_time_windows
            
            segment_list.append({
                'segment_id': i + 1,
                'start_frame': start_f,
                'end_frame': end_f,
                'unique_potholes': s_data['potholes'],
                'minor': s_data['minor'],
                'moderate': s_data['moderate'],
                'severe': s_data['severe'],
                'segment_psi': round(seg_psi, 4),
                'average_severity': round(s_data['total_score'] / max(s_data['potholes'], 1), 4)
            })
            
            if seg_psi > max_seg_psi:
                max_seg_psi = seg_psi
                worst_segment_idx = i + 1
                
        return {
            'global_stats': {
                'total_potholes': total_potholes,
                'minor_count': minor_count,
                'moderate_count': moderate_count,
                'severe_count': severe_count,
                'minor_percent': round((minor_count / max(total_potholes, 1)) * 100, 1),
                'moderate_percent': round((moderate_count / max(total_potholes, 1)) * 100, 1),
                'severe_percent': round((severe_count / max(total_potholes, 1)) * 100, 1),
                'average_severity': round(avg_severity, 4),
                'max_severity': round(max_severity, 4),
                'average_duration_frames': round(avg_duration, 1),
                'relative_pothole_severity_index': round(global_psi, 4)
            },
            'worst_segment_id': worst_segment_idx,
            'segments': segment_list
        }
