import json
import csv
from pathlib import Path

class Reporter:
    """Generates JSON, CSV, and matplotlib reports from the RoadAnalyzer output."""
    
    @staticmethod
    def save_json(report_data, output_path: Path):
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=4)
            
    @staticmethod
    def save_segment_csv(report_data, output_path: Path):
        segments = report_data.get('segments', [])
        if not segments:
            return
            
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=segments[0].keys())
            writer.writeheader()
            for seg in segments:
                writer.writerow(seg)
                
    @staticmethod
    def generate_chart(report_data, output_path: Path):
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg') # non-interactive backend
            
            segments = report_data.get('segments', [])
            if not segments:
                return
                
            seg_ids = [s['segment_id'] for s in segments]
            psis = [s['segment_psi'] for s in segments]
            
            plt.figure(figsize=(10, 5))
            plt.bar(seg_ids, psis, color='coral')
            plt.title('Relative Pothole Severity Index (PSI) per Video Segment')
            plt.xlabel('Segment ID')
            plt.ylabel('Segment PSI')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
        except ImportError:
            print("matplotlib is not installed. Skipping chart generation.")
