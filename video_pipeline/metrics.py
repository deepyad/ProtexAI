# video_pipeline/metrics.py
from prometheus_client import Counter, Gauge, Histogram
from collections import defaultdict

class PipelineMetrics:
    def __init__(self):
        # Prometheus metrics
        self.frames_processed = Counter(
            'frames_processed_total', 
            'Total frames processed'
        )
        self.detections_count = Counter(
            'detections_total', 
            'Total detections', 
            ['object_class']
        )
        self.processing_time = Histogram(
            'processing_seconds',
            'Frame processing time',
            buckets=[0.1, 0.5, 1, 2, 5]
        )
        self.frame_drops = Gauge(
            'frame_drops_total',
            'Total dropped frames'
        )
        
        self.frame_counts = {
            'total': 0,
            'success': 0,
            'dropped': 0,
            'duplicates': 0
        }
        
        self.detections = defaultdict(int)
        
        self.timings = {
            'video_loading': 0.0,
            'frame_processing': 0.0,
            'detection': 0.0,
            'io': 0.0
        }

# Singleton instance
metrics = PipelineMetrics()
