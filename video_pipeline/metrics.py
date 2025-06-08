# metrics.py
from prometheus_client import Counter, Gauge, Histogram

class PipelineMetrics:
    def __init__(self):
        self.frames_processed = Counter(
            'frames_processed_total', 
            'Total frames processed'
        )
        self.detections_count = Counter(
            'detections_total', 
            'Total detections', 
            ['class']
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

# Singleton instance
metrics = PipelineMetrics()

