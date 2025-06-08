from collections import Counter
import time
import logging
from collections import defaultdict

def generate_report(metrics, output_dir):
    total_time = time.time() - metrics.start_time
    report = f"""
# Video Processing Report

## Summary Statistics
- **Total Frames Processed**: {metrics.frame_counts['total']}
- **Successful Frames**: {metrics.frame_counts['success']}
- **Dropped Frames**: {metrics.frame_counts['dropped']}
- **Frame Drop Ratio**: {metrics.frame_counts['dropped']/metrics.frame_counts['total']:.2%}
- **Total Detections**: {sum(metrics.detections.values())}

## Timing Breakdown
| Stage | Time (s) | Percentage |
|-------|----------|------------|
| Video Loading | {metrics.timings['video_loading']:.2f} | {(metrics.timings['video_loading']/total_time):.2%} |
| Frame Processing | {metrics.timings['frame_processing']:.2f} | {(metrics.timings['frame_processing']/total_time):.2%} |
| Detection | {metrics.timings['detection']:.2f} | {(metrics.timings['detection']/total_time):.2%} |
| I/O Operations | {metrics.timings['io']:.2f} | {(metrics.timings['io']/total_time):.2%} |

## Class Distribution
{class_distribution_table(metrics.detections)}
    """
    
    with open(f"{output_dir}/pipeline_report.md", "w") as f:
        f.write(report)

def class_distribution_table(detections):
    if not detections:
        return "No detections found"
        
    table = "| Class | Count | Percentage |\n|------|-------|------------|\n"
    total = sum(detections.values())
    for cls, count in detections.items():
        table += f"| {cls} | {count} | {count/total:.2%} |\n"
    return table


def calculate_deduplication_metrics(total_frames, unique_frames):
    return {
        "original_frames": total_frames,
        "unique_frames": unique_frames,
        "reduction_ratio": f"{(1 - unique_frames/total_frames):.1%}" if total_frames else "N/A"
    }

def model_perf_metrics(detections):
    if not detections:
        return {"avg_confidence": 0, "class_distribution": {}}
    avg_conf = sum(d['confidence'] for d in detections) / len(detections)
    class_dist = Counter(d['class'] for d in detections)
    return {
        "avg_confidence": avg_conf,
        "class_distribution": dict(class_dist)
    }
