# video_pipeline/utility.py
from collections import Counter
from pathlib import Path
import json
import cv2 
from PIL import Image  # For Image.fromarray()
from imagehash import phash  # For perceptual hashing

def generate_report(metrics, output_dir, dedup_metrics, perf_metrics):
    """Generate comprehensive pipeline report"""
    report_content = f"""# Video Processing Pipeline Report

## Summary Statistics
- **Total Frames Processed**: {metrics.frame_counts['total']}
- **Successful Frames**: {metrics.frame_counts['success']}
- **Dropped Frames**: {metrics.frame_counts['dropped']}
- **Duplicate Frames**: {metrics.frame_counts['duplicates']}
- **Total Detections**: {sum(metrics.detections.values())}

## Deduplication Metrics
- **Original Frames**: {dedup_metrics['original_frames']}
- **Unique Frames**: {dedup_metrics['unique_frames']}
- **Reduction Ratio**: {dedup_metrics['reduction_ratio']}

## Timing Breakdown
| Stage | Time (seconds) | Percentage |
|-------|---------------|------------|
| Video Loading | {metrics.timings['video_loading']:.2f} | {(metrics.timings['video_loading']/sum(metrics.timings.values()))*100:.1f}% |
| Frame Processing | {metrics.timings['frame_processing']:.2f} | {(metrics.timings['frame_processing']/sum(metrics.timings.values()))*100:.1f}% |
| Detection | {metrics.timings['detection']:.2f} | {(metrics.timings['detection']/sum(metrics.timings.values()))*100:.1f}% |
| I/O Operations | {metrics.timings['io']:.2f} | {(metrics.timings['io']/sum(metrics.timings.values()))*100:.1f}% |

## Performance Metrics
- **Average Confidence**: {perf_metrics.get('avg_confidence', 0):.3f}

## Class Distribution
"""
    
    class_dist = perf_metrics.get('class_distribution', {})
    if class_dist:
        report_content += "| Class | Count | Percentage |\n|-------|-------|------------|\n"
        total_detections = sum(class_dist.values())
        for cls, count in class_dist.items():
            percentage = (count/total_detections)*100 if total_detections > 0 else 0
            report_content += f"| {cls} | {count} | {percentage:.1f}% |\n"
    else:
        report_content += "No detections found in processed frames.\n"
    
    # Save report
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/pipeline_report.md", "w") as f:
        f.write(report_content)
    
    # Generate COCO annotations file
    generate_coco_annotations(metrics, output_dir, class_dist)

def generate_coco_annotations(metrics, output_dir, class_dist):
    """Generate COCO format annotations"""
    coco_data = {
        "info": {
            "description": "Video Pipeline Generated Dataset",
            "version": "1.0",
            "frame_count": metrics.frame_counts['success']
        },
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    category_id = 1
    for class_name in class_dist.keys():
        coco_data["categories"].append({
            "id": category_id,
            "name": class_name,
            "supercategory": "object"
        })
        category_id += 1
    
    # Add images (simplified - no actual detections since they're failing)
    for i in range(metrics.frame_counts['success']):
        coco_data["images"].append({
            "id": i + 1,
            "file_name": f"frame_{i:06d}.jpg",
            "width": 1920,  # Default dimensions
            "height": 1080
        })
    
    # Save COCO file
    with open(f"{output_dir}/annotations.json", "w") as f:
        json.dump(coco_data, f, indent=2)

def calculate_deduplication_metrics(total_frames, unique_frames):
    """Calculate deduplication efficiency metrics"""
    return {
        "original_frames": total_frames,
        "unique_frames": unique_frames,
        "reduction_ratio": f"{(1 - unique_frames/total_frames)*100:.1f}%" if total_frames > 0 else "0.0%"
    }

def model_perf_metrics(detections):
    """Calculate model performance statistics"""
    if not detections:
        return {"avg_confidence": 0, "class_distribution": {}}
    
    avg_conf = sum(d['confidence'] for d in detections) / len(detections)
    class_dist = Counter(d['class'] for d in detections)
    
    return {
        "avg_confidence": avg_conf,
        "class_distribution": dict(class_dist)
    }

def deduplicate_frames(frames, threshold=2):
    hashes = set()
    unique_frames = []
    for frame in frames:
        # Convert to RGB if needed
        if frame.shape[2] == 3:  # Assume BGR if 3 channels
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
            
        pil_image = Image.fromarray(frame_rgb)
        current_hash = phash(pil_image)
        duplicate = any((current_hash - existing_hash) <= threshold 
                      for existing_hash in hashes)
        if not duplicate:
            hashes.add(current_hash)
            unique_frames.append(frame)
    return unique_frames

