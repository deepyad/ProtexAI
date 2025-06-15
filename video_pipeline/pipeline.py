import time
import logging
import yaml
import cv2
from collections import defaultdict
from pathlib import Path
from prometheus_client import start_http_server
from imagehash import phash
from PIL import Image
from ultralytics import YOLO
from .metrics import metrics  # Changed to relative import
from .utility import generate_report, calculate_deduplication_metrics, model_perf_metrics
from .yolo_utils import load_detection_model, detect_objects  # Fixed typo and made relative
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def process_video(input_path, output_dir, config_path):
    logger.info("Starting video processing pipeline")
    
    cap = None
    all_detections = []  # Moved outside try block to maintain scope
    hashes = set() 
    try:
        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Load detection model
        model = load_detection_model(config)

        # Video loading timing
        load_start = time.time()
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {input_path}")
        metrics.timings['video_loading'] = time.time() - load_start

        # Initialize metrics
        if metrics.frame_drops is not None:
            metrics.frame_drops.set(0)

        # Deduplication setup
        # hashes = set()
        threshold = config.get('deduplication', {}).get('threshold', 2)

        while True:
            metrics.frame_counts['total'] += 1
            read_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                metrics.frame_counts['dropped'] += 1
                if metrics.frame_drops is not None:
                    metrics.frame_drops.inc()
                break

            # Convert frame for hashing
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_hash = phash(pil_image)
            
            # Check for duplicates
            duplicate = any((current_hash - existing_hash) < threshold 
                          for existing_hash in hashes)
            if duplicate:
                metrics.frame_counts['duplicates'] += 1
                continue
            hashes.add(current_hash)

            # Update frames processed metric
            if metrics.frames_processed is not None:
                metrics.frames_processed.inc()

            # Frame processing
            proc_start = time.time()
            detections = detect_objects(frame, model)
            all_detections.extend(detections)  # Collect all detections
            metrics.timings['frame_processing'] += time.time() - proc_start
            

            for det in detections:
                metrics.detections[det['class']] += 1
                if metrics.detections_count is not None:
                    metrics.detections_count.labels(object_class=det['class']).inc()


            # Save frame
            save_start = time.time()

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            # cv2.imwrite(f"{output_dir}/frame_{metrics.frame_counts['success']:06d}.jpg", frame)
            
            if frame is not None and frame.size > 0:
                cv2.imwrite(f"{output_dir}/frame_{metrics.frame_counts['success']:06d}.jpg", frame)
            else:
                logger.warning(f"Skipping invalid frame {metrics.frame_counts['success']}")
            metrics.frame_counts['success'] += 1
            metrics.timings['io'] += time.time() - save_start
            # Update processing time metric
            if metrics.processing_time is not None:
                metrics.processing_time.observe(time.time() - read_start)
            
        # Calculate metrics
        dedup_metrics = calculate_deduplication_metrics(
            total_frames=metrics.frame_counts['total'],
            unique_frames=metrics.frame_counts['success']
        )

        perf_metrics = model_perf_metrics(all_detections)

        # Log and generate report
        logger.info(f"Deduplication Metrics: {dedup_metrics}")
        logger.info(f"Model Performance Metrics: {perf_metrics}")
        generate_report(metrics, output_dir, dedup_metrics, perf_metrics)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        if cap is not None:
            cap.release()
        logger.info("Pipeline completed with metrics: %s", {
            'frame_counts': metrics.frame_counts,
            'detections': dict(metrics.detections),
            'timings': metrics.timings,
            'duplicates': len(hashes)
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video processing pipeline")
    parser.add_argument("--input-video", required=True, help="Path to input video file")
    parser.add_argument("--output-dir", required=True, help="Directory to save output images/annotations")
    parser.add_argument("--model-config", default="default_model_config.yaml", help="Path to model config YAML")
    args = parser.parse_args()

    # Start Prometheus metrics server (if needed)
    start_http_server(8000)

    process_video(
        input_path=args.input_video,
        output_dir=args.output_dir,
        config_path=args.model_config
    )
