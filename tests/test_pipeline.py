import cv2
import pytest
import json
import numpy as np
from pathlib import Path
from unittest import mock
import prometheus_client

# Test configuration
TEST_VIDEO = "test-data/sample.mp4"
REFERENCE_IMAGE = "test-data/ref_frame.jpg"
SAMPLE_CONFIG = "default_model_config.yaml"

def test_frame_extraction(tmp_path):
    """Integration test for full pipeline execution"""
    from video_pipeline.pipeline import process_video
    
    output_dir = tmp_path / "output"
    process_video(TEST_VIDEO, output_dir, SAMPLE_CONFIG)
    
    # Verify outputs
    image_files = list(output_dir.glob("*.jpg"))
    assert len(image_files) > 0, "No output images generated"
    assert (output_dir / "annotations.json").exists(), "Missing COCO file"

def test_image_quality_and_consistency(tmp_path):
    """Validate image dimensions and file integrity"""
    from video_pipeline.pipeline import process_video
    
    output_dir = tmp_path / "output"
    process_video(TEST_VIDEO, output_dir, SAMPLE_CONFIG)
    
    # Check first image
    img_path = next(output_dir.glob("*.jpg"))
    img = cv2.imread(str(img_path))
    assert img.shape == (1080, 1920, 3), "Incorrect image dimensions"
    
    # Verify all images have consistent size
    for img_file in output_dir.glob("*.jpg"):
        img = cv2.imread(str(img_file))
        assert img.shape == (1080, 1920, 3), f"Size mismatch in {img_file.name}"

def test_coco_annotation_integrity(tmp_path):
    """Comprehensive COCO format validation"""
    from video_pipeline.pipeline import process_video
    
    output_dir = tmp_path / "output"
    process_video(TEST_VIDEO, output_dir, SAMPLE_CONFIG)
    
    with open(output_dir / "annotations.json") as f:
        data = json.load(f)
    
    # Structural validation
    required_keys = {"info", "images", "annotations", "categories"}
    assert required_keys.issubset(data.keys()), "Missing required COCO keys"
    
    # Content validation
    image_ids = {img["id"] for img in data["images"]}
    category_ids = {cat["id"] for cat in data["categories"]}
    
    for ann in data["annotations"]:
        assert ann["image_id"] in image_ids, "Orphaned annotation"
        assert ann["category_id"] in category_ids, "Invalid category reference"
        assert len(ann["bbox"]) == 4, "Invalid bounding box format"
        assert all(x >= 0 for x in ann["bbox"]), "Negative bbox coordinates"

def test_deduplication_logic():
    """Verify frame deduplication accuracy"""
    from video_pipeline.deduplication import deduplicate_frames
    
    # Create test frames
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    frame3 = np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    # Test different thresholds
    assert len(deduplicate_frames([frame1, frame2], threshold=0)) == 1
    assert len(deduplicate_frames([frame1, frame2, frame3], threshold=2)) == 2
    assert len(deduplicate_frames([frame1, frame3], threshold=50)) == 2

def test_detection_parsing():
    """Validate object detection output structure"""
    from Protex_Project.video_pipeline.yolo_utils import detect_objects, load_detection_model
    
    # Test with blank frame
    model = load_detection_model({"model": {"path": "yolov8n.pt"}})
    blank_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    detections = detect_objects(blank_frame, model)
    
    assert isinstance(detections, list)
    for det in detections:
        assert "class" in det and isinstance(det["class"], str)
        assert "confidence" in det and 0 <= det["confidence"] <= 1
        assert "bbox" in det and len(det["bbox"]) == 4

def test_metrics_collection(tmp_path):
    """Verify metrics tracking completeness"""
    from video_pipeline.pipeline import process_video
    from video_pipeline.metrics import metrics
    
    output_dir = tmp_path / "output"
    process_video(TEST_VIDEO, output_dir, SAMPLE_CONFIG)
    
    # Basic counts
    assert metrics.frame_counts['success'] > 0
    assert metrics.detections.total() >= 0
    assert metrics.timings['video_loading'] > 0
    
    # Ratios
    drop_ratio = metrics.frame_counts['dropped'] / metrics.frame_counts['total']
    assert 0 <= drop_ratio <= 1, "Invalid drop ratio"

def test_error_handling(tmp_path):
    """Verify pipeline robustness"""
    from video_pipeline.pipeline import process_video
    
    # Test invalid video path
    with pytest.raises(Exception):
        process_video("invalid.mp4", tmp_path, SAMPLE_CONFIG)
    
    # Test missing config
    with pytest.raises(FileNotFoundError):
        process_video(TEST_VIDEO, tmp_path, "missing_config.yaml")

def test_prometheus_metrics(tmp_path):
    """Validate metrics exposure"""
    from video_pipeline.pipeline import process_video
    
    output_dir = tmp_path / "output"
    process_video(TEST_VIDEO, output_dir, SAMPLE_CONFIG)
    
    # Check registered metrics
    registry = prometheus_client.REGISTRY
    assert registry.get_sample_value('frames_processed_total') > 0
    assert registry.get_sample_value('detections_total') >= 0

def test_report_generation(tmp_path):
    """Validate report content and structure"""
    from video_pipeline.pipeline import process_video
    from video_pipeline.utility import generate_report
    
    output_dir = tmp_path / "output"
    process_video(TEST_VIDEO, output_dir, SAMPLE_CONFIG)
    
    report_file = output_dir / "pipeline_report.md"
    assert report_file.exists()
    
    content = report_file.read_text()
    assert "## Summary Statistics" in content
    assert "## Timing Breakdown" in content
    assert "## Class Distribution" in content

def test_config_override(tmp_path):
    """Verify configuration flexibility"""
    from video_pipeline.pipeline import process_video
    from Protex_Project.video_pipeline.yolo_utils import load_detection_model
    
    custom_config = tmp_path / "custom_config.yaml"
    custom_config.write_text("""
    model:
      path: yolov8s.pt
      confidence: 0.7
    deduplication:
      threshold: 5
    """)
    
    output_dir = tmp_path / "output"
    process_video(TEST_VIDEO, output_dir, str(custom_config))
    
    # Verify config applied
    model = load_detection_model({"model": {"path": "yolov8s.pt"}})
    assert model.conf == 0.7

def test_output_validation(tmp_path):
    from validate_outputs import validate
    from video_pipeline.pipeline import process_video
    output_dir = tmp_path / "output"
    # Run pipeline
    process_video(TEST_VIDEO, output_dir, SAMPLE_CONFIG)
    # Validate
    validate(str(tmp_path / "output"))
