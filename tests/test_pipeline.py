import cv2
import pytest
from pathlib import Path
import json
import numpy as np

def test_frame_extraction(tmp_path):
    """Test frame extraction from sample video"""
    from video_pipeline.pipeline import process_video
    
    output_dir = tmp_path / "output"
    process_video("input/sample.mp4", str(output_dir), "default_model_config.yaml")
    
    assert len(list(output_dir.glob("*.jpg"))) > 0
    assert (output_dir / "annotations.json").exists()

def test_image_quality():
    """Verify images are saved with proper dimensions"""
    test_imgs = list(Path("output").glob("*.jpg"))
    if test_imgs:
        test_img = cv2.imread(str(test_imgs[0]))
        assert test_img is not None
        assert len(test_img.shape) == 3

def test_coco_annotations():
    """Validate COCO annotation structure"""
    try:
        with open("output/annotations.json") as f:
            data = json.load(f)
        
        required_keys = {"images", "annotations", "categories", "info"}
        assert required_keys.issubset(data.keys())
        
        # Validate structure
        for img in data["images"]:
            assert "id" in img and "file_name" in img
            assert "width" in img and "height" in img
        
        for cat in data["categories"]:
            assert "id" in cat and "name" in cat
            
    except FileNotFoundError:
        pytest.skip("COCO annotations file not found")

def test_deduplication():
    """Test deduplication logic"""
    from video_pipeline.utility import deduplicate_frames
    
    # Create test frames
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = frame1.copy()  # Duplicate
    frame3 = np.ones((100, 100, 3), dtype=np.uint8) * 255  # Different
    frames = [frame1, frame2, frame3]
    
    unique_frames = deduplicate_frames(frames, threshold=0)
    assert len(unique_frames) == 2  

def test_detection_parsing():
    """Test YOLO detection parsing"""
    from video_pipeline.yolo_utils import detect_objects, load_detection_model  # Fixed typo
    
    config = {"model": {"path": "yolov8n.pt", "confidence": 0.5}}
    model = load_detection_model(config)
    
    # Use a test frame
    frame = np.zeros((640, 480, 3), dtype=np.uint8)
    detections = detect_objects(frame, model)
    
    assert isinstance(detections, list)
    # Test structure if detections exist
    for det in detections:
        assert "class" in det
        assert "confidence" in det
        assert "bbox" in det
        assert len(det["bbox"]) == 4

def test_metrics_functions():
    """Test utility functions"""
    from video_pipeline.utility import calculate_deduplication_metrics, model_perf_metrics
    
    # Test deduplication metrics
    dedup = calculate_deduplication_metrics(10, 8)
    assert dedup["original_frames"] == 10
    assert dedup["unique_frames"] == 8
    assert "reduction_ratio" in dedup

    # Test performance metrics
    detections = [
        {"class": "person", "confidence": 0.9}, 
        {"class": "car", "confidence": 0.8}
    ]
    perf = model_perf_metrics(detections)
    # assert perf["avg_confidence"] == 0.85
    # assert perf["avg_confidence"] == pytest.approx(0.85)
    # or
    assert round(perf["avg_confidence"], 2) == 0.85
    assert perf["class_distribution"]["person"] == 1

# Additional comprehensive test cases
def test_empty_video(tmp_path):
    """Test pipeline with empty/corrupted video"""
    from video_pipeline.pipeline import process_video
    
    empty_video = tmp_path / "empty.mp4"
    empty_video.write_bytes(b"")
    output_dir = tmp_path / "output"
    
    with pytest.raises(Exception):
        process_video(str(empty_video), str(output_dir), "default_model_config.yaml")

def test_missing_config(tmp_path):
    """Test pipeline with missing config file"""
    from video_pipeline.pipeline import process_video
    
    output_dir = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        process_video("input/sample.mp4", str(output_dir), "nonexistent_config.yaml")

def test_report_generation(tmp_path):
    """Test report file generation"""
    from video_pipeline.utility import generate_report
    from video_pipeline.metrics import metrics
    
    # Mock metrics data
    metrics.frame_counts = {'total': 100, 'success': 95, 'dropped': 3, 'duplicates': 2}
    metrics.timings = {'video_loading': 1.0, 'frame_processing': 5.0, 'detection': 2.0, 'io': 1.0}
    
    dedup_metrics = {"original_frames": 100, "unique_frames": 95, "reduction_ratio": "5.0%"}
    perf_metrics = {"avg_confidence": 0.85, "class_distribution": {"person": 10, "car": 5}}
    
    generate_report(metrics, str(tmp_path), dedup_metrics, perf_metrics)
    
    report_file = tmp_path / "pipeline_report.md"
    assert report_file.exists()
    
    content = report_file.read_text()
    assert "Summary Statistics" in content
    assert "Timing Breakdown" in content
    assert "Class Distribution" in content

def test_prometheus_metrics():
    """Test Prometheus metrics functionality"""
    from video_pipeline.metrics import metrics
    
    # Test metric increments
    initial_count = metrics.frames_processed._value.get()
    metrics.frames_processed.inc()
    assert metrics.frames_processed._value.get() == initial_count + 1
    
    # Test detection counting
    metrics.detections_count.labels(object_class="person").inc()


def test_output_validation(tmp_path):
    """Test output validation script"""
    # Create mock output structure
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create mock image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(output_dir / "frame_000001.jpg"), img)

    # Create mock COCO file
    coco_data = {
        "info": {"description": "Test", "frame_count": 1},
        "images": [{"id": 1, "file_name": "frame_000001.jpg", "width": 640, "height": 480}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50]}],
        "categories": [{"id": 1, "name": "person"}]
    }
    
    with open(output_dir / "annotations.json", "w") as f:
        json.dump(coco_data, f)
    
    # Test validation
    from tests.validate_outputs import validate
    validate(str(output_dir))  # Should not raise an error

def test_config_override(tmp_path):
    """Test configuration flexibility"""
    from video_pipeline.yolo_utils import load_detection_model
    
    custom_config = {
        "model": {
            "path": "yolov8n.pt",
            "confidence": 0.7
        }
    }
    
    model = load_detection_model(custom_config)
    assert model.conf == 0.7

def test_pipeline_metrics_integration():
    """Test metrics collection during pipeline execution"""
    from video_pipeline.metrics import metrics
    
    # Verify metrics structure
    assert hasattr(metrics, 'frame_counts')
    assert hasattr(metrics, 'detections')
    assert hasattr(metrics, 'timings')
    
    # Test metrics updates
    metrics.frame_counts['total'] += 1
    assert metrics.frame_counts['total'] >= 1

def test_detection_with_sample_frame():
    """Test detection with actual frame data"""
    from video_pipeline.yolo_utils import load_detection_model, detect_objects
    
    # Create a realistic test frame (BGR format)
    frame = np.zeros((640, 480, 3), dtype=np.uint8)
    frame[100:200, 100:200] = 255  # Add white square
    
    config = {"model": {"path": "yolov8n.pt"}}
    model = load_detection_model(config)
    
    detections = detect_objects(frame, model)
    assert isinstance(detections, list)

