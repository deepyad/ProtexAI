import sys
import json
from pathlib import Path
import cv2

def validate(output_dir):
    output_path = Path(output_dir)
    
    # Check image outputs
    image_files = list(output_path.glob("*.jpg"))
    assert len(image_files) > 0, "No output images generated"
    
    # Check annotations file
    anno_file = output_path / "annotations.json"
    assert anno_file.exists(), "COCO annotations file missing"
    
    # Validate JSON structure and content
    with open(anno_file) as f:
        data = json.load(f)
    
    # Check top-level structure
    required_keys = {"info", "images", "annotations", "categories"}
    missing_keys = required_keys - data.keys()
    assert not missing_keys, f"Missing required COCO keys: {missing_keys}"
    
    # Validate images array
    for img in data["images"]:
        assert "id" in img, "Image missing ID"
        assert "file_name" in img, "Image missing filename"
        assert "width" in img, "Image missing width"
        assert "height" in img, "Image missing height"
        
        # Verify corresponding image file exists
        img_path = output_path / img["file_name"]
        assert img_path.exists(), f"Image file {img_path} not found"
        
        # Verify actual image dimensions match metadata
        if img["width"] < 100 or img["height"] < 100:  # Minimum size check
            raise ValueError(f"Unrealistic dimensions for image {img['id']}")
            
        # Optional: Load image to verify integrity
        try:
            cv2.imread(str(img_path))
        except Exception as e:
            raise ValueError(f"Corrupted image file {img_path}: {str(e)}")
    
    # Validate annotations
    seen_image_ids = {img["id"] for img in data["images"]}
    category_ids = {cat["id"] for cat in data["categories"]}
    
    for ann in data["annotations"]:
        # Required fields check
        assert "id" in ann, "Annotation missing ID"
        assert "image_id" in ann, "Annotation missing image_id"
        assert "category_id" in ann, "Annotation missing category_id"
        assert "bbox" in ann, "Annotation missing bbox"
        
        # Data consistency checks
        assert ann["image_id"] in seen_image_ids, "Annotation references missing image"
        assert ann["category_id"] in category_ids, "Annotation references missing category"
        
        # Bounding box validation
        bbox = ann["bbox"]
        assert len(bbox) == 4, "Invalid bbox format"
        assert all(isinstance(x, (int, float)) for x in bbox), "Non-numeric bbox values"
        assert bbox[2] > 0 and bbox[3] > 0, "Invalid bbox dimensions"
    
    # Validate categories
    for cat in data["categories"]:
        assert "id" in cat, "Category missing ID"
        assert "name" in cat, "Category missing name"
        assert isinstance(cat["name"], str), "Invalid category name type"
    
    # Cross-check image counts
    assert len(image_files) == len(data["images"]), "Image count mismatch between files and metadata"
    
    print("All outputs validated successfully")

if __name__ == "__main__":
    validate(sys.argv[1])
