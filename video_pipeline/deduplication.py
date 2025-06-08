from imagehash import phash
from PIL import Image
import cv2

def deduplicate_frames(frames, threshold=2):
    hashes = {}
    unique_frames = []
    
    for idx, frame in enumerate(frames):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        current_hash = phash(pil_image)
        
        # Check against existing hashes
        duplicate = False
        for stored_hash in hashes:
            if current_hash - stored_hash < threshold:
                duplicate = True
                break
                
        if not duplicate:
            hashes[current_hash] = idx
            unique_frames.append(frame)
            
    return unique_frames
