import io
import base64
from typing import Tuple, Optional, List
import numpy as np
from PIL import Image
import cv2


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes to numpy array (BGR format for OpenCV)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def load_image_from_base64(base64_string: str) -> np.ndarray:
    """
    Load image from base64 string
    """
    # Remove header if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    image_bytes = base64.b64decode(base64_string)
    return load_image_from_bytes(image_bytes)


def image_to_base64(image: np.ndarray, format: str = "jpeg") -> str:
    """
    Convert numpy array image to base64 string
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    pil_image = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format.upper())
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_image(image: np.ndarray, max_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to fit within max_size while maintaining aspect ratio
    """
    height, width = image.shape[:2]
    max_width, max_height = max_size
    
    if width <= max_width and height <= max_height:
        return image
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def crop_face(image: np.ndarray, bbox: List[int], margin: float = 0.2) -> np.ndarray:
    """
    Crop face from image with optional margin
    
    Args:
        image: Input image (BGR format)
        bbox: Bounding box [x1, y1, x2, y2]
        margin: Margin ratio to add around face
    
    Returns:
        Cropped face image
    """
    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    
    # Add margin
    face_width = x2 - x1
    face_height = y2 - y1
    margin_x = int(face_width * margin)
    margin_y = int(face_height * margin)
    
    # Calculate new coordinates with margin
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(width, x2 + margin_x)
    y2 = min(height, y2 + margin_y)
    
    return image[y1:y2, x1:x2]


def preprocess_face(face_image: np.ndarray, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Preprocess face image for recognition model
    
    Args:
        face_image: Cropped face image
        target_size: Target size for the model (default 112x112 for InsightFace)
    
    Returns:
        Preprocessed face image
    """
    # Resize to target size
    face = cv2.resize(face_image, target_size, interpolation=cv2.INTER_LINEAR)
    return face


def draw_bounding_boxes(image: np.ndarray, bboxes: List[dict], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image
        bboxes: List of bounding box dictionaries with 'box' key
        color: BGR color for the boxes
    
    Returns:
        Image with drawn bounding boxes
    """
    result = image.copy()
    for bbox_info in bboxes:
        x1, y1, x2, y2 = bbox_info['box']
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw confidence if available
        if 'confidence' in bbox_info:
            label = f"{bbox_info['confidence']:.2f}"
            cv2.putText(result, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result
