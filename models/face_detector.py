import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
import cv2

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: facenet-pytorch not installed. Face detection will not work.")

import sys
sys.path.append('..')
from config import FACE_DETECTION_CONFIDENCE


class FaceDetector:
    """
    Face detector using MTCNN from facenet-pytorch
    """
    
    def __init__(self, device: str = 'cpu', min_face_size: int = 20):
        """
        Initialize MTCNN face detector
        
        Args:
            device: 'cpu' or 'cuda' for GPU acceleration
            min_face_size: Minimum face size to detect in pixels
        """
        self.device = device
        self.min_face_size = min_face_size
        
        if MTCNN_AVAILABLE:
            self.detector = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=min_face_size,
                thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
                factor=0.709,
                post_process=True,
                device=device,
                keep_all=True  # Detect all faces, not just the largest
            )
        else:
            self.detector = None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image
        
        Args:
            image: Input image in BGR format (OpenCV)
        
        Returns:
            List of dictionaries containing:
                - box: [x1, y1, x2, y2] bounding box coordinates
                - confidence: Detection confidence score
                - landmarks: Facial landmarks (eyes, nose, mouth corners)
        """
        if self.detector is None:
            raise RuntimeError("MTCNN not available. Please install facenet-pytorch.")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Detect faces
        boxes, probs, landmarks = self.detector.detect(pil_image, landmarks=True)
        
        results = []
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob >= FACE_DETECTION_CONFIDENCE:
                    face_info = {
                        'box': [int(coord) for coord in box],  # [x1, y1, x2, y2]
                        'confidence': float(prob)
                    }
                    
                    # Add landmarks if available
                    if landmarks is not None and i < len(landmarks):
                        landmark = landmarks[i]
                        face_info['landmarks'] = {
                            'left_eye': [float(landmark[0][0]), float(landmark[0][1])],
                            'right_eye': [float(landmark[1][0]), float(landmark[1][1])],
                            'nose': [float(landmark[2][0]), float(landmark[2][1])],
                            'mouth_left': [float(landmark[3][0]), float(landmark[3][1])],
                            'mouth_right': [float(landmark[4][0]), float(landmark[4][1])]
                        }
                    
                    results.append(face_info)
        
        return results
    
    def extract_faces(self, image: np.ndarray, margin: float = 0.2) -> List[Tuple[np.ndarray, Dict]]:
        """
        Detect and extract face images from the input
        
        Args:
            image: Input image in BGR format
            margin: Margin ratio to add around detected face
        
        Returns:
            List of tuples (cropped_face_image, face_info)
        """
        detections = self.detect_faces(image)
        faces = []
        
        height, width = image.shape[:2]
        
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            
            # Add margin
            face_width = x2 - x1
            face_height = y2 - y1
            margin_x = int(face_width * margin)
            margin_y = int(face_height * margin)
            
            # Calculate new coordinates with margin and bounds checking
            x1_new = max(0, x1 - margin_x)
            y1_new = max(0, y1 - margin_y)
            x2_new = min(width, x2 + margin_x)
            y2_new = min(height, y2 + margin_y)
            
            # Crop face
            face_image = image[y1_new:y2_new, x1_new:x2_new]
            
            # Update detection with new box coordinates
            detection['box'] = [x1_new, y1_new, x2_new, y2_new]
            
            faces.append((face_image, detection))
        
        return faces
    
    def get_largest_face(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Get the largest detected face in the image
        
        Args:
            image: Input image in BGR format
        
        Returns:
            Tuple of (face_image, face_info) or None if no face detected
        """
        faces = self.extract_faces(image)
        
        if not faces:
            return None
        
        # Find the largest face by area
        largest_face = max(faces, key=lambda x: (x[1]['box'][2] - x[1]['box'][0]) * 
                                                  (x[1]['box'][3] - x[1]['box'][1]))
        
        return largest_face


# Singleton instance for reuse
_detector_instance: Optional[FaceDetector] = None


def get_face_detector(device: str = 'cpu') -> FaceDetector:
    """
    Get or create a FaceDetector singleton instance
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector(device=device)
    return _detector_instance
