"""
Blink Detection Module

Uses MediaPipe Face Mesh to detect eye blinks via Eye Aspect Ratio (EAR).
Real faces blink naturally; photos and videos typically don't.

Compatible with MediaPipe 0.10+ Tasks API.
"""

import numpy as np
import cv2
from typing import Dict, Optional, List, Tuple
from collections import deque
from pathlib import Path

# MediaPipe Tasks API (0.10+)
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    from mediapipe import solutions
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    try:
        import mediapipe as mp
        MEDIAPIPE_AVAILABLE = True
    except ImportError:
        MEDIAPIPE_AVAILABLE = False
        mp = None
        print("Warning: mediapipe not installed. Blink detection will not work.")


class BlinkDetector:
    """
    Blink detector using MediaPipe Face Mesh and Eye Aspect Ratio (EAR)
    
    EAR formula: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Where p1-p6 are the 6 eye landmarks
    """
    
    # MediaPipe Face Mesh eye landmark indices (468 landmarks total)
    # Left eye landmarks (from subject's perspective)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    # Right eye landmarks  
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    # EAR threshold for blink detection
    EAR_THRESHOLD = 0.25
    # Minimum consecutive frames for a valid blink
    CONSEC_FRAMES = 2
    
    def __init__(self, ear_threshold: float = 0.25, consec_frames: int = 2):
        """
        Initialize blink detector
        
        Args:
            ear_threshold: EAR value below which eye is considered closed
            consec_frames: Minimum frames with low EAR to count as blink
        """
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        
        # Blink tracking state
        self.frame_counter = 0
        self.blink_count = 0
        self.ear_history = deque(maxlen=30)  # Keep last 30 EAR values
        
        # Initialize MediaPipe Face Landmarker (Tasks API)
        self.face_landmarker = None
        if MEDIAPIPE_AVAILABLE:
            self._init_face_landmarker()
    
    def _init_face_landmarker(self):
        """Initialize MediaPipe Face Landmarker using Tasks API"""
        try:
            # Try Tasks API first (MediaPipe 0.10+)
            model_path = self._get_model_path()
            if model_path and Path(model_path).exists():
                base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1
                )
                self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
                print("Blink detector initialized with MediaPipe Tasks API")
            else:
                print(f"Face landmarker model not found. Blink detection disabled.")
                self.face_landmarker = None
        except Exception as e:
            print(f"Failed to initialize MediaPipe Face Landmarker: {e}")
            self.face_landmarker = None
    
    def _get_model_path(self) -> Optional[str]:
        """Get path to face landmarker model"""
        # Check common locations
        possible_paths = [
            Path(__file__).parent.parent / "data" / "models" / "face_landmarker.task",
            Path.home() / ".mediapipe" / "face_landmarker.task",
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    @staticmethod
    def calculate_ear(eye_landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        
        Args:
            eye_landmarks: List of 6 (x, y) tuples for eye landmarks
        
        Returns:
            EAR value (0-0.5 typically)
        """
        # Compute euclidean distances
        # Vertical distances
        v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        
        # Horizontal distance
        h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        # EAR formula
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        
        return ear
    
    def get_eye_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """
        Extract eye landmarks from image using MediaPipe
        
        Args:
            image: Face image in BGR format
        
        Returns:
            Dictionary with left_eye, right_eye landmarks or None
        """
        if self.face_landmarker is None:
            return None
        
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Detect landmarks
            detection_result = self.face_landmarker.detect(mp_image)
            
            if not detection_result.face_landmarks:
                return None
            
            # Get first face landmarks
            face_landmarks = detection_result.face_landmarks[0]
            
            h, w = image.shape[:2]
            
            # Extract left eye landmarks
            left_eye = []
            for idx in self.LEFT_EYE:
                landmark = face_landmarks[idx]
                left_eye.append((landmark.x * w, landmark.y * h))
            
            # Extract right eye landmarks
            right_eye = []
            for idx in self.RIGHT_EYE:
                landmark = face_landmarks[idx]
                right_eye.append((landmark.x * w, landmark.y * h))
            
            return {
                'left_eye': left_eye,
                'right_eye': right_eye
            }
        except Exception as e:
            print(f"Eye landmark detection failed: {e}")
            return None
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect blink in a single frame
        
        Args:
            image: Face image in BGR format
        
        Returns:
            Dictionary with EAR values and blink detection status
        """
        result = {
            'ear_left': 0.0,
            'ear_right': 0.0,
            'ear_avg': 0.0,
            'eyes_closed': False,
            'blink_detected': False,
            'blink_count': self.blink_count,
            'available': self.face_landmarker is not None
        }
        
        if self.face_landmarker is None:
            return result
        
        # Get eye landmarks
        eyes = self.get_eye_landmarks(image)
        
        if eyes is None:
            return result
        
        # Calculate EAR for both eyes
        ear_left = self.calculate_ear(eyes['left_eye'])
        ear_right = self.calculate_ear(eyes['right_eye'])
        ear_avg = (ear_left + ear_right) / 2.0
        
        result['ear_left'] = float(ear_left)
        result['ear_right'] = float(ear_right)
        result['ear_avg'] = float(ear_avg)
        
        # Track EAR history
        self.ear_history.append(ear_avg)
        
        # Check if eyes are closed
        if ear_avg < self.ear_threshold:
            self.frame_counter += 1
            result['eyes_closed'] = True
        else:
            # If eyes were closed for enough frames, count as blink
            if self.frame_counter >= self.consec_frames:
                self.blink_count += 1
                result['blink_detected'] = True
            self.frame_counter = 0
        
        result['blink_count'] = self.blink_count
        
        return result
    
    def detect_from_frames(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze blink across multiple frames
        
        Args:
            frames: List of face images in BGR format
        
        Returns:
            Summary of blink detection across all frames
        """
        # Reset state
        self.frame_counter = 0
        self.blink_count = 0
        self.ear_history.clear()
        
        results = []
        for frame in frames:
            result = self.detect(frame)
            results.append(result)
        
        # Compute summary
        ear_values = [r['ear_avg'] for r in results if r['ear_avg'] > 0]
        
        summary = {
            'total_frames': len(frames),
            'blink_count': self.blink_count,
            'has_blink': self.blink_count > 0,
            'ear_mean': float(np.mean(ear_values)) if ear_values else 0.0,
            'ear_std': float(np.std(ear_values)) if ear_values else 0.0,
            'ear_min': float(np.min(ear_values)) if ear_values else 0.0,
            'is_live': self.blink_count > 0,  # At least one blink = likely live
            'available': self.face_landmarker is not None
        }
        
        return summary
    
    def reset(self):
        """Reset blink counter and frame counter"""
        self.frame_counter = 0
        self.blink_count = 0
        self.ear_history.clear()


# Singleton instance
_blink_detector_instance: Optional[BlinkDetector] = None


def get_blink_detector() -> BlinkDetector:
    """
    Get or create a BlinkDetector singleton instance
    """
    global _blink_detector_instance
    if _blink_detector_instance is None:
        _blink_detector_instance = BlinkDetector()
    return _blink_detector_instance
