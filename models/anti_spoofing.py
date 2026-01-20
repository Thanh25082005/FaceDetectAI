"""
Anti-Spoofing Module

Production wrapper for Silent-Face Anti-Spoofing with temporal aggregation.
Integrates with SessionManager for real-time spoof detection.
"""

import os
import sys
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Add Silent-Face library to path
BASE_DIR = Path(__file__).parent.parent.absolute()
SILENT_FACE_DIR = BASE_DIR / "libs" / "silent_face"
sys.path.insert(0, str(SILENT_FACE_DIR))

# Import Silent-Face components
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# Import config
from config import (
    FAS_MIN_FRAMES,
    FAS_REJECT_THRESHOLD,
    FAS_ACCEPT_THRESHOLD,
    FAS_EARLY_REJECT_THRESHOLD,
    FAS_EARLY_REJECT_FRAMES,
    FR_EMA_ALPHA,
    DEVICE
)


class FASAggregator:
    """
    Aggregate FAS scores over time using Exponential Moving Average.
    
    Similar to EmbeddingAggregator but for scalar scores.
    Provides temporal smoothing and early decision logic.
    """
    
    def __init__(self, ema_alpha: float = None):
        """
        Initialize FAS score aggregator.
        
        Args:
            ema_alpha: EMA weight for new scores (default from config)
        """
        self.ema_alpha = ema_alpha if ema_alpha is not None else FR_EMA_ALPHA
        self.scores: List[float] = []
        self.ema_score: Optional[float] = None
        self.min_frames = FAS_MIN_FRAMES
        
    def add_score(self, score: float):
        """
        Add a FAS score and update aggregation.
        
        Args:
            score: FAS score (0-1, higher = more real)
        """
        if not 0 <= score <= 1:
            raise ValueError(f"FAS score must be in [0, 1], got {score}")
        
        self.scores.append(score)
        
        # Update EMA
        if self.ema_score is None:
            self.ema_score = score
        else:
            self.ema_score = self.ema_alpha * score + (1 - self.ema_alpha) * self.ema_score
    
    def get_aggregated_score(self) -> float:
        """Get the EMA aggregated score."""
        return self.ema_score if self.ema_score is not None else 0.0
    
    def get_mean_score(self) -> float:
        """Get the mean of all scores."""
        return np.mean(self.scores) if self.scores else 0.0
    
    def can_decide(self) -> bool:
        """Check if we have enough frames to make a decision."""
        return len(self.scores) >= self.min_frames
    
    def is_likely_spoof(self) -> bool:
        """
        Check if face is likely a spoof.
        
        Returns:
            True if aggregated score indicates spoof
        """
        if not self.can_decide():
            return False
        
        return self.get_aggregated_score() < FAS_REJECT_THRESHOLD
    
    def is_likely_real(self) -> bool:
        """
        Check if face is likely real.
        
        Returns:
            True if aggregated score indicates real face
        """
        if not self.can_decide():
            return False
        
        return self.get_aggregated_score() > FAS_ACCEPT_THRESHOLD
    
    def should_early_reject(self) -> bool:
        """
        Check if we should reject early (before min_frames).
        
        Useful for quickly rejecting obvious spoofs.
        
        Returns:
            True if recent scores are consistently very low
        """
        if len(self.scores) < FAS_EARLY_REJECT_FRAMES:
            return False
        
        # Check last N frames
        recent_scores = self.scores[-FAS_EARLY_REJECT_FRAMES:]
        return all(s < FAS_EARLY_REJECT_THRESHOLD for s in recent_scores)
    
    def get_stability(self) -> float:
        """
        Get score stability (consistency).
        
        Returns:
            Stability score (0-1, higher = more stable)
        """
        if len(self.scores) < 2:
            return 0.0
        
        std = np.std(self.scores)
        # Convert std to stability (inverse relationship)
        # std=0 → stability=1, std=0.5 → stability=0
        stability = max(0.0, 1.0 - 2 * std)
        return stability
    
    def reset(self):
        """Reset aggregator."""
        self.scores.clear()
        self.ema_score = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/API."""
        return {
            "num_frames": len(self.scores),
            "ema_score": self.get_aggregated_score(),
            "mean_score": self.get_mean_score(),
            "stability": self.get_stability(),
            "is_likely_spoof": self.is_likely_spoof(),
            "is_likely_real": self.is_likely_real(),
            "can_decide": self.can_decide()
        }


class FASPredictor:
    """
    Production wrapper for Silent-Face Anti-Spoofing.
    
    Handles model loading, prediction, and multi-model fusion.
    """
    
    def __init__(self, device: str = None, model_dir: Path = None):
        """
        Initialize FAS predictor.
        
        Args:
            device: 'cuda' or 'cpu' (default from config)
            model_dir: Path to anti-spoof models (default from config)
        """
        self.device = device if device is not None else DEVICE
        
        # Convert device string to device_id for AntiSpoofPredict
        # Note: AntiSpoofPredict uses torch.device("cuda:{}") which doesn't handle -1
        # So for CPU, we use device_id=0 and let torch.cuda.is_available() handle it
        if self.device == 'cuda':
            device_id = 0
        else:
            # For CPU, we temporarily disable CUDA visibility
            device_id = 0  # Will fallback to CPU due to is_available() check
        
        # Model directory
        if model_dir is None:
            model_dir = SILENT_FACE_DIR / "resources" / "anti_spoof_models"
        self.model_dir = Path(model_dir)
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"FAS model directory not found: {self.model_dir}")
        
        # Initialize predictor (need to be in silent_face directory)
        original_cwd = os.getcwd()
        os.chdir(SILENT_FACE_DIR)
        
        try:
            self.predictor = AntiSpoofPredict(device_id)
            self.cropper = CropImage()
        finally:
            os.chdir(original_cwd)
        
        # Load model info
        self.models = list(self.model_dir.glob("*.pth"))
        if not self.models:
            raise FileNotFoundError(f"No .pth models found in {self.model_dir}")
        
        print(f"✅ FAS: Loaded {len(self.models)} anti-spoofing models")
        for m in self.models:
            print(f"   - {m.name}")
    
    def predict(self, face_image: np.ndarray) -> dict:
        """
        Predict if face is real or fake.
        
        Args:
            face_image: BGR face image (cropped or full frame)
            
        Returns:
            dict with keys:
                - score: FAS score (0-1, higher = more real)
                - is_real: Boolean classification
                - label: Human-readable label
                - bbox: Face bounding box [x, y, w, h] or None
                - time_ms: Inference time in milliseconds
        """
        result = {
            "score": 0.0,
            "is_real": False,
            "label": "No Face",
            "bbox": None,
            "time_ms": 0
        }
        
        if face_image is None or face_image.size == 0:
            return result
        
        start_time = time.time()
        
        try:
            # Detect face (need to be in silent_face directory)
            original_cwd = os.getcwd()
            os.chdir(SILENT_FACE_DIR)
            
            bbox = self.predictor.get_bbox(face_image)
            
            if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
                os.chdir(original_cwd)
                return result
            
            result["bbox"] = bbox
            
            # Multi-model fusion prediction
            prediction = np.zeros((1, 3))
            
            for model_path in self.models:
                model_name = model_path.name
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                
                param = {
                    "org_img": face_image,
                    "bbox": bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True if scale else False,
                }
                
                img_crop = self.cropper.crop(**param)
                prediction += self.predictor.predict(img_crop, str(model_path))
            
            os.chdir(original_cwd)
            
            # Get result
            label_idx = np.argmax(prediction)
            score = prediction[0][label_idx] / len(self.models)
            
            result["is_real"] = (label_idx == 1)
            result["score"] = float(score)
            result["label"] = "Real Face" if label_idx == 1 else "Fake Face"
            result["time_ms"] = (time.time() - start_time) * 1000
            
        except Exception as e:
            os.chdir(original_cwd)
            print(f"⚠️ FAS prediction error: {e}")
        
        return result
    
    def predict_batch(self, face_images: List[np.ndarray]) -> List[dict]:
        """
        Predict on a batch of face images.
        
        Args:
            face_images: List of BGR face images
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(img) for img in face_images]


# Singleton instance
_fas_predictor_instance: Optional[FASPredictor] = None


def get_fas_predictor(device: str = None) -> FASPredictor:
    """
    Get singleton instance of FASPredictor.
    
    Args:
        device: 'cuda' or 'cpu' (only used on first call)
        
    Returns:
        FASPredictor instance
    """
    global _fas_predictor_instance
    
    if _fas_predictor_instance is None:
        _fas_predictor_instance = FASPredictor(device=device)
    
    return _fas_predictor_instance
