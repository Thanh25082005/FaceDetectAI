"""
Silent-Face Anti-Spoofing Module

Deep learning-based face anti-spoofing using MiniFASNetV2.
Uses official Silent-Face model architecture.
"""

import numpy as np
import cv2
from typing import Dict, Optional
from pathlib import Path
import torch
import torch.nn.functional as F
import sys

# Add libs to path for Silent-Face imports
LIBS_DIR = Path(__file__).parent.parent / "libs" / "silent_face" / "src"
sys.path.insert(0, str(LIBS_DIR))

from config import BASE_DIR

# Import from Silent-Face repo
try:
    from model_lib.MiniFASNet import MiniFASNetV2
    SILENT_FACE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Silent-Face models: {e}")
    SILENT_FACE_AVAILABLE = False


class SilentFace:
    """
    Silent-Face Anti-Spoofing detector using MiniFASNetV2
    
    Detects print attacks and screen replay attacks with ~99% accuracy.
    """
    
    INPUT_SIZE = (80, 80)
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        """
        Initialize Silent-Face detector
        
        Args:
            model_path: Path to .pth model file
            device: 'cuda' or 'cpu'
        """
        if model_path is None:
            model_path = str(BASE_DIR / "data" / "models" / "2.7_80x80_MiniFASNetV2.onnx")
        
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Load PyTorch model using original Silent-Face architecture"""
        if not SILENT_FACE_AVAILABLE:
            raise RuntimeError("Silent-Face model not available")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Create model with correct architecture
        # MiniFASNetV2: embedding_size=128, conv6_kernel=(5,5) for 80x80 input
        self.model = MiniFASNetV2(
            embedding_size=128,
            conv6_kernel=(5, 5),  # 5x5 for 80x80 input (output is 5x5)
            drop_p=0.2,
            num_classes=3,
            img_channel=3
        )
        
        # Load weights
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Handle 'module.' prefix from DataParallel training
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Silent-Face model loaded successfully on {self.device}")
    
    def preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for model input
        
        Args:
            face_image: Face image in BGR format (any size)
        
        Returns:
            Preprocessed tensor ready for inference
        """
        # Resize to model input size
        face_resized = cv2.resize(face_image, self.INPUT_SIZE)
        
        # Convert BGR to RGB and normalize to [0, 1]
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW and add batch dimension
        face_tensor = torch.from_numpy(face_normalized.transpose(2, 0, 1)).unsqueeze(0)
        
        return face_tensor.to(self.device)
    
    @torch.no_grad()
    def predict(self, face_image: np.ndarray) -> Dict:
        """
        Predict if face is real or spoofed
        
        Args:
            face_image: Cropped face image in BGR format
        
        Returns:
            Dictionary with:
                - is_real: bool
                - score: float (0-1, higher = more likely real)
                - label: str ("real" or "spoof")
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Validate input image
        if face_image is None or face_image.size == 0:
            return {
                'is_real': False,
                'score': 0.0,
                'label': 'spoof',
                'probabilities': {'fake_2d': 1.0, 'real': 0.0, 'fake_3d': 0.0}
            }
        
        # Preprocess
        input_tensor = self.preprocess(face_image)
        
        # Run inference
        output = self.model(input_tensor)
        
        # Apply softmax to get probabilities
        probs = F.softmax(output, dim=1)[0].cpu().numpy()
        
        # probs: [fake_2d, real, fake_3d]
        real_score = float(probs[1])
        is_real = real_score > 0.5
        
        return {
            'is_real': is_real,
            'score': real_score,
            'label': 'real' if is_real else 'spoof',
            'probabilities': {
                'fake_2d': float(probs[0]),
                'real': float(probs[1]),
                'fake_3d': float(probs[2]) if len(probs) > 2 else 0.0
            }
        }
    
    def predict_batch(self, face_images: list) -> list:
        """Predict on multiple face images"""
        results = []
        for face in face_images:
            result = self.predict(face)
            results.append(result)
        return results


# Singleton instance
_silent_face_instance: Optional[SilentFace] = None


def get_silent_face(device: str = 'cuda') -> SilentFace:
    """
    Get or create a SilentFace singleton instance
    """
    global _silent_face_instance
    if _silent_face_instance is None:
        _silent_face_instance = SilentFace(device=device)
    return _silent_face_instance
