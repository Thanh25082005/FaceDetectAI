"""
Quality Filter Module

Filters low-quality face frames before FAS/FR processing.
Quality checks include: size, blur, pose, brightness, occlusion.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.append('..')
from config import (
    MIN_FACE_SIZE,
    BLUR_THRESHOLD,
    POSE_THRESHOLD,
    BRIGHTNESS_MIN,
    BRIGHTNESS_MAX
)


@dataclass
class QualityResult:
    """Result of quality assessment"""
    is_acceptable: bool
    overall_score: float  # 0-1, higher is better
    
    # Individual scores
    size_score: float
    blur_score: float
    brightness_score: float
    pose_score: float
    
    # Reasons for rejection
    rejection_reasons: list
    
    def to_dict(self) -> Dict:
        return {
            'is_acceptable': self.is_acceptable,
            'overall_score': self.overall_score,
            'size_score': self.size_score,
            'blur_score': self.blur_score,
            'brightness_score': self.brightness_score,
            'pose_score': self.pose_score,
            'rejection_reasons': self.rejection_reasons
        }


class QualityFilter:
    """
    Filter for assessing face image quality.
    
    Only high-quality frames are used for FAS and FR processing.
    """
    
    def __init__(
        self,
        min_face_size: int = None,
        blur_threshold: float = None,
        pose_threshold: float = None,
        brightness_min: int = None,
        brightness_max: int = None
    ):
        """
        Initialize quality filter.
        
        Args:
            min_face_size: Minimum face size in pixels
            blur_threshold: Laplacian variance threshold
            pose_threshold: Frontal pose ratio
            brightness_min: Minimum average brightness
            brightness_max: Maximum average brightness
        """
        self.min_face_size = min_face_size or MIN_FACE_SIZE
        self.blur_threshold = blur_threshold or BLUR_THRESHOLD
        self.pose_threshold = pose_threshold or POSE_THRESHOLD
        self.brightness_min = brightness_min or BRIGHTNESS_MIN
        self.brightness_max = brightness_max or BRIGHTNESS_MAX
    
    def check_quality(
        self,
        face_image: np.ndarray,
        bbox: list = None,
        landmarks: Dict = None
    ) -> QualityResult:
        """
        Assess quality of a face image.
        
        Args:
            face_image: Cropped face image in BGR format
            bbox: Bounding box [x1, y1, x2, y2] (optional)
            landmarks: Facial landmarks dict (optional)
        
        Returns:
            QualityResult with scores and acceptance status
        """
        rejection_reasons = []
        
        # Validate input
        if face_image is None or face_image.size == 0:
            return QualityResult(
                is_acceptable=False,
                overall_score=0.0,
                size_score=0.0,
                blur_score=0.0,
                brightness_score=0.0,
                pose_score=0.0,
                rejection_reasons=["Empty image"]
            )
        
        # 1. Check face size
        size_score = self._check_size(face_image, bbox)
        if size_score < 0.5:
            rejection_reasons.append(f"Face too small (score: {size_score:.2f})")
        
        # 2. Check blur
        blur_score = self._check_blur(face_image)
        if blur_score < 0.5:
            rejection_reasons.append(f"Image too blurry (score: {blur_score:.2f})")
        
        # 3. Check brightness
        brightness_score = self._check_brightness(face_image)
        if brightness_score < 0.5:
            rejection_reasons.append(f"Bad lighting (score: {brightness_score:.2f})")
        
        # 4. Check pose
        pose_score = self._check_pose(landmarks)
        if pose_score < 0.5:
            rejection_reasons.append(f"Non-frontal pose (score: {pose_score:.2f})")
        
        # Calculate overall score
        weights = {
            'size': 0.25,
            'blur': 0.30,
            'brightness': 0.20,
            'pose': 0.25
        }
        
        overall_score = (
            weights['size'] * size_score +
            weights['blur'] * blur_score +
            weights['brightness'] * brightness_score +
            weights['pose'] * pose_score
        )
        
        is_acceptable = len(rejection_reasons) == 0 and overall_score >= 0.5
        
        return QualityResult(
            is_acceptable=is_acceptable,
            overall_score=overall_score,
            size_score=size_score,
            blur_score=blur_score,
            brightness_score=brightness_score,
            pose_score=pose_score,
            rejection_reasons=rejection_reasons
        )
    
    def _check_size(self, face_image: np.ndarray, bbox: list = None) -> float:
        """
        Check if face is large enough.
        
        Returns:
            Score 0-1 (1 = good size)
        """
        h, w = face_image.shape[:2]
        min_dim = min(h, w)
        
        if min_dim < self.min_face_size * 0.5:
            return 0.0
        elif min_dim < self.min_face_size:
            return 0.5 * (min_dim / self.min_face_size)
        else:
            return min(1.0, min_dim / (self.min_face_size * 2))
    
    def _check_blur(self, face_image: np.ndarray) -> float:
        """
        Check if image is blurry using Laplacian variance.
        
        Returns:
            Score 0-1 (1 = sharp)
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < self.blur_threshold * 0.5:
            return 0.0
        elif laplacian_var < self.blur_threshold:
            return 0.5 * (laplacian_var / self.blur_threshold)
        else:
            return min(1.0, laplacian_var / (self.blur_threshold * 3))
    
    def _check_brightness(self, face_image: np.ndarray) -> float:
        """
        Check if lighting is appropriate.
        
        Returns:
            Score 0-1 (1 = good lighting)
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Too dark
        if mean_brightness < self.brightness_min:
            return max(0.0, mean_brightness / self.brightness_min)
        
        # Too bright
        if mean_brightness > self.brightness_max:
            return max(0.0, (255 - mean_brightness) / (255 - self.brightness_max))
        
        # Good range - score based on distance from optimal (128)
        optimal = (self.brightness_min + self.brightness_max) / 2
        distance_from_optimal = abs(mean_brightness - optimal)
        max_distance = (self.brightness_max - self.brightness_min) / 2
        
        return 1.0 - (distance_from_optimal / max_distance) * 0.3
    
    def _check_pose(self, landmarks: Dict = None) -> float:
        """
        Check if face is roughly frontal using landmarks.
        
        Returns:
            Score 0-1 (1 = frontal)
        """
        if landmarks is None:
            return 0.7  # Default score if no landmarks
        
        try:
            # Get eye positions
            left_eye = landmarks.get('left_eye')
            right_eye = landmarks.get('right_eye')
            nose = landmarks.get('nose')
            
            if not all([left_eye, right_eye, nose]):
                return 0.7
            
            # Calculate eye distance
            eye_dist = np.sqrt(
                (right_eye[0] - left_eye[0]) ** 2 +
                (right_eye[1] - left_eye[1]) ** 2
            )
            
            # Calculate nose position relative to eyes
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_offset = abs(nose[0] - eye_center_x)
            
            # Frontal if nose is centered between eyes
            if eye_dist > 0:
                offset_ratio = nose_offset / eye_dist
                frontal_score = max(0.0, 1.0 - offset_ratio * 2)
                return frontal_score
            
            return 0.7
            
        except (KeyError, TypeError, IndexError):
            return 0.7
    
    def filter_frames(
        self,
        frames: list,
        bboxes: list = None,
        landmarks_list: list = None
    ) -> list:
        """
        Filter a list of frames, keeping only quality ones.
        
        Args:
            frames: List of face images
            bboxes: Optional list of bounding boxes
            landmarks_list: Optional list of landmarks
        
        Returns:
            List of (frame, quality_result) tuples for acceptable frames
        """
        results = []
        
        for i, frame in enumerate(frames):
            bbox = bboxes[i] if bboxes and i < len(bboxes) else None
            landmarks = landmarks_list[i] if landmarks_list and i < len(landmarks_list) else None
            
            quality = self.check_quality(frame, bbox, landmarks)
            
            if quality.is_acceptable:
                results.append((frame, quality))
        
        return results


# Singleton instance
_quality_filter_instance: Optional[QualityFilter] = None


def get_quality_filter() -> QualityFilter:
    """Get or create QualityFilter singleton"""
    global _quality_filter_instance
    if _quality_filter_instance is None:
        _quality_filter_instance = QualityFilter()
    return _quality_filter_instance
