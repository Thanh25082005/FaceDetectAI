"""
Face Tracker Module

IOU-based multi-object tracker for linking face detections across frames.
Each track represents one person's journey through the camera view.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

import sys
sys.path.append('..')
from config import TRACK_MAX_AGE, TRACK_MIN_HITS, TRACK_IOU_THRESHOLD


class TrackState(Enum):
    """Track lifecycle states"""
    NEW = "new"           # Just created, not yet stable
    STABLE = "stable"     # Consistently detected, ready for processing
    CONFIRMED = "confirmed"  # Identity confirmed, check-in in progress
    EXIT = "exit"         # Lost tracking, cleaning up


@dataclass
class FaceTrack:
    """
    Represents a single person's track through frames.
    
    Lifecycle: NEW → STABLE → CONFIRMED → EXIT
    """
    track_id: str
    state: TrackState = TrackState.NEW
    
    # Bounding box history
    bbox: List[int] = field(default_factory=list)  # Current [x1, y1, x2, y2]
    bbox_history: List[List[int]] = field(default_factory=list)
    
    # Detection info
    confidence: float = 0.0
    landmarks: Optional[Dict] = None
    
    # Frame history for FAS/FR processing
    quality_frames: List[np.ndarray] = field(default_factory=list)
    
    # FAS accumulation
    fas_scores: List[float] = field(default_factory=list)
    fas_decision: Optional[bool] = None  # None=pending, True=real, False=spoof
    
    # FR accumulation
    embeddings: List[np.ndarray] = field(default_factory=list)
    aggregated_embedding: Optional[np.ndarray] = None
    matched_user_id: Optional[str] = None
    match_similarity: float = 0.0
    
    # Timing
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    # Counters
    hits: int = 0  # Consecutive detections
    age: int = 0   # Frames since last detection
    total_frames: int = 0
    
    def update_bbox(self, bbox: List[int], confidence: float, landmarks: Dict = None):
        """Update with new detection"""
        self.bbox = bbox
        self.bbox_history.append(bbox)
        self.confidence = confidence
        self.landmarks = landmarks
        self.last_seen = time.time()
        self.hits += 1
        self.age = 0
        self.total_frames += 1
        
        # State transition: NEW → STABLE
        if self.state == TrackState.NEW and self.hits >= TRACK_MIN_HITS:
            self.state = TrackState.STABLE
    
    def mark_missed(self):
        """Called when track not matched in current frame"""
        self.age += 1
        self.hits = 0
        
        # State transition to EXIT if too old
        if self.age > TRACK_MAX_AGE:
            self.state = TrackState.EXIT
    
    def add_quality_frame(self, frame: np.ndarray):
        """Add a quality-filtered frame for processing"""
        self.quality_frames.append(frame)
        # Keep max 30 frames
        if len(self.quality_frames) > 30:
            self.quality_frames.pop(0)
    
    def add_fas_score(self, score: float):
        """Add FAS score from a frame"""
        self.fas_scores.append(score)
    
    def add_embedding(self, embedding: np.ndarray, ema_alpha: float = 0.7):
        """Add embedding and update aggregation using EMA"""
        self.embeddings.append(embedding)
        
        if self.aggregated_embedding is None:
            self.aggregated_embedding = embedding.copy()
        else:
            # Exponential Moving Average
            self.aggregated_embedding = (
                ema_alpha * embedding + 
                (1 - ema_alpha) * self.aggregated_embedding
            )
            # Normalize
            self.aggregated_embedding /= np.linalg.norm(self.aggregated_embedding)
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of current bbox"""
        if not self.bbox:
            return (0, 0)
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_area(self) -> float:
        """Get area of current bbox"""
        if not self.bbox:
            return 0
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        return {
            'track_id': self.track_id,
            'state': self.state.value,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'hits': self.hits,
            'age': self.age,
            'total_frames': self.total_frames,
            'fas_scores_count': len(self.fas_scores),
            'embeddings_count': len(self.embeddings),
            'fas_decision': self.fas_decision,
            'matched_user_id': self.matched_user_id,
            'match_similarity': self.match_similarity,
            'created_at': self.created_at,
            'last_seen': self.last_seen
        }


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """
    Compute Intersection over Union between two bboxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        IOU value (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    
    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


class FaceTracker:
    """
    Multi-object face tracker using IOU matching.
    
    Links detections across frames to maintain person identity.
    """
    
    def __init__(self, iou_threshold: float = None, max_age: int = None):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IOU for matching
            max_age: Frames before track is removed
        """
        self.iou_threshold = iou_threshold or TRACK_IOU_THRESHOLD
        self.max_age = max_age or TRACK_MAX_AGE
        
        self.tracks: Dict[str, FaceTrack] = {}
        self.frame_count = 0
    
    def update(self, detections: List[Dict], frame: np.ndarray = None) -> List[FaceTrack]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with 'box', 'confidence', 'landmarks'
            frame: Current frame (optional, for storing quality frames)
        
        Returns:
            List of active FaceTrack objects
        """
        self.frame_count += 1
        
        # Get current track bboxes
        track_ids = list(self.tracks.keys())
        
        if not detections:
            # No detections - mark all tracks as missed
            for track_id in track_ids:
                self.tracks[track_id].mark_missed()
            self._cleanup_tracks()
            return self.get_active_tracks()
        
        if not track_ids:
            # No existing tracks - create new ones
            for det in detections:
                self._create_track(det)
            return self.get_active_tracks()
        
        # Compute IOU matrix
        det_boxes = [det['box'] for det in detections]
        track_boxes = [self.tracks[tid].bbox for tid in track_ids]
        
        iou_matrix = np.zeros((len(det_boxes), len(track_boxes)))
        for i, det_box in enumerate(det_boxes):
            for j, track_box in enumerate(track_boxes):
                iou_matrix[i, j] = compute_iou(det_box, track_box)
        
        # Greedy matching
        matched_dets = set()
        matched_tracks = set()
        
        while True:
            if iou_matrix.size == 0:
                break
            
            # Find best match
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break
            
            max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            det_idx, track_idx = max_idx
            
            # Update matched track
            track_id = track_ids[track_idx]
            det = detections[det_idx]
            self.tracks[track_id].update_bbox(
                det['box'],
                det['confidence'],
                det.get('landmarks')
            )
            
            matched_dets.add(det_idx)
            matched_tracks.add(track_idx)
            
            # Remove matched from matrix
            iou_matrix[det_idx, :] = -1
            iou_matrix[:, track_idx] = -1
        
        # Handle unmatched detections - create new tracks
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._create_track(det)
        
        # Handle unmatched tracks - mark as missed
        for j, track_id in enumerate(track_ids):
            if j not in matched_tracks:
                self.tracks[track_id].mark_missed()
        
        self._cleanup_tracks()
        return self.get_active_tracks()
    
    def _create_track(self, detection: Dict) -> FaceTrack:
        """Create new track from detection"""
        track_id = str(uuid.uuid4())[:8]
        track = FaceTrack(track_id=track_id)
        track.update_bbox(
            detection['box'],
            detection['confidence'],
            detection.get('landmarks')
        )
        self.tracks[track_id] = track
        return track
    
    def _cleanup_tracks(self):
        """Remove tracks in EXIT state"""
        to_remove = [
            tid for tid, track in self.tracks.items()
            if track.state == TrackState.EXIT
        ]
        for tid in to_remove:
            del self.tracks[tid]
    
    def get_active_tracks(self) -> List[FaceTrack]:
        """Get all non-EXIT tracks"""
        return [
            track for track in self.tracks.values()
            if track.state != TrackState.EXIT
        ]
    
    def get_stable_tracks(self) -> List[FaceTrack]:
        """Get tracks that are STABLE or CONFIRMED"""
        return [
            track for track in self.tracks.values()
            if track.state in [TrackState.STABLE, TrackState.CONFIRMED]
        ]
    
    def get_track(self, track_id: str) -> Optional[FaceTrack]:
        """Get track by ID"""
        return self.tracks.get(track_id)
    
    def reset(self):
        """Clear all tracks"""
        self.tracks.clear()
        self.frame_count = 0


# Singleton instance
_tracker_instance: Optional[FaceTracker] = None


def get_face_tracker() -> FaceTracker:
    """Get or create FaceTracker singleton"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = FaceTracker()
    return _tracker_instance
