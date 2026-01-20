"""
Session Manager Module

Manages track lifecycle and orchestrates FAS + FR + Quality score fusion
for session-level check-in decisions.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

import sys
sys.path.append('..')
from config import (
    DECISION_CONFIDENCE_THRESHOLD,
    FR_THRESHOLD,
    FAS_ACCEPT_THRESHOLD,
    FAS_REJECT_THRESHOLD,
    FAS_ENABLE,
    FUSION_WEIGHT_FR,
    FUSION_WEIGHT_FAS,
    FUSION_WEIGHT_QUALITY,
    SESSION_TIMEOUT_SECONDS
)

from models.tracker import FaceTrack, TrackState
from models.face_recognizer import EmbeddingAggregator, get_face_recognizer
from models.quality_filter import QualityFilter, get_quality_filter
from models.database import get_face_database
from models.anti_spoofing import FASAggregator, get_fas_predictor


class CheckinDecision(Enum):
    """Final check-in decision"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED_SPOOF = "rejected_spoof"
    REJECTED_UNKNOWN = "rejected_unknown"
    REJECTED_LOW_QUALITY = "rejected_low_quality"
    TIMEOUT = "timeout"


@dataclass
class TrackSession:
    """
    Session state for a single track's check-in attempt.
    
    Aggregates FAS scores, FR embeddings, and quality scores
    to make a session-level decision.
    """
    track_id: str
    
    # Accumulators
    embedding_aggregator: EmbeddingAggregator = field(default_factory=EmbeddingAggregator)
    fas_aggregator: FASAggregator = field(default_factory=FASAggregator)
    
    # Quality tracking
    quality_scores: List[float] = field(default_factory=list)
    frames_processed: int = 0
    quality_frames_count: int = 0
    
    # Recognition results
    matched_user_id: Optional[str] = None
    match_similarity: float = 0.0
    top_candidates: List[Dict] = field(default_factory=list)
    
    # Decision state
    decision: CheckinDecision = CheckinDecision.PENDING
    decision_confidence: float = 0.0
    
    # Timing
    created_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    # Evidence
    best_frame: Optional[np.ndarray] = None
    best_frame_quality: float = 0.0
    
    def is_timeout(self) -> bool:
        """Check if session has timed out"""
        return time.time() - self.created_at > SESSION_TIMEOUT_SECONDS
    
    def add_quality_score(self, score: float):
        """Track quality scores"""
        self.quality_scores.append(score)
        self.last_update = time.time()
    
    def get_average_quality(self) -> float:
        """Get average quality score"""
        return np.mean(self.quality_scores) if self.quality_scores else 0.0
    
    def update_best_frame(self, frame: np.ndarray, quality: float):
        """Update best frame if current is better"""
        if quality > self.best_frame_quality:
            self.best_frame = frame.copy()
            self.best_frame_quality = quality
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        return {
            'track_id': self.track_id,
            'decision': self.decision.value,
            'decision_confidence': self.decision_confidence,
            'matched_user_id': self.matched_user_id,
            'match_similarity': self.match_similarity,
            'frames_processed': self.frames_processed,
            'quality_frames_count': self.quality_frames_count,
            'average_quality': self.get_average_quality(),
            'embedding_info': self.embedding_aggregator.to_dict(),
            'fas_info': self.fas_aggregator.to_dict(),
            'created_at': self.created_at,
            'session_duration': time.time() - self.created_at
        }


class SessionManager:
    """
    Orchestrates track processing and decision making.
    
    Coordinates:
    - Quality filtering
    - FAS score accumulation
    - FR embedding aggregation
    - Score fusion for final decision
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize session manager.
        
        Args:
            device: 'cuda' or 'cpu' for deep learning models
        """
        self.device = device
        self.sessions: Dict[str, TrackSession] = {}
        
        # Get model instances
        self.recognizer = get_face_recognizer(device=device)
        self.quality_filter = get_quality_filter()
        self.database = get_face_database()
        
        # Initialize FAS if enabled
        self.fas_enabled = FAS_ENABLE
        if self.fas_enabled:
            try:
                self.fas_predictor = get_fas_predictor(device=device)
                print("✅ FAS enabled in SessionManager")
            except Exception as e:
                print(f"⚠️ FAS initialization failed: {e}")
                print("   Continuing without FAS...")
                self.fas_enabled = False
    
    def get_or_create_session(self, track_id: str) -> TrackSession:
        """Get existing session or create new one"""
        if track_id not in self.sessions:
            self.sessions[track_id] = TrackSession(track_id=track_id)
        return self.sessions[track_id]
    
    def process_frame(
        self,
        track: FaceTrack,
        face_image: np.ndarray,
        aligned_face: np.ndarray = None
    ) -> TrackSession:
        """
        Process a frame for a track.
        
        Args:
            track: FaceTrack object
            face_image: Cropped face image
            aligned_face: Pre-aligned face (112x112) for FR
        
        Returns:
            Updated TrackSession
        """
        session = self.get_or_create_session(track.track_id)
        session.frames_processed += 1
        session.last_update = time.time()
        
        # Check timeout
        if session.is_timeout():
            session.decision = CheckinDecision.TIMEOUT
            return session
        
        # If already decided, return
        if session.decision != CheckinDecision.PENDING:
            return session
        
        # 1. Quality check
        quality = self.quality_filter.check_quality(
            face_image,
            track.bbox,
            track.landmarks
        )
        
        session.add_quality_score(quality.overall_score)
        
        if not quality.is_acceptable:
            return session  # Skip low-quality frame
        
        session.quality_frames_count += 1
        session.update_best_frame(face_image, quality.overall_score)
        
        # 2. FAS check
        if self.fas_enabled:
            fas_result = self.fas_predictor.predict(face_image)
            session.fas_aggregator.add_score(fas_result['score'])
            
            # Early reject if clearly fake
            if session.fas_aggregator.should_early_reject():
                session.decision = CheckinDecision.REJECTED_SPOOF
                session.decision_confidence = 1.0 - session.fas_aggregator.get_aggregated_score()
                return session
        
        # 3. FR embedding extraction

        
        # 3. FR embedding extraction
        use_face = aligned_face if aligned_face is not None else face_image
        embedding = self.recognizer.get_embedding_direct(use_face)
        
        if embedding is not None:
            session.embedding_aggregator.add_embedding(embedding)
            
            # Try recognition if enough embeddings
            if session.embedding_aggregator.can_recognize():
                self._try_recognize(session)
        
        # 4. Check if we can make a decision
        self._evaluate_decision(session)
        
        return session
    
    def _try_recognize(self, session: TrackSession):
        """Attempt face recognition using aggregated embedding"""
        aggregated = session.embedding_aggregator.get_aggregated()
        if aggregated is None:
            return
        
        # Get database embeddings
        db_embeddings = self.database.get_all_embeddings()
        if not db_embeddings:
            return
        
        # Find best match
        result = self.recognizer.recognize(
            aggregated,
            db_embeddings,
            threshold=FR_THRESHOLD
        )
        
        if result:
            session.matched_user_id = result.get('user_id')
            session.match_similarity = result.get('similarity', 0.0)
            session.embedding_aggregator.add_match_result(session.match_similarity)
            
            # Store top candidates
            session.top_candidates = result.get('candidates', [])[:5]
    
    def _evaluate_decision(self, session: TrackSession):
        """Evaluate if we can make a final decision"""
        emb_acc = session.embedding_aggregator
        
        # Need minimum data
        if not emb_acc.can_recognize():
            return
        
        # Check FR result
        if session.matched_user_id:
            # Fuse scores: FR + FAS + Quality
            if self.fas_enabled and session.fas_aggregator.can_decide():
                # Full fusion with FAS
                combined_confidence = (
                    FUSION_WEIGHT_FR * session.match_similarity +
                    FUSION_WEIGHT_FAS * session.fas_aggregator.get_aggregated_score() +
                    FUSION_WEIGHT_QUALITY * session.get_average_quality()
                )
                
                # FAS gate: reject if spoof detected
                if session.fas_aggregator.is_likely_spoof():
                    session.decision = CheckinDecision.REJECTED_SPOOF
                    session.decision_confidence = 1.0 - session.fas_aggregator.get_aggregated_score()
                    return
            else:
                # Fallback: FR + Quality only (original formula)
                combined_confidence = (
                    0.9 * session.match_similarity +
                    0.1 * session.get_average_quality()
                )
            
            if combined_confidence >= DECISION_CONFIDENCE_THRESHOLD:
                session.decision = CheckinDecision.ACCEPTED
                session.decision_confidence = combined_confidence
                return
        
        # No match found but stable embeddings
        if session.matched_user_id is None:
            if emb_acc.get_stability() > 0.8:
                session.decision = CheckinDecision.REJECTED_UNKNOWN
                session.decision_confidence = emb_acc.get_stability()

    
    def finalize_session(self, track_id: str) -> Optional[TrackSession]:
        """
        Finalize a session when track exits.
        
        Returns:
            Final session state
        """
        session = self.sessions.get(track_id)
        if session is None:
            return None
        
        # Force decision if still pending
        if session.decision == CheckinDecision.PENDING:
            if session.quality_frames_count < 3:
                session.decision = CheckinDecision.REJECTED_LOW_QUALITY
            elif session.matched_user_id:
                session.decision = CheckinDecision.ACCEPTED
            else:
                session.decision = CheckinDecision.REJECTED_UNKNOWN

        
        return session
    
    def remove_session(self, track_id: str):
        """Remove a session"""
        self.sessions.pop(track_id, None)
    
    def get_session(self, track_id: str) -> Optional[TrackSession]:
        """Get session by track ID"""
        return self.sessions.get(track_id)
    
    def get_all_sessions(self) -> List[TrackSession]:
        """Get all active sessions"""
        return list(self.sessions.values())
    
    def cleanup_old_sessions(self, max_age: float = 60.0):
        """Remove sessions older than max_age seconds"""
        now = time.time()
        to_remove = [
            tid for tid, session in self.sessions.items()
            if now - session.last_update > max_age
        ]
        for tid in to_remove:
            self.remove_session(tid)


# Singleton instance
_session_manager_instance: Optional[SessionManager] = None


def get_session_manager(device: str = None) -> SessionManager:
    """
    Get singleton instance of SessionManager
    """
    global _session_manager_instance
    if device is None:
        from config import DEVICE
        device = DEVICE
    if _session_manager_instance is None:
        _session_manager_instance = SessionManager(device=device)
    return _session_manager_instance
