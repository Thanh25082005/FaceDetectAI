"""
Stream Processor Module

Real-time frame processing with queue management and drop strategy.
Orchestrates the complete pipeline: Detection → Tracking → Quality → FAS → FR → Decision
"""

import asyncio
import numpy as np
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

import sys
sys.path.append('..')
from config import STREAM_FPS, MAX_QUEUE_SIZE

from models.face_detector import get_face_detector
from models.tracker import get_face_tracker, FaceTrack, TrackState
from models.quality_filter import get_quality_filter
from models.session_manager import get_session_manager, CheckinDecision, TrackSession
from models.checkin_logger import get_checkin_logger


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for a frame and metadata"""
    frame: np.ndarray
    camera_id: str
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0


@dataclass 
class CheckinEvent:
    """Event emitted when check-in decision is made"""
    track_id: str
    decision: CheckinDecision
    user_id: Optional[str]
    confidence: float
    camera_id: str
    timestamp: float
    session: TrackSession
    
    def to_dict(self) -> Dict:
        return {
            'track_id': self.track_id,
            'decision': self.decision.value,
            'user_id': self.user_id,
            'confidence': self.confidence,
            'camera_id': self.camera_id,
            'timestamp': self.timestamp,
            'session': self.session.to_dict()
        }


class StreamProcessor:
    """
    Async frame processor with queue management.
    
    Features:
    - Queue-based frame input with drop strategy
    - Full pipeline processing (detect → track → quality → FAS → FR)
    - Event emission on check-in decisions
    - Multi-camera support
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        max_queue_size: int = None,
        on_checkin: Callable[[CheckinEvent], None] = None
    ):
        """
        Initialize stream processor.
        
        Args:
            device: 'cuda' or 'cpu'
            max_queue_size: Max frames in queue before dropping
            on_checkin: Callback for check-in events
        """
        self.device = device
        self.max_queue_size = max_queue_size or MAX_QUEUE_SIZE
        self.on_checkin = on_checkin
        
        # Frame queue
        self.frame_queue: asyncio.Queue = None
        
        # Processing state
        self.is_running = False
        self.frame_count = 0
        self.process_count = 0
        self.drop_count = 0
        
        # Event history
        self.events: List[CheckinEvent] = []
        self.max_events = 100
        
        # Components (lazy init)
        self._detector = None
        self._tracker = None
        self._quality_filter = None
        self._session_manager = None
        self._checkin_logger = None
    
    @property
    def detector(self):
        if self._detector is None:
            self._detector = get_face_detector(device=self.device)
        return self._detector
    
    @property
    def tracker(self):
        if self._tracker is None:
            self._tracker = get_face_tracker()
        return self._tracker
    
    @property
    def quality_filter(self):
        if self._quality_filter is None:
            self._quality_filter = get_quality_filter()
        return self._quality_filter
    
    @property
    def session_manager(self):
        if self._session_manager is None:
            self._session_manager = get_session_manager(device=self.device)
        return self._session_manager
    
    @property
    def checkin_logger(self):
        if self._checkin_logger is None:
            self._checkin_logger = get_checkin_logger()
        return self._checkin_logger
    
    async def start(self):
        """Start the processor"""
        if self.is_running:
            return
        
        self.frame_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.is_running = True
        logger.info("StreamProcessor started")
    
    async def stop(self):
        """Stop the processor"""
        self.is_running = False
        logger.info("StreamProcessor stopped")
    
    async def add_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "default"
    ) -> bool:
        """
        Add frame to processing queue.
        
        Drops old frames if queue is full (low latency priority).
        
        Args:
            frame: BGR image
            camera_id: Camera identifier
        
        Returns:
            True if added, False if dropped
        """
        if not self.is_running or self.frame_queue is None:
            return False
        
        self.frame_count += 1
        
        frame_data = FrameData(
            frame=frame,
            camera_id=camera_id,
            frame_id=self.frame_count
        )
        
        # Drop old frames if queue full
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
                self.drop_count += 1
            except asyncio.QueueEmpty:
                pass
        
        try:
            self.frame_queue.put_nowait(frame_data)
            return True
        except asyncio.QueueFull:
            self.drop_count += 1
            return False
    
    async def process_loop(self):
        """Main processing loop - run as async task"""
        await self.start()
        
        while self.is_running:
            try:
                # Get frame from queue with timeout
                try:
                    frame_data = await asyncio.wait_for(
                        self.frame_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process frame
                events = await self.process_frame(frame_data)
                
                # Emit events
                for event in events:
                    self.events.append(event)
                    if len(self.events) > self.max_events:
                        self.events.pop(0)
                    
                    if self.on_checkin:
                        self.on_checkin(event)
                
                self.process_count += 1
                
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
    
    async def process_frame(self, frame_data: FrameData) -> List[CheckinEvent]:
        """
        Process a single frame through the pipeline.
        
        Returns:
            List of CheckinEvents (decisions made this frame)
        """
        frame = frame_data.frame
        camera_id = frame_data.camera_id
        events = []
        
        # 1. Face Detection
        detections = self.detector.detect_faces(frame)
        
        if not detections:
            # No faces - update tracker for missed frames
            self.tracker.update([], frame)
            return events
        
        # 2. Update Tracker
        tracks = self.tracker.update(detections, frame)
        
        # 3. Process each stable track
        for track in tracks:
            if track.state not in [TrackState.STABLE, TrackState.CONFIRMED]:
                continue
            
            # Get aligned face for this track
            aligned_faces = self.detector.extract_aligned_faces(frame)
            
            # Find aligned face matching this track (by IOU)
            aligned_face = None
            face_image = None
            
            for af, det in aligned_faces:
                if self._box_iou(det['box'], track.bbox) > 0.5:
                    aligned_face = af
                    # Crop face from frame
                    x1, y1, x2, y2 = track.bbox
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    face_image = frame[y1:y2, x1:x2].copy()
                    break
            
            if face_image is None or face_image.size == 0:
                continue
            
            # 4. Process through session manager
            session = self.session_manager.process_frame(
                track,
                face_image,
                aligned_face
            )
            
            # 5. Check for decision
            if session.decision != CheckinDecision.PENDING:
                # Check cooldown for accepted check-ins
                if session.decision == CheckinDecision.ACCEPTED:
                    if self.checkin_logger.is_on_cooldown(session.matched_user_id):
                        logger.info(f"User {session.matched_user_id} on cooldown, skipping")
                        continue
                    
                    # Log check-in
                    self.checkin_logger.log_checkin(
                        user_id=session.matched_user_id,
                        camera_id=camera_id,
                        confidence=session.decision_confidence,
                        similarity=session.match_similarity,
                        evidence_frame=session.best_frame
                    )
                
                # Create event
                event = CheckinEvent(
                    track_id=track.track_id,
                    decision=session.decision,
                    user_id=session.matched_user_id,
                    confidence=session.decision_confidence,
                    camera_id=camera_id,
                    timestamp=time.time(),
                    session=session
                )
                events.append(event)
                
                # Mark track as confirmed
                track.state = TrackState.CONFIRMED
        
        # 6. Cleanup exited tracks
        for track in self.tracker.tracks.values():
            if track.state == TrackState.EXIT:
                session = self.session_manager.finalize_session(track.track_id)
                if session and session.decision != CheckinDecision.PENDING:
                    # Already handled or no decision
                    pass
                self.session_manager.remove_session(track.track_id)
        
        return events
    
    def _box_iou(self, box1: List[int], box2: List[int]) -> float:
        """Compute IOU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0.0
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'is_running': self.is_running,
            'frame_count': self.frame_count,
            'process_count': self.process_count,
            'drop_count': self.drop_count,
            'drop_rate': self.drop_count / max(1, self.frame_count),
            'queue_size': self.frame_queue.qsize() if self.frame_queue else 0,
            'active_tracks': len(self.tracker.get_active_tracks()),
            'active_sessions': len(self.session_manager.get_all_sessions()),
            'recent_events': len(self.events)
        }
    
    def get_recent_events(self, count: int = 10) -> List[Dict]:
        """Get recent check-in events"""
        return [e.to_dict() for e in self.events[-count:]]


# Singleton instance
_stream_processor_instance: Optional[StreamProcessor] = None


def get_stream_processor(device: str = 'cuda') -> StreamProcessor:
    """Get or create StreamProcessor singleton"""
    global _stream_processor_instance
    if _stream_processor_instance is None:
        _stream_processor_instance = StreamProcessor(device=device)
    return _stream_processor_instance
