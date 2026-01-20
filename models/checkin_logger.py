"""
Check-in Logger Module

Logs check-in events and manages cooldown to prevent re-check-in.
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2

import sys
sys.path.append('..')
from config import CHECKIN_COOLDOWN_MINUTES, CHECKIN_LOG_PATH, EVIDENCE_DIR


@dataclass
class CheckinRecord:
    """A single check-in record"""
    id: int
    user_id: str
    camera_id: str
    timestamp: datetime
    confidence: float
    fas_score: float
    similarity: float
    evidence_path: Optional[str]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'camera_id': self.camera_id,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'fas_score': self.fas_score,
            'similarity': self.similarity,
            'evidence_path': self.evidence_path
        }


class CheckinLogger:
    """
    Logs check-in events and manages cooldown.
    
    Features:
    - SQLite storage for check-in records
    - Cooldown checking to prevent re-check-in
    - Evidence image saving
    """
    
    def __init__(self, db_path: str = None, evidence_dir: str = None):
        """
        Initialize check-in logger.
        
        Args:
            db_path: Path to SQLite database
            evidence_dir: Directory for evidence images
        """
        self.db_path = db_path or str(CHECKIN_LOG_PATH)
        self.evidence_dir = Path(evidence_dir) if evidence_dir else EVIDENCE_DIR
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database tables"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                camera_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                fas_score REAL,
                similarity REAL,
                evidence_path TEXT,
                metadata TEXT
            )
        """)
        
        # Index for quick cooldown checks
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_checkin_user_time 
            ON checkins(user_id, timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def log_checkin(
        self,
        user_id: str,
        camera_id: str = None,
        confidence: float = 0.0,
        fas_score: float = 0.0,
        similarity: float = 0.0,
        evidence_frame: np.ndarray = None,
        metadata: Dict = None
    ) -> CheckinRecord:
        """
        Log a successful check-in.
        
        Args:
            user_id: User identifier
            camera_id: Camera identifier
            confidence: Overall decision confidence
            fas_score: FAS (anti-spoofing) score
            similarity: Face recognition similarity
            evidence_frame: Best face frame for evidence
            metadata: Additional metadata
        
        Returns:
            CheckinRecord
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Save evidence image if provided
        evidence_path = None
        if evidence_frame is not None:
            evidence_path = self._save_evidence(user_id, evidence_frame)
        
        cursor.execute("""
            INSERT INTO checkins 
            (user_id, camera_id, confidence, fas_score, similarity, evidence_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            camera_id,
            confidence,
            fas_score,
            similarity,
            evidence_path,
            json.dumps(metadata) if metadata else None
        ))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return CheckinRecord(
            id=record_id,
            user_id=user_id,
            camera_id=camera_id,
            timestamp=datetime.now(),
            confidence=confidence,
            fas_score=fas_score,
            similarity=similarity,
            evidence_path=evidence_path
        )
    
    def _save_evidence(self, user_id: str, frame: np.ndarray) -> str:
        """Save evidence frame and return path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{user_id}_{timestamp}.jpg"
        filepath = self.evidence_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        return str(filepath)
    
    def is_on_cooldown(
        self,
        user_id: str,
        cooldown_minutes: int = None
    ) -> bool:
        """
        Check if user is on cooldown (recently checked in).
        
        Args:
            user_id: User identifier
            cooldown_minutes: Cooldown period (uses config default if None)
        
        Returns:
            True if on cooldown, False otherwise
        """
        cooldown = cooldown_minutes or CHECKIN_COOLDOWN_MINUTES
        cutoff = datetime.now() - timedelta(minutes=cooldown)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM checkins 
            WHERE user_id = ? AND timestamp > ?
        """, (user_id, cutoff))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def get_last_checkin(self, user_id: str) -> Optional[CheckinRecord]:
        """Get the most recent check-in for a user"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM checkins 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return CheckinRecord(
            id=row['id'],
            user_id=row['user_id'],
            camera_id=row['camera_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            confidence=row['confidence'],
            fas_score=row['fas_score'],
            similarity=row['similarity'],
            evidence_path=row['evidence_path']
        )
    
    def get_recent_checkins(
        self,
        minutes: int = 30,
        user_id: str = None,
        camera_id: str = None
    ) -> List[CheckinRecord]:
        """
        Get recent check-ins.
        
        Args:
            minutes: How far back to look
            user_id: Filter by user (optional)
            camera_id: Filter by camera (optional)
        
        Returns:
            List of CheckinRecords
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM checkins WHERE timestamp > ?"
        params = [cutoff]
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [
            CheckinRecord(
                id=row['id'],
                user_id=row['user_id'],
                camera_id=row['camera_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                confidence=row['confidence'],
                fas_score=row['fas_score'],
                similarity=row['similarity'],
                evidence_path=row['evidence_path']
            )
            for row in rows
        ]
    
    def get_checkin_count_today(self, user_id: str = None) -> int:
        """Get number of check-ins today"""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute("""
                SELECT COUNT(*) FROM checkins 
                WHERE user_id = ? AND timestamp > ?
            """, (user_id, today))
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM checkins 
                WHERE timestamp > ?
            """, (today,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count


# Singleton instance
_checkin_logger_instance: Optional[CheckinLogger] = None


def get_checkin_logger() -> CheckinLogger:
    """Get or create CheckinLogger singleton"""
    global _checkin_logger_instance
    if _checkin_logger_instance is None:
        _checkin_logger_instance = CheckinLogger()
    return _checkin_logger_instance
