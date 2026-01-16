import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path

import sys
sys.path.append('..')
from config import DATABASE_PATH


class FaceDatabase:
    """
    SQLite database for storing face embeddings
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            db_path = str(DATABASE_PATH)
        
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
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
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                name TEXT,
                embedding TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on user_id for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id ON faces(user_id)
        """)
        
        conn.commit()
        conn.close()
    
    def add_face(
        self,
        user_id: str,
        embedding: np.ndarray,
        name: str = None,
        metadata: Dict = None
    ) -> Dict:
        """
        Add a new face to the database
        
        Args:
            user_id: Unique identifier for the user
            embedding: 512-d face embedding vector
            name: Optional name for the user
            metadata: Optional additional metadata
        
        Returns:
            Dictionary with operation result
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Convert embedding to JSON string
            embedding_json = json.dumps(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                INSERT INTO faces (user_id, name, embedding, metadata)
                VALUES (?, ?, ?, ?)
            """, (user_id, name, embedding_json, metadata_json))
            
            conn.commit()
            
            return {
                'success': True,
                'message': f'Face added for user {user_id}',
                'id': cursor.lastrowid
            }
        except sqlite3.IntegrityError:
            return {
                'success': False,
                'message': f'User {user_id} already exists. Use update_face to modify.'
            }
        finally:
            conn.close()
    
    def get_face(self, user_id: str) -> Optional[Dict]:
        """
        Get face data by user_id
        
        Args:
            user_id: User identifier
        
        Returns:
            Face data dictionary or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM faces WHERE user_id = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return {
            'id': row['id'],
            'user_id': row['user_id'],
            'name': row['name'],
            'embedding': json.loads(row['embedding']),
            'metadata': json.loads(row['metadata']) if row['metadata'] else None,
            'created_at': row['created_at'],
            'updated_at': row['updated_at']
        }
    
    def update_face(
        self,
        user_id: str,
        embedding: np.ndarray = None,
        name: str = None,
        metadata: Dict = None
    ) -> Dict:
        """
        Update face data for a user
        
        Args:
            user_id: User identifier
            embedding: Optional new embedding
            name: Optional new name
            metadata: Optional new metadata
        
        Returns:
            Dictionary with operation result
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT id FROM faces WHERE user_id = ?", (user_id,))
        if cursor.fetchone() is None:
            conn.close()
            return {
                'success': False,
                'message': f'User {user_id} not found'
            }
        
        # Build update query
        updates = []
        params = []
        
        if embedding is not None:
            embedding_json = json.dumps(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
            updates.append("embedding = ?")
            params.append(embedding_json)
        
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))
        
        if updates:
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(user_id)
            
            query = f"UPDATE faces SET {', '.join(updates)} WHERE user_id = ?"
            cursor.execute(query, params)
            conn.commit()
        
        conn.close()
        
        return {
            'success': True,
            'message': f'Face updated for user {user_id}'
        }
    
    def delete_face(self, user_id: str) -> Dict:
        """
        Delete face data for a user
        
        Args:
            user_id: User identifier
        
        Returns:
            Dictionary with operation result
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM faces WHERE user_id = ?", (user_id,))
        
        if cursor.rowcount == 0:
            conn.close()
            return {
                'success': False,
                'message': f'User {user_id} not found'
            }
        
        conn.commit()
        conn.close()
        
        return {
            'success': True,
            'message': f'Face deleted for user {user_id}'
        }
    
    def get_all_embeddings(self) -> List[Dict]:
        """
        Get all face embeddings from database
        
        Returns:
            List of dictionaries with user_id and embedding
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT user_id, name, embedding FROM faces")
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'user_id': row['user_id'],
                'name': row['name'],
                'embedding': json.loads(row['embedding'])
            })
        
        conn.close()
        return results
    
    def get_user_count(self) -> int:
        """Get total number of users in database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM faces")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def search_by_name(self, name_query: str) -> List[Dict]:
        """
        Search users by name (partial match)
        
        Args:
            name_query: Partial name to search
        
        Returns:
            List of matching user records
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, name, created_at, updated_at
            FROM faces
            WHERE name LIKE ?
        """, (f"%{name_query}%",))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'user_id': row['user_id'],
                'name': row['name'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            })
        
        conn.close()
        return results


# Singleton instance for reuse
_database_instance: Optional[FaceDatabase] = None


def get_face_database() -> FaceDatabase:
    """
    Get or create a FaceDatabase singleton instance
    """
    global _database_instance
    if _database_instance is None:
        _database_instance = FaceDatabase()
    return _database_instance
