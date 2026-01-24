"""
SQL Server Database for Face Recognition System

Uses SQL Server for face embeddings storage.
Embeddings stored as VARBINARY (binary format).
"""

import os
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import pyodbc

import sys
sys.path.append('..')
from config import MSSQL_HOST, MSSQL_PORT, MSSQL_USER, MSSQL_PASSWORD, MSSQL_DATABASE


def get_connection_string() -> str:
    """Get ODBC connection string for SQL Server."""
    return (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={MSSQL_HOST},{MSSQL_PORT};"
        f"DATABASE={MSSQL_DATABASE};"
        f"UID={MSSQL_USER};"
        f"PWD={MSSQL_PASSWORD};"
        f"TrustServerCertificate=yes;"
        f"Connection Timeout=30"
    )


def numpy_to_bytes(array) -> bytes:
    """Convert numpy array or list to bytes for VARBINARY storage."""
    if isinstance(array, list):
        array = np.array(array, dtype=np.float32)
    elif not isinstance(array, np.ndarray):
        array = np.array(array, dtype=np.float32)
    return array.astype(np.float32).tobytes()


def bytes_to_numpy(data: bytes) -> Optional[np.ndarray]:
    """Convert bytes from VARBINARY back to numpy array."""
    if data is None:
        return None
    return np.frombuffer(data, dtype=np.float32)


class FaceDatabase:
    """
    SQL Server database for storing face embeddings.
    Embeddings stored as VARBINARY(MAX) for efficient storage.
    """
    
    def __init__(self):
        """Initialize database connection."""
        self._conn_str = get_connection_string()
        print("✅ SQL Server FaceDatabase initialized")
    
    def _get_connection(self) -> pyodbc.Connection:
        """Get database connection."""
        return pyodbc.connect(self._conn_str)

    # =========================================================================
    # User Authentication Methods (stubs for compatibility)
    # =========================================================================
    
    def create_user(self, username, password_hash, full_name, dob, face_user_id) -> Dict:
        return {'success': False, 'message': 'User management not implemented'}

    def get_user_by_username(self, username) -> Optional[Dict]:
        return None

    # =========================================================================
    # Face Management Methods
    # =========================================================================

    def add_face(self, user_id: str, embedding, name: str = None, metadata: Dict = None) -> Dict:
        """Add a new face embedding to the database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Convert embedding to bytes for VARBINARY column
            embedding_bytes = numpy_to_bytes(embedding)
            
            cursor.execute("""
                INSERT INTO faces (user_id, name_user, embedding)
                VALUES (?, ?, ?)
            """, (user_id, name, embedding_bytes))
            
            # Get the inserted ID
            cursor.execute("SELECT SCOPE_IDENTITY()")
            row = cursor.fetchone()
            last_id = int(row[0]) if row and row[0] else None
            
            conn.commit()
            print(f"✅ Face added: user_id={user_id}, id={last_id}")
            return {'success': True, 'message': f'Face added for user {user_id}', 'id': last_id}
        except pyodbc.IntegrityError:
            return {'success': False, 'message': f'Face entry for {user_id} already exists'}
        except Exception as e:
            print(f"❌ Error adding face: {e}")
            return {'success': False, 'message': str(e)}
        finally:
            conn.close()

    def get_face(self, user_id: str) -> Optional[Dict]:
        """Get face data by user_id"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT id, user_id, name_user, embedding, created_at FROM faces WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                embedding_array = bytes_to_numpy(row[3])
                created = str(row[4]) if row[4] else datetime.now().isoformat()
                return {
                    'id': row[0],
                    'user_id': row[1],
                    'name': row[2],
                    'embedding': embedding_array.tolist() if embedding_array is not None else None,
                    'created_at': created,
                    'updated_at': created  # Use same as created for compatibility
                }
        except Exception as e:
            print(f"Error getting face: {e}")
        finally:
            conn.close()
        return None

    def update_face(self, user_id: str, embedding = None, name: str = None, metadata: Dict = None) -> Dict:
        """Update face data for a user"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            updates = []
            params = []
            
            if embedding is not None:
                updates.append("embedding = ?")
                params.append(numpy_to_bytes(embedding))
            if name is not None:
                updates.append("name_user = ?")
                params.append(name)
            
            if not updates:
                return {'success': False, 'message': 'No updates provided'}
            
            params.append(user_id)
            
            query = f"UPDATE faces SET {', '.join(updates)} WHERE user_id = ?"
            cursor.execute(query, params)
            conn.commit()
            
            if cursor.rowcount == 0:
                return {'success': False, 'message': f'User {user_id} not found'}
            
            return {'success': True, 'message': f'Face updated for user {user_id}'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
        finally:
            conn.close()

    def delete_face(self, user_id: str) -> Dict:
        """Delete face data for a user"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM faces WHERE user_id = ?", (user_id,))
            conn.commit()
            if cursor.rowcount == 0:
                return {'success': False, 'message': f'User {user_id} not found'}
            return {'success': True, 'message': f'Face deleted for user {user_id}'}
        finally:
            conn.close()

    def get_all_embeddings(self) -> List[Dict]:
        """Get all face embeddings from database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        results = []
        try:
            cursor.execute("SELECT user_id, name_user, embedding FROM faces")
            for row in cursor.fetchall():
                embedding_array = bytes_to_numpy(row[2])
                results.append({
                    'user_id': row[0],
                    'name': row[1],
                    'embedding': embedding_array.tolist() if embedding_array is not None else None
                })
        except Exception as e:
            print(f"Error getting embeddings: {e}")
        finally:
            conn.close()
        return results

    def get_user_count(self) -> int:
        """Get total number of users in database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        count = 0
        try:
            cursor.execute("SELECT COUNT(*) FROM faces")
            count = cursor.fetchone()[0]
        except Exception as e:
            print(f"Error counting users: {e}")
        finally:
            conn.close()
        return count

    def search_by_name(self, name_query: str) -> List[Dict]:
        """Search users by name (partial match)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        results = []
        try:
            cursor.execute(
                "SELECT user_id, name_user, created_at FROM faces WHERE name_user LIKE ?", 
                (f"%{name_query}%",)
            )
            for row in cursor.fetchall():
                results.append({
                    'user_id': row[0],
                    'name': row[1],
                    'created_at': str(row[2]) if row[2] else None
                })
        except Exception as e:
            print(f"Error searching: {e}")
        finally:
            conn.close()
        return results


# Singleton instance
_database_instance: Optional[FaceDatabase] = None


def get_face_database() -> FaceDatabase:
    """Get or create a FaceDatabase singleton instance"""
    global _database_instance
    if _database_instance is None:
        _database_instance = FaceDatabase()
    return _database_instance
