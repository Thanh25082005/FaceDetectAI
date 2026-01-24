"""
Face Recognition Service with In-Memory Caching

Provides high-performance face recognition by caching embeddings in RAM.
Designed for real-time face recognition with ~3,000 employees.

Features:
- Async SQL Server integration
- In-memory numpy matrix for fast cosine similarity
- Singleton pattern for shared cache
- Automatic cache refresh
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import sys
sys.path.append('..')

from config import FR_THRESHOLD


# =============================================================================
# Serialization Utilities
# =============================================================================

def numpy_to_bytes(array: np.ndarray) -> bytes:
    """
    Convert numpy array to bytes for storing in VARBINARY(MAX).
    
    Args:
        array: Numpy array (typically float32, 512-dimensional)
    
    Returns:
        bytes: Binary representation of the array
    
    Notes:
        - Ensures float32 dtype for consistent 2048 bytes (512 * 4)
        - Uses little-endian byte order (default for most systems)
    
    Example:
        >>> embedding = np.random.randn(512).astype(np.float32)
        >>> data = numpy_to_bytes(embedding)
        >>> len(data)
        2048
    """
    # Ensure float32 for consistent size and precision
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    
    # Convert to bytes using numpy's built-in method
    # This preserves the exact binary representation
    return array.tobytes()


def bytes_to_numpy(bytes_data: bytes, dtype: np.dtype = np.float32) -> Optional[np.ndarray]:
    """
    Convert bytes from VARBINARY(MAX) back to numpy array.
    
    Args:
        bytes_data: Binary data from database
        dtype: Expected numpy dtype (default: float32)
    
    Returns:
        np.ndarray: Reconstructed numpy array, or None if invalid
    
    Notes:
        - Validates data length matches expected size
        - For 512-d float32: expects exactly 2048 bytes
    
    Example:
        >>> data = b'...'  # 2048 bytes from database
        >>> embedding = bytes_to_numpy(data)
        >>> embedding.shape
        (512,)
    """
    if bytes_data is None or len(bytes_data) == 0:
        return None
    
    try:
        # Reconstruct array from bytes
        array = np.frombuffer(bytes_data, dtype=dtype)
        
        # Validate expected size for face embeddings
        if len(array) != 512:
            print(f"⚠️ Warning: Expected 512-d vector, got {len(array)}-d")
        
        return array
    except Exception as e:
        print(f"❌ Error converting bytes to numpy: {e}")
        return None


def image_to_bytes(image: np.ndarray) -> bytes:
    """
    Convert image numpy array to bytes for storage.
    
    Args:
        image: OpenCV image (BGR format, uint8)
    
    Returns:
        bytes: JPEG-encoded image data
    """
    import cv2
    _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return encoded.tobytes()


def bytes_to_image(bytes_data: bytes) -> Optional[np.ndarray]:
    """
    Convert bytes back to OpenCV image.
    
    Args:
        bytes_data: JPEG-encoded image data
    
    Returns:
        np.ndarray: OpenCV image (BGR format)
    """
    if bytes_data is None:
        return None
    
    import cv2
    try:
        nparr = np.frombuffer(bytes_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ Error converting bytes to image: {e}")
        return None


# =============================================================================
# Face Recognition Service (Singleton)
# =============================================================================

@dataclass
class CachedFace:
    """Cached face data for in-memory storage."""
    user_id: str
    name_user: str
    embedding: np.ndarray


class FaceRecognitionService:
    """
    Singleton service for face recognition with in-memory caching.
    
    Caches all face embeddings in RAM for fast real-time recognition.
    Uses numpy vectorized operations for efficient cosine similarity.
    
    Attributes:
        _embeddings_matrix: (N, 512) numpy matrix of all embeddings
        _user_ids: List of user_ids corresponding to matrix rows
        _names: List of names corresponding to matrix rows
        _is_loaded: Whether cache has been loaded
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern - only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize service (only runs once due to singleton)."""
        if self._initialized:
            return
        
        # In-memory cache
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._user_ids: List[str] = []
        self._names: List[str] = []
        self._is_loaded: bool = False
        self._last_refresh: Optional[datetime] = None
        self._lock = asyncio.Lock()
        
        self._initialized = True
        print("🧠 FaceRecognitionService initialized")
    
    @property
    def is_loaded(self) -> bool:
        """Check if embeddings are loaded in memory."""
        return self._is_loaded
    
    @property
    def face_count(self) -> int:
        """Get number of faces in cache."""
        return len(self._user_ids)
    
    async def load_embeddings_to_memory(self) -> int:
        """
        Load all face embeddings from SQL Server into RAM.
        
        Uses async SQLAlchemy session to fetch all active faces,
        then converts to numpy matrix for fast similarity search.
        
        Returns:
            int: Number of faces loaded
        
        Notes:
            - Should be called on application startup
            - Uses asyncio lock to prevent concurrent loads
            - Memory usage: ~3000 faces * 512 * 4 bytes = ~6 MB
        """
        async with self._lock:
            print("📥 Loading embeddings from SQL Server...")
            start_time = datetime.now()
            
            try:
                from database.connection import get_session_factory
                from database.models import Face
                from sqlalchemy import select
                
                factory = get_session_factory()
                async with factory() as session:
                    # Fetch all faces with embeddings
                    stmt = select(Face.user_id, Face.name_user, Face.embedding)
                    result = await session.execute(stmt)
                    rows = result.fetchall()
                
                if not rows:
                    print("⚠️ No faces found in database")
                    self._embeddings_matrix = np.array([]).reshape(0, 512)
                    self._user_ids = []
                    self._names = []
                    self._is_loaded = True
                    return 0
                
                # Convert to numpy matrix
                embeddings = []
                user_ids = []
                names = []
                
                for row in rows:
                    user_id, name_user, embedding_bytes = row
                    embedding = bytes_to_numpy(embedding_bytes)
                    
                    if embedding is not None:
                        embeddings.append(embedding)
                        user_ids.append(user_id)
                        names.append(name_user or "")
                    else:
                        print(f"⚠️ Skipping invalid embedding for user {user_id}")
                
                # Create matrix (N, 512)
                self._embeddings_matrix = np.array(embeddings, dtype=np.float32)
                self._user_ids = user_ids
                self._names = names
                self._is_loaded = True
                self._last_refresh = datetime.now()
                
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"✅ Loaded {len(user_ids)} embeddings in {elapsed:.2f}s")
                print(f"   Matrix shape: {self._embeddings_matrix.shape}")
                print(f"   Memory usage: {self._embeddings_matrix.nbytes / 1024:.1f} KB")
                
                return len(user_ids)
                
            except Exception as e:
                print(f"❌ Failed to load embeddings: {e}")
                raise
    
    async def refresh_cache(self) -> int:
        """
        Refresh cache from database.
        
        Alias for load_embeddings_to_memory().
        Call this when new faces are added.
        
        Returns:
            int: Number of faces in cache
        """
        return await self.load_embeddings_to_memory()
    
    def add_face_to_cache(self, user_id: str, name_user: str, embedding: np.ndarray):
        """
        Add a new face to cache without database reload.
        
        Useful for adding faces in real-time without full refresh.
        
        Args:
            user_id: User identifier
            name_user: User name
            embedding: 512-d face embedding
        """
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        if self._embeddings_matrix is None or len(self._embeddings_matrix) == 0:
            self._embeddings_matrix = embedding.reshape(1, -1)
        else:
            self._embeddings_matrix = np.vstack([self._embeddings_matrix, embedding])
        
        self._user_ids.append(user_id)
        self._names.append(name_user or "")
        print(f"➕ Added {user_id} to cache. Total: {len(self._user_ids)}")
    
    def remove_face_from_cache(self, user_id: str) -> bool:
        """
        Remove a face from cache.
        
        Args:
            user_id: User identifier to remove
        
        Returns:
            bool: True if removed, False if not found
        """
        try:
            idx = self._user_ids.index(user_id)
            self._embeddings_matrix = np.delete(self._embeddings_matrix, idx, axis=0)
            self._user_ids.pop(idx)
            self._names.pop(idx)
            print(f"➖ Removed {user_id} from cache. Total: {len(self._user_ids)}")
            return True
        except ValueError:
            return False
    
    def identify_face(
        self, 
        input_vector: np.ndarray, 
        threshold: float = None
    ) -> Optional[Dict]:
        """
        Identify a face by comparing with cached embeddings.
        
        Uses cosine similarity for matching. Vectorized computation
        makes this very fast even with thousands of faces.
        
        Args:
            input_vector: 512-d face embedding from camera
            threshold: Similarity threshold (default: FR_THRESHOLD from config)
        
        Returns:
            Dict with keys:
                - user_id: Matched user ID
                - name_user: Matched user name
                - similarity: Cosine similarity score (0-1)
                - is_match: Whether similarity > threshold
            Or None if no faces in cache
        
        Performance:
            - 3000 faces: ~0.5ms with numpy
            - Can be improved to ~0.05ms with FAISS
        """
        if not self._is_loaded or len(self._user_ids) == 0:
            print("⚠️ Cache not loaded or empty")
            return None
        
        threshold = threshold or FR_THRESHOLD
        
        # Ensure correct shape and dtype
        if input_vector.dtype != np.float32:
            input_vector = input_vector.astype(np.float32)
        
        if input_vector.ndim == 1:
            input_vector = input_vector.reshape(1, -1)
        
        # Normalize input vector
        input_norm = input_vector / (np.linalg.norm(input_vector) + 1e-8)
        
        # Normalize cached embeddings (should already be normalized, but ensure)
        cache_norms = np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True) + 1e-8
        normalized_cache = self._embeddings_matrix / cache_norms
        
        # Compute cosine similarity with all cached embeddings
        # Shape: (1, 512) @ (512, N) = (1, N)
        similarities = np.dot(input_norm, normalized_cache.T).flatten()
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = float(similarities[best_idx])
        
        return {
            "user_id": self._user_ids[best_idx],
            "name_user": self._names[best_idx],
            "similarity": best_similarity,
            "is_match": best_similarity >= threshold
        }
    
    def identify_face_top_k(
        self, 
        input_vector: np.ndarray, 
        k: int = 5,
        threshold: float = None
    ) -> List[Dict]:
        """
        Get top-k matches for a face.
        
        Args:
            input_vector: 512-d face embedding
            k: Number of top matches to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of dicts with user_id, name_user, similarity
        """
        if not self._is_loaded or len(self._user_ids) == 0:
            return []
        
        threshold = threshold or 0.0
        
        if input_vector.dtype != np.float32:
            input_vector = input_vector.astype(np.float32)
        
        if input_vector.ndim == 1:
            input_vector = input_vector.reshape(1, -1)
        
        input_norm = input_vector / (np.linalg.norm(input_vector) + 1e-8)
        cache_norms = np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True) + 1e-8
        normalized_cache = self._embeddings_matrix / cache_norms
        
        similarities = np.dot(input_norm, normalized_cache.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append({
                    "user_id": self._user_ids[idx],
                    "name_user": self._names[idx],
                    "similarity": sim
                })
        
        return results
    
    def get_status(self) -> Dict:
        """Get service status for health check."""
        return {
            "is_loaded": self._is_loaded,
            "face_count": len(self._user_ids),
            "matrix_shape": self._embeddings_matrix.shape if self._embeddings_matrix is not None else None,
            "memory_bytes": self._embeddings_matrix.nbytes if self._embeddings_matrix is not None else 0,
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None
        }


# =============================================================================
# Singleton Accessor
# =============================================================================

_face_service_instance: Optional[FaceRecognitionService] = None


def get_face_service() -> FaceRecognitionService:
    """
    Get singleton instance of FaceRecognitionService.
    
    Returns:
        FaceRecognitionService: The singleton instance
    """
    global _face_service_instance
    if _face_service_instance is None:
        _face_service_instance = FaceRecognitionService()
    return _face_service_instance
