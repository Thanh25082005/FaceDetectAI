import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: insightface not installed. Face recognition will not work.")

from scipy.spatial.distance import cosine
import sys
sys.path.append('..')
from config import FACE_RECOGNITION_THRESHOLD, FACE_SIZE


class FaceRecognizer:
    """
    Face recognizer using InsightFace with ArcFace model
    """
    
    def __init__(self, model_name: str = 'buffalo_l', device: str = 'cpu'):
        """
        Initialize InsightFace recognizer
        
        Args:
            model_name: InsightFace model name ('buffalo_l', 'buffalo_s', etc.)
            device: 'cpu' or 'cuda' for GPU (-1 for CPU, 0 for GPU)
        """
        self.model_name = model_name
        self.device = device
        
        if INSIGHTFACE_AVAILABLE:
            # Set provider based on device
            providers = ['CPUExecutionProvider']
            if device == 'cuda':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.app = FaceAnalysis(
                name=model_name,
                providers=providers
            )
            # Prepare model for inference
            ctx_id = 0 if device == 'cuda' else -1
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        else:
            self.app = None
    
    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from a cropped face image
        
        Args:
            face_image: Cropped face image in BGR format
        
        Returns:
            512-dimensional embedding vector or None if no face detected
        """
        if self.app is None:
            raise RuntimeError("InsightFace not available. Please install insightface.")
        
        # InsightFace expects BGR format
        faces = self.app.get(face_image)
        
        if len(faces) == 0:
            return None
        
        # Get embedding from the first (or largest) face
        # Sort by face area (det_score can also be used)
        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        
        return faces[0].embedding
    
    def get_embedding_from_full_image(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces and extract embeddings from a full image
        
        Args:
            image: Full image in BGR format
        
        Returns:
            List of dictionaries containing face info and embeddings
        """
        if self.app is None:
            raise RuntimeError("InsightFace not available. Please install insightface.")
        
        faces = self.app.get(image)
        
        results = []
        for face in faces:
            face_info = {
                'box': [int(coord) for coord in face.bbox],
                'embedding': face.embedding.tolist(),
                'det_score': float(face.det_score) if hasattr(face, 'det_score') else None,
                'age': int(face.age) if hasattr(face, 'age') else None,
                'gender': 'M' if hasattr(face, 'gender') and face.gender == 1 else 'F' if hasattr(face, 'gender') else None
            }
            results.append(face_info)
        
        return results
    
    @staticmethod
    def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score (0-1, higher is more similar)
        """
        # Convert to numpy arrays if needed
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2)
        
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity (1 - cosine distance)
        similarity = 1 - cosine(embedding1, embedding2)
        
        return float(similarity)
    
    def recognize(
        self,
        query_embedding: np.ndarray,
        database_embeddings: List[Dict],
        threshold: float = None
    ) -> Optional[Dict]:
        """
        Recognize a face by comparing with database embeddings
        
        Args:
            query_embedding: Embedding of the query face
            database_embeddings: List of dicts with 'user_id' and 'embedding'
            threshold: Similarity threshold (uses default from config if None)
        
        Returns:
            Best match dict with 'user_id', 'similarity', 'is_match' or None if no matches
        """
        if threshold is None:
            threshold = FACE_RECOGNITION_THRESHOLD
        
        if not database_embeddings:
            return None
        
        best_match = None
        best_similarity = -1
        
        for db_entry in database_embeddings:
            db_embedding = db_entry['embedding']
            if isinstance(db_embedding, list):
                db_embedding = np.array(db_embedding)
            
            similarity = self.compute_similarity(query_embedding, db_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    'user_id': db_entry['user_id'],
                    'similarity': similarity,
                    'is_match': similarity >= threshold
                }
        
        return best_match
    
    def compare_faces(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = None
    ) -> Dict:
        """
        Compare two face embeddings and return match result
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            threshold: Similarity threshold (uses default from config if None)
        
        Returns:
            Dictionary with 'similarity' and 'is_match'
        """
        if threshold is None:
            threshold = FACE_RECOGNITION_THRESHOLD
        
        similarity = self.compute_similarity(embedding1, embedding2)
        
        return {
            'similarity': similarity,
            'is_match': similarity >= threshold
        }


# Singleton instance for reuse
_recognizer_instance: Optional[FaceRecognizer] = None


def get_face_recognizer(device: str = 'cpu') -> FaceRecognizer:
    """
    Get or create a FaceRecognizer singleton instance
    """
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = FaceRecognizer(device=device)
    return _recognizer_instance
