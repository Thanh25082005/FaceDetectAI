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
from config import FR_THRESHOLD, FACE_SIZE, FR_EMA_ALPHA, FR_MIN_FRAMES


class EmbeddingAggregator:
    """
    Aggregate face embeddings over time for a single track.
    
    Uses Exponential Moving Average (EMA) for smooth aggregation,
    with normalization to maintain unit vector property.
    """
    
    def __init__(self, ema_alpha: float = None):
        """
        Initialize embedding aggregator.
        
        Args:
            ema_alpha: EMA weight (higher = more weight to recent)
        """
        self.ema_alpha = ema_alpha or FR_EMA_ALPHA
        self.embeddings: List[np.ndarray] = []
        self.aggregated: Optional[np.ndarray] = None
        self.similarity_history: List[float] = []
    
    def add_embedding(self, embedding: np.ndarray):
        """
        Add an embedding and update aggregation.
        
        Args:
            embedding: 512-d normalized embedding vector
        """
        if embedding is None:
            return
        
        # Normalize input
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self.embeddings.append(embedding)
        
        # Update EMA
        if self.aggregated is None:
            self.aggregated = embedding.copy()
        else:
            self.aggregated = (
                self.ema_alpha * embedding + 
                (1 - self.ema_alpha) * self.aggregated
            )
            # Re-normalize
            norm = np.linalg.norm(self.aggregated)
            if norm > 0:
                self.aggregated = self.aggregated / norm
    
    def get_aggregated(self) -> Optional[np.ndarray]:
        """Get the aggregated embedding (EMA)"""
        return self.aggregated
    
    def get_mean(self) -> Optional[np.ndarray]:
        """Get the mean of all embeddings"""
        if not self.embeddings:
            return None
        mean = np.mean(self.embeddings, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean
    
    def can_recognize(self) -> bool:
        """Check if we have enough embeddings for recognition"""
        return len(self.embeddings) >= FR_MIN_FRAMES
    
    def get_stability(self) -> float:
        """
        Get embedding stability (consistency) score.
        
        High stability = consistent face embeddings = more reliable match.
        """
        if len(self.embeddings) < 2:
            return 0.0
        
        # Compute pairwise similarities of recent embeddings
        recent = self.embeddings[-min(10, len(self.embeddings)):]
        similarities = []
        
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                sim = 1 - cosine(recent[i], recent[j])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def add_match_result(self, similarity: float):
        """Track similarity scores from recognition attempts"""
        self.similarity_history.append(similarity)
    
    def get_average_similarity(self) -> float:
        """Get average similarity from match attempts"""
        if not self.similarity_history:
            return 0.0
        return np.mean(self.similarity_history)
    
    def reset(self):
        """Reset aggregator"""
        self.embeddings.clear()
        self.aggregated = None
        self.similarity_history.clear()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'count': len(self.embeddings),
            'can_recognize': self.can_recognize(),
            'stability': self.get_stability(),
            'average_similarity': self.get_average_similarity()
        }


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
            
            # Cache recognition model for direct embedding extraction
            self.rec_model = None
            for model in self.app.models.values():
                if hasattr(model, 'get_feat'):
                    self.rec_model = model
                    break
        else:
            self.app = None
            self.rec_model = None
    
    def get_embedding_direct(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding directly from a pre-aligned face image (112x112).
        This method does NOT run face detection - use for aligned faces from MTCNN.
        
        Args:
            aligned_face: Pre-aligned face image (112x112) in BGR format
        
        Returns:
            512-dimensional embedding vector or None if extraction fails
        """
        if self.rec_model is None:
            raise RuntimeError("Recognition model not available. Please install insightface.")
        
        try:
            # Ensure input is 112x112
            if aligned_face.shape[:2] != (112, 112):
                aligned_face = cv2.resize(aligned_face, (112, 112))
            
            # Use cv2.dnn.blobFromImage for correct preprocessing
            # - Scale by 1/127.5
            # - Subtract mean (127.5, 127.5, 127.5)
            # - Swap BGR to RGB
            blob = cv2.dnn.blobFromImage(
                aligned_face, 
                scalefactor=1.0/127.5, 
                size=(112, 112), 
                mean=(127.5, 127.5, 127.5), 
                swapRB=True
            )
            
            # Run inference directly with ONNX session
            input_name = self.rec_model.session.get_inputs()[0].name
            output = self.rec_model.session.run(None, {input_name: blob})
            embedding = output[0].flatten()
            
            return embedding
        except Exception as e:
            print(f"Embedding extraction failed: {e}")
            return None
    
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
            threshold = FR_THRESHOLD
        
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
            threshold = FR_THRESHOLD
        
        similarity = self.compute_similarity(embedding1, embedding2)
        
        return {
            'similarity': similarity,
            'is_match': similarity >= threshold
        }


# Singleton instance for reuse
_recognizer_instance: Optional[FaceRecognizer] = None


def get_face_recognizer(device: str = None) -> FaceRecognizer:
    """
    Get singleton instance of FaceRecognizer
    """
    global _recognizer_instance
    if device is None:
        from config import DEVICE
        device = DEVICE
    if _recognizer_instance is None:
        _recognizer_instance = FaceRecognizer(device=device)
    return _recognizer_instance
