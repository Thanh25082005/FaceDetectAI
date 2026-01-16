import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image
import numpy as np

import sys
sys.path.insert(0, '..')
from main import app

client = TestClient(app)


def create_test_image(width: int = 200, height: int = 200) -> bytes:
    """Create a simple test image as bytes"""
    # Create a simple image with a face-like pattern (for testing)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some color to simulate a face
    image[50:150, 50:150] = [200, 180, 160]  # Skin tone area
    
    # Convert to bytes
    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_check(self):
        """Test health endpoint returns OK"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'models_loaded' in data
        assert 'database_users' in data


class TestRootEndpoint:
    """Tests for root endpoint"""
    
    def test_root(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'docs' in data


class TestFaceDetection:
    """Tests for face detection endpoint"""
    
    def test_detect_face_no_file(self):
        """Test detection without file returns error"""
        response = client.post("/api/v1/detect_face")
        assert response.status_code == 422  # Validation error
    
    def test_detect_face_with_image(self):
        """Test detection with valid image"""
        image_bytes = create_test_image()
        files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
        
        response = client.post("/api/v1/detect_face", files=files)
        assert response.status_code == 200
        data = response.json()
        assert 'success' in data
        assert 'faces_count' in data
        assert 'faces' in data
        assert 'image_size' in data


class TestFaceDatabase:
    """Tests for face database operations"""
    
    def test_get_nonexistent_face(self):
        """Test getting a face that doesn't exist"""
        response = client.get("/api/v1/get_face/nonexistent_user_123")
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is False
        assert 'not found' in data['message'].lower()
    
    def test_delete_nonexistent_face(self):
        """Test deleting a face that doesn't exist"""
        response = client.delete("/api/v1/delete_face/nonexistent_user_123")
        assert response.status_code == 404


class TestAntiSpoofing:
    """Tests for anti-spoofing endpoint"""
    
    def test_anti_spoofing_with_image(self):
        """Test anti-spoofing with valid image"""
        image_bytes = create_test_image()
        files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
        
        response = client.post("/api/v1/anti_spoofing", files=files)
        assert response.status_code == 200
        data = response.json()
        assert 'success' in data
        assert 'is_live' in data
        assert 'confidence' in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
