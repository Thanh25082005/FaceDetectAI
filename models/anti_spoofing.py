import numpy as np
from typing import List, Dict, Optional
import cv2
from skimage.feature import local_binary_pattern

import sys
sys.path.append('..')
from config import TEXTURE_THRESHOLD


class AntiSpoofing:    
    def __init__(self):
        """Initialize anti-spoofing detector"""
        pass
    
    def analyze_texture(self, face_image: np.ndarray) -> Dict:
        """
        Analyze face texture using Local Binary Patterns (LBP)
        
        Real faces have different texture patterns compared to printed photos or screens.
        LBP captures micro-texture patterns that differ between real and fake faces.
        
        Args:
            face_image: Cropped face image in BGR format
        
        Returns:
            Dictionary with texture analysis results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistent analysis
        gray = cv2.resize(gray, (128, 128))
        
        # Apply LBP with different parameters
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Compute histogram
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # Compute texture features
        variance = np.var(hist)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        uniformity = np.sum(hist ** 2)
        
        # Laplacian variance (sharpness/blur detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_variance = laplacian.var()
        
        # Heuristic score calculation
        texture_score = (
            0.3 * min(variance * 1000, 1.0) +
            0.3 * min(entropy / 5.0, 1.0) +
            0.4 * min(laplacian_variance / 500.0, 1.0)
        )
        
        is_real = texture_score > TEXTURE_THRESHOLD
        
        return {
            'texture_variance': float(variance),
            'texture_entropy': float(entropy),
            'texture_uniformity': float(uniformity),
            'laplacian_variance': float(laplacian_variance),
            'texture_score': float(texture_score),
            'is_real': is_real
        }
    
    def detect_screen_reflection(self, face_image: np.ndarray) -> Dict:
        """
        Detect screen/print artifacts using FFT analysis
        
        Screens display faces with characteristic moire patterns (high-frequency noise).
        Printed photos have different frequency characteristics than real faces.
        
        Args:
            face_image: Cropped face image in BGR format
        
        Returns:
            Dictionary with reflection detection results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))
        
        # Apply FFT to detect high-frequency patterns
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Analyze frequency distribution
        h, w = magnitude_spectrum.shape
        center_region = magnitude_spectrum[h//4:3*h//4, w//4:3*w//4]
        outer_region = np.concatenate([
            magnitude_spectrum[:h//4, :].flatten(),
            magnitude_spectrum[3*h//4:, :].flatten(),
            magnitude_spectrum[:, :w//4].flatten(),
            magnitude_spectrum[:, 3*w//4:].flatten()
        ])
        
        center_energy = np.mean(center_region)
        outer_energy = np.mean(outer_region)
        
        # High outer energy relative to center suggests moire patterns
        energy_ratio = outer_energy / (center_energy + 1e-10)
        
        # Heuristic: screens typically have energy_ratio > 0.85
        is_screen = energy_ratio > 0.85
        
        return {
            'center_energy': float(center_energy),
            'outer_energy': float(outer_energy),
            'energy_ratio': float(energy_ratio),
            'is_screen': is_screen
        }
    
    def analyze_color_distribution(self, face_image: np.ndarray) -> Dict:
        """
        Analyze color distribution to detect unnatural color patterns
        
        Real faces have natural skin tone distribution.
        Printed/screen faces often have abnormal color histograms.
        
        Args:
            face_image: Cropped face image in BGR format
        
        Returns:
            Dictionary with color analysis results
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(face_image, cv2.COLOR_BGR2YCrCb)
        
        # Analyze saturation channel (real faces have natural saturation)
        saturation = hsv[:, :, 1]
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        
        # Analyze Cr channel (red-difference, important for skin detection)
        cr_channel = ycrcb[:, :, 1]
        cr_mean = np.mean(cr_channel)
        cr_std = np.std(cr_channel)
        
        # Natural skin tones have specific Cr ranges (133-173 typically)
        skin_cr_score = 1.0 if 133 <= cr_mean <= 173 else max(0, 1 - abs(cr_mean - 153) / 50)
        
        # Natural saturation should be moderate (not too high, not too low)
        sat_score = 1.0 if 30 <= sat_mean <= 150 else 0.5
        
        color_score = 0.6 * skin_cr_score + 0.4 * sat_score
        is_natural = color_score > 0.5
        
        return {
            'saturation_mean': float(sat_mean),
            'saturation_std': float(sat_std),
            'cr_mean': float(cr_mean),
            'cr_std': float(cr_std),
            'color_score': float(color_score),
            'is_natural': is_natural
        }
    
    def detect_blur(self, face_image: np.ndarray) -> Dict:
        """
        Detect if image is blurry (common in screen captures)
        
        Args:
            face_image: Cropped face image in BGR format
        
        Returns:
            Dictionary with blur detection results
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        
        # Laplacian variance method
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2).mean()
        
        # Thresholds (lower values = more blur)
        is_sharp = laplacian_var > 100 and gradient_mag > 20
        
        sharpness_score = min(1.0, laplacian_var / 500)
        
        return {
            'laplacian_variance': float(laplacian_var),
            'gradient_magnitude': float(gradient_mag),
            'sharpness_score': float(sharpness_score),
            'is_sharp': is_sharp
        }
    
    def check_liveness(self, face_image: np.ndarray, frames: List[np.ndarray] = None) -> Dict:
        results = {
            'is_live': False,
            'confidence': 0.0,
            'checks': {}
        }
        
        # 1. Texture analysis
        texture_result = self.analyze_texture(face_image)
        results['checks']['texture'] = texture_result
        
        # 2. Screen reflection detection
        reflection_result = self.detect_screen_reflection(face_image)
        results['checks']['reflection'] = reflection_result
        
        # 3. Color distribution analysis
        color_result = self.analyze_color_distribution(face_image)
        results['checks']['color'] = color_result
        
        # 4. Blur detection
        blur_result = self.detect_blur(face_image)
        results['checks']['blur'] = blur_result
        
        # Combined decision with weights
        weights = {
            'texture': 0.30,      # LBP texture patterns
            'reflection': 0.25,  # FFT moire detection
            'color': 0.25,       # Natural color distribution
            'blur': 0.20         # Sharpness check
        }
        
        confidence = (
            weights['texture'] * (1.0 if texture_result['is_real'] else 0.0) +
            weights['reflection'] * (1.0 if not reflection_result['is_screen'] else 0.0) +
            weights['color'] * (1.0 if color_result['is_natural'] else 0.0) +
            weights['blur'] * (1.0 if blur_result['is_sharp'] else 0.0)
        )
        
        results['confidence'] = float(confidence)
        results['is_live'] = confidence >= 0.5
        
        return results


# Singleton instance for reuse
_anti_spoofing_instance: Optional[AntiSpoofing] = None


def get_anti_spoofing() -> AntiSpoofing:
    """
    Get or create an AntiSpoofing singleton instance
    """
    global _anti_spoofing_instance
    if _anti_spoofing_instance is None:
        _anti_spoofing_instance = AntiSpoofing()
    return _anti_spoofing_instance
