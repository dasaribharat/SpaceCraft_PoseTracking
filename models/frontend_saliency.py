"""
Frontend Saliency Extraction for Harsh Scenes (Shi & Ulrich).

Extracts non-cooperative satellite body from cluttered Earth backgrounds and
harsh space lighting conditions using multi-scale Difference-of-Gaussians (DoG),
Spectral Residual saliency, and morphological graph segmentation.

References:
- Jian-Feng Shi, Steve Ulrich.
  'Uncooperative Spacecraft Pose Estimation using Monocular Monochromatic Images', 2021.
  Section II 'Image-driven Saliency Generation' and Algorithm 3.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any


class SaliencyFrontEnd:
    """
    Image-driven foreground extraction that does not rely on neural network priors,
    providing robust spacecraft initialization and fallback in cluttered space environments.
    """
    def __init__(
        self,
        dog_sigma1: float = 1.0,
        dog_sigma2: float = 3.0,
        saliency_threshold_factor: float = 0.5,
        min_target_area: int = 150
    ):
        self.dog_sigma1 = dog_sigma1
        self.dog_sigma2 = dog_sigma2
        self.saliency_threshold_factor = saliency_threshold_factor
        self.min_target_area = min_target_area

    def compute_spectral_residual_saliency(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Spectral Residual (SR) saliency generation (Shi & Ulrich Section II.B & Hou & Zhang):
        1. 2D Fourier Transform of grayscale image
        2. Compute log amplitude spectrum: L(f) = log(A(f))
        3. Compute Spectral Residual: R(f) = L(f) - h_n(f) * L(f) (average filtered)
        4. Inverse 2D FFT to spatial domain and Gaussian smooth -> Saliency Map
        """
        h, w = gray_image.shape
        # Compute 2D DFT
        dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        phase = cv2.phase(dft_shift[:, :, 0], dft_shift[:, :, 1])

        # Log amplitude spectrum
        log_amplitude = np.log(magnitude + 1e-6)

        # Average filter over log amplitude (3x3 mean filter)
        avg_log = cv2.blur(log_amplitude, (3, 3))
        spectral_residual = log_amplitude - avg_log

        # Reconstruct complex spectrum
        res_real = np.exp(spectral_residual) * np.cos(phase)
        res_imag = np.exp(spectral_residual) * np.sin(phase)
        res_complex = np.stack([res_real, res_imag], axis=-1)

        # Inverse DFT
        idft_shift = np.fft.ifftshift(res_complex)
        idft = cv2.idft(idft_shift)
        saliency = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

        # Gaussian smooth to produce final saliency map
        saliency = cv2.GaussianBlur(saliency, (7, 7), 2.5)
        # Normalize to [0, 1]
        s_min, s_max = saliency.min(), saliency.max()
        if s_max - s_min > 1e-6:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        return saliency

    def compute_dog_saliency(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Difference of Gaussians (DoG) high-frequency response (Shi & Ulrich Eq. 1).
        """
        g1 = cv2.GaussianBlur(gray_image.astype(np.float32), (0, 0), self.dog_sigma1)
        g2 = cv2.GaussianBlur(gray_image.astype(np.float32), (0, 0), self.dog_sigma2)
        dog = np.abs(g1 - g2)

        d_min, d_max = dog.min(), dog.max()
        if d_max - d_min > 1e-6:
            dog = (dog - d_min) / (d_max - d_min)
        else:
            dog = np.zeros_like(dog)
        return dog

    def extract_satellite_roi(
        self,
        image: np.ndarray,
        fallback_bbox: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Segments spacecraft and estimates the bounding box [xmin, ymin, xmax, ymax].

        Args:
            image: HxW or HxWxC uint8 image
            fallback_bbox: Optional prior bounding box from detector

        Returns:
            Dict containing:
              - 'bbox': [xmin, ymin, xmax, ymax]
              - 'saliency_map': float32 [0, 1]
              - 'binary_mask': uint8 mask [0, 255]
              - 'success': bool
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[2] == 3 else image[:, :, 0]
        else:
            gray = image.copy()

        h, w = gray.shape

        # Combine Spectral Residual with Difference of Gaussians (HiSafe combination)
        sr_sal = self.compute_spectral_residual_saliency(gray)
        dog_sal = self.compute_dog_saliency(gray)
        combined_sal = 0.6 * sr_sal + 0.4 * dog_sal

        # Thresholding (Otsu + adaptive floor)
        sal_uint8 = (combined_sal * 255).astype(np.uint8)
        otsu_thresh, _ = cv2.threshold(sal_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_val = max(int(otsu_thresh * self.saliency_threshold_factor), 35)
        _, binary = cv2.threshold(sal_uint8, threshold_val, 255, cv2.THRESH_BINARY)

        # Morphological closing to bridge fragmented solar arrays and antenna masts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

        best_bbox = None
        max_area = 0

        # Background is label 0; find largest non-background component
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > self.min_target_area and area > max_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]

                # Filter out border artifacts spanning the full frame
                if bw > 0.95 * w and bh > 0.95 * h:
                    continue

                best_bbox = np.array([x, y, x + bw, y + bh], dtype=np.float32)
                max_area = area

        if best_bbox is not None:
            # Add small 5% margin
            xmin, ymin, xmax, ymax = best_bbox
            bw = xmax - xmin
            bh = ymax - ymin
            margin_x = bw * 0.08
            margin_y = bh * 0.08
            best_bbox = np.array([
                max(0, xmin - margin_x),
                max(0, ymin - margin_y),
                min(w, xmax + margin_x),
                min(h, ymax + margin_y)
            ], dtype=np.float32)
            success = True
        else:
            best_bbox = fallback_bbox if fallback_bbox is not None else np.array([0, 0, w, h], dtype=np.float32)
            success = False

        return {
            "bbox": best_bbox,
            "saliency_map": combined_sal,
            "binary_mask": closed,
            "success": success
        }
