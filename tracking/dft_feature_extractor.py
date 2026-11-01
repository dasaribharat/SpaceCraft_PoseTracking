"""
Stage 3: Characteristic Frequency Feature Extractor via Discrete Fourier Transform (DFT).

Compresses 200-500 quaternion frames into a 15-dimensional characteristic frequency
feature vector for inertia ratio regression.

References:
- Ma et al., 'Robust attitude estimation of rotating space debris based on
  virtual observations of neural network', 2021. Section 3.1 'Discrete Fourier transformation'.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Dict, Any, Optional


class QuaternionDFTFeatureExtractor:
    """
    Extracts 15 characteristic frequency parameters from quaternion time series:
    - 3 dominant tumbling frequencies (f1, f2, f3)
    - 12 spectral peak heights (p_ij for 4 quaternion channels i=0..3 at frequencies fj)
    Total vector: xc = [f1, f2, f3, p01, p02, p03, p11, p12, p13, p21, p22, p23, p31, p32, p33]^T in R^15.
    """
    def __init__(self, dt: float = 0.1, min_freq: float = 0.01, max_freq: float = 2.0):
        self.dt = dt
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.mean_vec: Optional[np.ndarray] = None
        self.std_vec: Optional[np.ndarray] = None
        self.cholesky_inv: Optional[np.ndarray] = None

    def extract_features_single(self, quat_series: np.ndarray) -> np.ndarray:
        """
        Extracts 15-dim feature vector from a single sequence of quaternions.
        Args:
            quat_series: (N, 4) quaternion sequence [qw, qx, qy, qz]
        Returns:
            xc: (15,) characteristic frequency feature vector
        """
        N = len(quat_series)
        freqs = np.fft.rfftfreq(N, d=self.dt)  # (N//2 + 1,)
        # Filter frequencies within [min_freq, max_freq]
        valid_freq_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)

        # Compute FFT magnitude for each component
        fft_mags = np.zeros((4, len(freqs)), dtype=np.float64)
        for i in range(4):
            # Subtract mean to remove DC component
            signal = quat_series[:, i] - np.mean(quat_series[:, i])
            # Apply Hanning window to suppress spectral leakage
            windowed = signal * np.hanning(N)
            fft_vals = np.fft.rfft(windowed)
            fft_mags[i] = np.abs(fft_vals) / N

        # Combined spectral energy across all 4 quaternion components
        total_energy = np.sum(fft_mags ** 2, axis=0)
        total_energy[~valid_freq_mask] = 0.0

        # Find dominant peaks in total energy spectrum
        peaks, props = find_peaks(total_energy, distance=max(1, int(0.02 / (freqs[1] - freqs[0]))))

        if len(peaks) >= 3:
            # Sort peaks by energy descending
            peak_heights = total_energy[peaks]
            top_3_idx = np.argsort(peak_heights)[::-1][:3]
            top_3_peaks = peaks[top_3_idx]
            # Sort the 3 frequencies ascending: f1 < f2 < f3
            top_3_freqs = np.sort(freqs[top_3_peaks])
        elif len(peaks) > 0:
            top_freqs = freqs[peaks]
            pad = np.array([top_freqs[-1] * 1.5, top_freqs[-1] * 2.0])
            top_3_freqs = np.sort(np.concatenate([top_freqs, pad])[:3])
        else:
            # Fallback if no peaks detected (very slow or stationary motion)
            top_3_freqs = np.array([0.05, 0.10, 0.15], dtype=np.float64)

        f1, f2, f3 = top_3_freqs[0], top_3_freqs[1], top_3_freqs[2]

        # Extract peak heights p_ij for each quaternion channel i=0..3 at the 3 frequencies
        # Find closest frequency bin indices
        bin1 = np.argmin(np.abs(freqs - f1))
        bin2 = np.argmin(np.abs(freqs - f2))
        bin3 = np.argmin(np.abs(freqs - f3))

        peak_heights = []
        for i in range(4):
            p_i1 = float(fft_mags[i, bin1])
            p_i2 = float(fft_mags[i, bin2])
            p_i3 = float(fft_mags[i, bin3])
            peak_heights.extend([p_i1, p_i2, p_i3])

        # Assemble 15-dim feature vector: [f1, f2, f3, p01..p33]
        xc = np.array([f1, f2, f3] + peak_heights, dtype=np.float32)
        return xc

    def extract_batch(self, quat_batch: np.ndarray) -> np.ndarray:
        """
        Extracts features for a batch of sequences: (B, N, 4) -> (B, 15).
        """
        B = quat_batch.shape[0]
        xc_batch = np.zeros((B, 15), dtype=np.float32)
        for b in range(B):
            xc_batch[b] = self.extract_features_single(quat_batch[b])
        return xc_batch

    def fit_normalizer(self, xc_batch: np.ndarray):
        """
        Fits mean vector mu and Cholesky decomposition sigma of covariance
        per Ma et al. Eq. (6)-(8):
            \tilde{x}_c = \sigma^{-1} (x_c - \mu)
        """
        self.mean_vec = np.mean(xc_batch, axis=0)
        cov = np.cov(xc_batch, rowvar=False)
        # Regularize covariance to ensure positive definiteness
        cov += np.eye(15) * 1e-4
        L = np.linalg.cholesky(cov)
        self.cholesky_inv = np.linalg.inv(L)
        self.std_vec = np.std(xc_batch, axis=0) + 1e-6

    def normalize(self, xc: np.ndarray) -> np.ndarray:
        """
        Applies whitening normalization to xc.
        """
        if self.mean_vec is None:
            return xc
        diff = xc - self.mean_vec
        if self.cholesky_inv is not None:
            if diff.ndim == 1:
                return np.dot(self.cholesky_inv, diff).astype(np.float32)
            else:
                return np.dot(diff, self.cholesky_inv.T).astype(np.float32)
        else:
            return (diff / self.std_vec).astype(np.float32)

    def save_stats(self, filepath: str):
        np.savez(
            filepath,
            mean_vec=self.mean_vec,
            std_vec=self.std_vec,
            cholesky_inv=self.cholesky_inv
        )

    def load_stats(self, filepath: str):
        data = np.load(filepath)
        self.mean_vec = data["mean_vec"]
        self.std_vec = data["std_vec"]
        self.cholesky_inv = data["cholesky_inv"]
