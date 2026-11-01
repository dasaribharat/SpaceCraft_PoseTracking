"""
Space Stray-Light Simulation Models (Sun et al., Aerospace 2023).

Simulates space optical interferences:
1. Moonlight glare simulation (off-axis angle sigma < 30 deg)
2. Earth atmospheric radiation / limb scatter (off-axis angle theta < 22 deg)
3. Background starlight point spread function (PSF) diffraction

Reference:
- Sun et al., 'A Vision-Based Pose Estimation of a Non-Cooperative Target Based
  on a Self-Supervised Transformer Network', Aerospace 2023, 10, 997.
  Section 3.2 'Stray Light Simulation'.
"""

import numpy as np
import cv2
import torch
from typing import Tuple, Optional


class SpaceStrayLightSimulator:
    """
    Simulates stray light artifacts on space imagery per Sun et al. (2023).
    """

    @staticmethod
    def apply_moonlight_glare(
        image: np.ndarray,
        moon_angle_deg: float = 20.0,
        center: Optional[Tuple[float, float]] = None,
        intensity: float = 0.7
    ) -> np.ndarray:
        """
        Simulates moonlight stray light (Section 3.2.1).
        When the moonlight angle sigma < 30 deg, stray light enters the CCD sensor,
        generating a bright diffusive Gaussian halo spreading outward from the Moon direction.

        Args:
            image: HxW or HxWxC uint8/float32 image [0, 255] or [0, 1]
            moon_angle_deg: Angle between camera line-of-sight and Moon vector (deg)
            center: (x, y) location of the Moon projected source (can be slightly off-sensor)
            intensity: Peak glare intensity [0, 1]

        Returns:
            Degraded image with moonlight stray light interference
        """
        if moon_angle_deg >= 30.0:
            return image.copy()

        h, w = image.shape[:2]
        # Calculate glare strength based on angle (stronger when angle is smaller)
        angle_factor = (30.0 - moon_angle_deg) / 30.0  # (0, 1]
        effective_intensity = intensity * angle_factor

        if center is None:
            # Place near an edge or corner to simulate off-axis entry
            center = (np.random.uniform(-0.1 * w, 1.1 * w), np.random.uniform(-0.1 * h, 0.3 * h))

        cx, cy = center
        y_grid, x_grid = np.ogrid[:h, :w]

        # Spread radius scales with field of view
        sigma = 0.35 * max(w, h)
        gaussian_halo = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2.0 * sigma ** 2))

        # Add secondary lens flare rings
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        ring = np.exp(-((dist - 0.4 * max(w, h)) ** 2) / (2.0 * (0.05 * max(w, h)) ** 2)) * 0.25

        glare_mask = effective_intensity * (gaussian_halo + ring)
        glare_mask = np.clip(glare_mask, 0.0, 1.0)

        is_uint8 = (image.dtype == np.uint8)
        img_float = image.astype(np.float32) / 255.0 if is_uint8 else image.copy()

        if img_float.ndim == 3:
            glare_mask = glare_mask[:, :, np.newaxis]

        # Moonlight is cool pale-white
        moon_tint = np.array([0.9, 0.95, 1.0], dtype=np.float32) if img_float.ndim == 3 else 1.0
        result = img_float + glare_mask * moon_tint
        result = np.clip(result, 0.0, 1.0)

        return (result * 255.0).astype(np.uint8) if is_uint8 else result

    @staticmethod
    def apply_earth_atmosphere_radiation(
        image: np.ndarray,
        off_axis_angle_deg: float = 15.0,
        limb_position: str = "bottom",
        intensity: float = 0.65
    ) -> np.ndarray:
        """
        Simulates Earth atmospheric radiation / airlight (Section 3.2.2).
        When the off-axis angle theta < 22 deg from the atmospheric boundary,
        Earth airlight scattered into the camera creates broad, curved gradient glare.

        Args:
            image: HxW or HxWxC image
            off_axis_angle_deg: Angle theta to atmospheric boundary (deg)
            limb_position: 'bottom', 'top', 'left', or 'right'
            intensity: Peak stray light luminance

        Returns:
            Degraded image with Earth atmospheric scatter
        """
        if off_axis_angle_deg >= 22.0:
            return image.copy()

        h, w = image.shape[:2]
        angle_factor = (22.0 - off_axis_angle_deg) / 22.0
        effective_intensity = intensity * angle_factor

        y_grid, x_grid = np.mgrid[:h, :w]

        # Atmospheric limb simulated by multi-source grid approximation (Eq. from Fig 5)
        if limb_position == "bottom":
            dist_to_boundary = (h - 1 - y_grid) / float(h)
        elif limb_position == "top":
            dist_to_boundary = y_grid / float(h)
        elif limb_position == "left":
            dist_to_boundary = x_grid / float(w)
        else:
            dist_to_boundary = (w - 1 - x_grid) / float(w)

        # Exponential fall-off of atmospheric airlight
        scatter = np.exp(-dist_to_boundary * 3.5) * effective_intensity
        scatter = np.clip(scatter, 0.0, 1.0)

        is_uint8 = (image.dtype == np.uint8)
        img_float = image.astype(np.float32) / 255.0 if is_uint8 else image.copy()

        # Earth atmosphere airlight has distinct Rayleigh cyan-blue tint
        if img_float.ndim == 3:
            earth_tint = np.array([0.4, 0.7, 1.0], dtype=np.float32)  # RGB
            scatter_3d = scatter[:, :, np.newaxis] * earth_tint
            result = img_float + scatter_3d
        else:
            result = img_float + scatter

        result = np.clip(result, 0.0, 1.0)
        return (result * 255.0).astype(np.uint8) if is_uint8 else result

    @staticmethod
    def apply_starlight_psf(
        image: np.ndarray,
        num_stars: int = 40,
        max_psf_sigma: float = 1.8
    ) -> np.ndarray:
        """
        Simulates background starlight with Gaussian Point Spread Function (PSF) (Section 3.2.3, Eq. 1).
        Point sources form diffraction spots described by 2D Gaussian PSF.

        Args:
            image: HxW or HxWxC image
            num_stars: Number of star background points
            max_psf_sigma: Optical diffraction radius sigma

        Returns:
            Image with background star field
        """
        h, w = image.shape[:2]
        star_layer = np.zeros((h, w), dtype=np.float32)

        for _ in range(num_stars):
            cx = np.random.uniform(0, w)
            cy = np.random.uniform(0, h)
            brightness = np.random.uniform(0.3, 1.0)
            sigma1 = np.random.uniform(0.6, max_psf_sigma)
            sigma2 = sigma1 * np.random.uniform(0.9, 1.1)

            # Local PSF bounding box
            rad = int(math_ceil(3.0 * max(sigma1, sigma2)))
            x0 = max(0, int(cx) - rad)
            x1 = min(w, int(cx) + rad + 1)
            y0 = max(0, int(cy) - rad)
            y1 = min(h, int(cy) + rad + 1)

            if x1 <= x0 or y1 <= y0:
                continue

            yg, xg = np.ogrid[y0:y1, x0:x1]
            psf = brightness * np.exp(-0.5 * (((xg - cx) / sigma1) ** 2 + ((yg - cy) / sigma2) ** 2))
            star_layer[y0:y1, x0:x1] += psf

        star_layer = np.clip(star_layer, 0.0, 1.0)

        is_uint8 = (image.dtype == np.uint8)
        img_float = image.astype(np.float32) / 255.0 if is_uint8 else image.copy()

        if img_float.ndim == 3:
            star_layer = star_layer[:, :, np.newaxis]

        result = np.clip(img_float + star_layer, 0.0, 1.0)
        return (result * 255.0).astype(np.uint8) if is_uint8 else result


def math_ceil(x: float) -> int:
    return int(np.ceil(x))
