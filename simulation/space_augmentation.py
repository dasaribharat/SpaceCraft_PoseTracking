"""
Space Illumination and Spatial Augmentations for FastPose-ViT.

Implements:
1. Spacecraft-specific illumination augmentations:
   - High-contrast direct solar glare & deep shadowing
   - MLI (Multi-Layer Insulation) specular reflection saturation
   - Space CCD sensor shot noise & cosmic ray hits
2. Bounding-Box Jittering (FastPose-ViT Section 3.2):
   - Random expansion/contraction of bbox boundaries by up to +/- 10%
3. Spatial In-Plane Rotation with exact analytical 3D pose transformation (Eq. 1-2):
   - R_new = Rz(theta) * R
   - T_new = Rz(theta) * T

References:
- FastPose-ViT (2512.09792v1.pdf), Section 3.2 'Data Augmentation Pipeline'.
"""

import numpy as np
import cv2
import torch
from typing import Tuple, Dict, Any, Optional
from simulation.space_stray_light import SpaceStrayLightSimulator


class SpaceAugmentor:
    """
    Comprehensive data augmentation pipeline hardening FastPose-ViT against space imaging conditions.
    """
    def __init__(
        self,
        bbox_jitter_ratio: float = 0.10,
        spatial_aug_prob: float = 0.50,
        harsh_glare_prob: float = 0.40,
        specular_sat_prob: float = 0.35,
        shadow_darkening_prob: float = 0.40,
        sensor_noise_prob: float = 0.30,
        stray_light_prob: float = 0.30
    ):
        self.bbox_jitter_ratio = bbox_jitter_ratio
        self.spatial_aug_prob = spatial_aug_prob
        self.harsh_glare_prob = harsh_glare_prob
        self.specular_sat_prob = specular_sat_prob
        self.shadow_darkening_prob = shadow_darkening_prob
        self.sensor_noise_prob = sensor_noise_prob
        self.stray_light_prob = stray_light_prob

    def apply_illumination_augmentations(self, image: np.ndarray) -> np.ndarray:
        """
        Applies space-specific illumination degradations.
        """
        img = image.astype(np.float32) / 255.0

        # 1. Harsh Sun Glare / High Contrast
        if np.random.rand() < self.harsh_glare_prob:
            gamma = np.random.uniform(1.5, 3.0)  # Crushes dark shadows into pitch black
            contrast = np.random.uniform(1.2, 1.8)
            img = np.power(img, gamma) * contrast

        # 2. MLI Specular Foil Saturation
        if np.random.rand() < self.specular_sat_prob:
            # High intensity regions saturate completely (pure white blowout)
            sat_threshold = np.random.uniform(0.65, 0.85)
            sat_mask = img > sat_threshold
            img[sat_mask] = 1.0

        # 3. Deep Shadow / Eclipse penumbra
        if np.random.rand() < self.shadow_darkening_prob:
            brightness = np.random.uniform(0.4, 0.9)
            img = img * brightness

        # 4. Sensor Shot Noise
        if np.random.rand() < self.sensor_noise_prob:
            noise = np.random.normal(0, np.random.uniform(0.01, 0.04), img.shape)
            img = img + noise

        # 5. Stray Light (Sun et al.)
        if np.random.rand() < self.stray_light_prob:
            sub_type = np.random.choice(["moon", "earth", "stars"])
            if sub_type == "moon":
                img = SpaceStrayLightSimulator.apply_moonlight_glare(
                    img, moon_angle_deg=np.random.uniform(5.0, 28.0), intensity=np.random.uniform(0.4, 0.8)
                )
            elif sub_type == "earth":
                pos = np.random.choice(["bottom", "top", "left", "right"])
                img = SpaceStrayLightSimulator.apply_earth_atmosphere_radiation(
                    img, off_axis_angle_deg=np.random.uniform(5.0, 20.0), limb_position=pos
                )
            else:
                img = SpaceStrayLightSimulator.apply_starlight_psf(img, num_stars=np.random.randint(15, 50))

        img = np.clip(img, 0.0, 1.0)
        return (img * 255.0).astype(np.uint8)

    def jitter_bounding_box(
        self,
        bbox: np.ndarray,
        img_w: int,
        img_h: int
    ) -> np.ndarray:
        """
        Perturbs bounding box by up to +/- 10% on each edge (FastPose-ViT Section 3.2).
        bbox: [xmin, ymin, xmax, ymax]
        """
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin

        dw_min = np.random.uniform(-self.bbox_jitter_ratio, self.bbox_jitter_ratio) * w
        dw_max = np.random.uniform(-self.bbox_jitter_ratio, self.bbox_jitter_ratio) * w
        dh_min = np.random.uniform(-self.bbox_jitter_ratio, self.bbox_jitter_ratio) * h
        dh_max = np.random.uniform(-self.bbox_jitter_ratio, self.bbox_jitter_ratio) * h

        new_xmin = np.clip(xmin + dw_min, 0, img_w - 10)
        new_ymin = np.clip(ymin + dh_min, 0, img_h - 10)
        new_xmax = np.clip(xmax + dw_max, new_xmin + 10, img_w)
        new_ymax = np.clip(ymax + dh_max, new_ymin + 10, img_h)

        return np.array([new_xmin, new_ymin, new_xmax, new_ymax], dtype=np.float32)

    def apply_spatial_inplane_rotation(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        R_matrix: np.ndarray,
        T_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies 2D in-plane rotation around image center with exact 3D pose analytical update.
        FastPose-ViT Eq. (1)-(2):
            R_new = Rz(theta) * R
            T_new = Rz(theta) * T
        """
        if np.random.rand() > self.spatial_aug_prob:
            return image, bbox, R_matrix, T_vector

        # Select either minor rotation [-20, 20] deg or major rotation [160, 200] deg
        if np.random.rand() < 0.5:
            theta_deg = np.random.uniform(-20.0, 20.0)
        else:
            theta_deg = np.random.uniform(160.0, 200.0)

        theta_rad = np.radians(theta_deg)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        # 3D rotation matrix around camera Z-axis: Rz(theta)
        Rz = np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float32)

        # Update 3D pose labels
        R_new = np.matmul(Rz, R_matrix.astype(np.float32))
        T_new = np.matmul(Rz, T_vector.astype(np.float32))

        # Rotate image around image center
        h, w = image.shape[:2]
        center = (w * 0.5, h * 0.5)
        M_rot = cv2.getRotationMatrix2D(center, -theta_deg, 1.0)  # negative because cv2 is CCW
        rotated_image = cv2.warpAffine(image, M_rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Rotate bounding box corners to compute new axis-aligned bounding box
        xmin, ymin, xmax, ymax = bbox
        corners = np.array([
            [xmin, ymin, 1.0],
            [xmax, ymin, 1.0],
            [xmax, ymax, 1.0],
            [xmin, ymax, 1.0]
        ]).T  # (3, 4)

        rotated_corners = np.matmul(M_rot, corners)  # (2, 4)
        new_xmin = np.clip(np.min(rotated_corners[0, :]), 0, w - 10)
        new_xmax = np.clip(np.max(rotated_corners[0, :]), new_xmin + 10, w)
        new_ymin = np.clip(np.min(rotated_corners[1, :]), 0, h - 10)
        new_ymax = np.clip(np.max(rotated_corners[1, :]), new_ymin + 10, h)
        new_bbox = np.array([new_xmin, new_ymin, new_xmax, new_ymax], dtype=np.float32)

        return rotated_image, new_bbox, R_new, T_new
