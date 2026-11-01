"""
Dataset Loader and Verification Generator for FastPose-ViT.

Complies strictly with DATASET_SPECIFICATION.md:
- Loads sequence frames, bounding boxes, 3D translations, and rotations
- Applies SpaceAugmentor (space illumination, bbox jitter, in-plane spatial rotations)
- Computes target apparent rotation R' and normalized translation U
- Crops spacecraft patch, resizes to (224, 224), and converts to normalized PyTorch tensor

Also provides a self-contained synthetic space dataset generator for immediate verification.
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional, Tuple

from models.apparent_rotation import (
    true_to_apparent_rotation,
    compute_normalized_translation_targets,
    quaternion_to_matrix
)
from simulation.space_augmentation import SpaceAugmentor
from simulation.dynamics_generator import SpacecraftDynamicsSimulator


class SpacecraftPoseDataset(Dataset):
    """
    PyTorch Dataset loading spacecraft video sequences and annotations.
    """
    def __init__(
        self,
        sequence_dirs: List[str],
        crop_size: int = 224,
        full_w: float = 1920.0,
        full_h: float = 1200.0,
        fx: float = 1920.0,
        fy: float = 1920.0,
        cx: float = 960.0,
        cy: float = 600.0,
        alpha: float = 1.6,
        augment: bool = True,
        augmentor: Optional[SpaceAugmentor] = None
    ):
        self.crop_size = crop_size
        self.full_w = full_w
        self.full_h = full_h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.alpha = alpha
        self.augment = augment
        self.augmentor = augmentor if augmentor is not None else SpaceAugmentor()

        # ImageNet normalization statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.samples = []
        for seq_dir in sequence_dirs:
            ann_file = os.path.join(seq_dir, "annotations.json")
            if not os.path.exists(ann_file):
                continue
            with open(ann_file, "r") as f:
                data = json.load(f)

            for frame in data.get("frames", []):
                img_path = os.path.join(seq_dir, frame["image_file"])
                if os.path.exists(img_path):
                    self.samples.append({
                        "img_path": img_path,
                        "bbox": np.array(frame["bounding_box_xyxy"], dtype=np.float32),
                        "translation": np.array(frame["translation_xyz"], dtype=np.float32),
                        "rotation_matrix": np.array(frame["rotation_matrix_3x3"], dtype=np.float32),
                        "quaternion": np.array(frame["quaternion_wxyz"], dtype=np.float32)
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = cv2.imread(sample["img_path"])
        if image is None:
            # Fallback black image
            image = np.zeros((int(self.full_h), int(self.full_w), 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox = sample["bbox"].copy()
        R_true = sample["rotation_matrix"].copy()
        T_true = sample["translation"].copy()

        # Apply spatial in-plane rotation and illumination augmentations if enabled
        if self.augment:
            # 1. Spatial rotation with analytical pose update
            image, bbox, R_true, T_true = self.augmentor.apply_spatial_inplane_rotation(
                image, bbox, R_true, T_true
            )
            # 2. Bounding-box perturbation (+/- 10%)
            bbox = self.augmentor.jitter_bounding_box(bbox, int(self.full_w), int(self.full_h))
            # 3. Space illumination degradations
            image = self.augmentor.apply_illumination_augmentations(image)

        # Crop spacecraft from full frame
        xmin, ymin, xmax, ymax = [int(v) for v in bbox]
        xmin = max(0, min(xmin, int(self.full_w) - 10))
        ymin = max(0, min(ymin, int(self.full_h) - 10))
        xmax = max(xmin + 10, min(xmax, int(self.full_w)))
        ymax = max(ymin + 10, min(ymax, int(self.full_h)))

        crop = image[ymin:ymax, xmin:xmax]
        crop_resized = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)

        # Normalize crop tensor: (C, H, W)
        crop_norm = (crop_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        crop_tensor = torch.from_numpy(crop_norm.transpose(2, 0, 1)).float()

        # Compute regression targets
        T_torch = torch.from_numpy(T_true).float()
        R_torch = torch.from_numpy(R_true).float()
        bbox_torch = torch.from_numpy(bbox).float()

        # Apparent rotation R' = Delta_R(T) * R_true
        R_prime = true_to_apparent_rotation(R_torch.unsqueeze(0), T_torch.unsqueeze(0)).squeeze(0)

        # Normalized translation U = (Ux, Uy, Uz)
        U_target = compute_normalized_translation_targets(
            T_torch.unsqueeze(0), bbox_torch.unsqueeze(0),
            full_w=self.full_w, full_h=self.full_h,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            alpha=self.alpha
        ).squeeze(0)

        return {
            "crop": crop_tensor,
            "bbox": bbox_torch,
            "R_prime_gt": R_prime,
            "U_gt": U_target,
            "T_gt": T_torch,
            "R_gt": R_torch,
            "q_gt": torch.from_numpy(sample["quaternion"]).float()
        }


def generate_synthetic_verification_dataset(
    output_dir: str = "data/synthetic_sample",
    num_sequences: int = 2,
    frames_per_sequence: int = 50,
    img_w: int = 640,
    img_h: int = 480
):
    """
    Renders synthetic tumbling spacecraft sequences with 3D projection,
    satellite solar panels, MLI foil, and dark space background.
    """
    os.makedirs(output_dir, exist_ok=True)
    fx = float(img_w)
    fy = float(img_w)
    cx = img_w * 0.5
    cy = img_h * 0.5
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Global intrinsics file
    intrinsics = {
        "image_width": img_w,
        "image_height": img_h,
        "aspect_ratio_correction_factor": 1.0,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "fps": 10.0
    }
    with open(os.path.join(output_dir, "camera_intrinsics.json"), "w") as f:
        json.dump(intrinsics, f, indent=2)

    sim = SpacecraftDynamicsSimulator(dt=0.1)

    # 3D vertices of a simple satellite model (central bus + 2 solar wings)
    bus_corners = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
        # Left solar panel
        [-1.8, -0.3, 0.0], [-0.5, -0.3, 0.0], [-0.5, 0.3, 0.0], [-1.8, 0.3, 0.0],
        # Right solar panel
        [ 0.5, -0.3, 0.0], [ 1.8, -0.3, 0.0], [ 1.8, 0.3, 0.0], [ 0.5, 0.3, 0.0]
    ], dtype=np.float32)

    for seq_idx in range(1, num_sequences + 1):
        seq_name = f"seq_{seq_idx:03d}"
        seq_dir = os.path.join(output_dir, seq_name)
        img_dir = os.path.join(seq_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        data = sim.simulate_sequence(num_frames=frames_per_sequence, measurement_noise_std=0.01)
        frames_meta = []

        for k in range(frames_per_sequence):
            q_k = data["quaternions_true"][k]
            T_k = data["translations"][k]
            v_k = data["linear_velocities"][k]
            w_k = data["angular_velocities"][k]

            # Convert quaternion to rotation matrix
            R_k = quaternion_to_matrix(torch.from_numpy(q_k).unsqueeze(0)).squeeze(0).numpy()

            # Transform 3D vertices to camera frame: P_c = R * P_b + T
            P_c = (R_k @ bus_corners.T).T + T_k

            # Project onto 2D image plane: [u, v]^T = [fx * X / Z + cx, fy * Y / Z + cy]
            z_safe = np.clip(P_c[:, 2], 1.0, None)
            u = (fx * P_c[:, 0] / z_safe) + cx
            v = (fy * P_c[:, 1] / z_safe) + cy

            # Compute bounding box
            xmin = max(0, float(np.min(u) - 15))
            xmax = min(img_w, float(np.max(u) + 15))
            ymin = max(0, float(np.min(v) - 15))
            ymax = min(img_h, float(np.max(v) + 15))
            bbox = [xmin, ymin, xmax, ymax]

            # Render synthetic frame
            frame_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

            # Draw central spacecraft bus with gold/foil specular texture
            bus_pts = np.int32(np.stack([u[:8], v[:8]], axis=-1))
            hull = cv2.convexHull(bus_pts)
            cv2.fillConvexPoly(frame_img, hull, (210, 180, 80)) # gold MLI foil
            cv2.polylines(frame_img, [hull], True, (255, 230, 130), 2)

            # Draw solar wings in blue
            wing1 = np.int32(np.stack([u[8:12], v[8:12]], axis=-1))
            wing2 = np.int32(np.stack([u[12:16], v[12:16]], axis=-1))
            cv2.fillConvexPoly(frame_img, wing1, (160, 80, 30)) # blue solar cell
            cv2.polylines(frame_img, [wing1], True, (200, 120, 50), 1)
            cv2.fillConvexPoly(frame_img, wing2, (160, 80, 30))
            cv2.polylines(frame_img, [wing2], True, (200, 120, 50), 1)

            # Save frame
            img_filename = f"frame_{k:06d}.png"
            img_rel_path = os.path.join("images", img_filename)
            cv2.imwrite(os.path.join(img_dir, img_filename), cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))

            frames_meta.append({
                "frame_idx": k,
                "timestamp_s": float(data["timestamps"][k]),
                "image_file": img_rel_path,
                "visible": True,
                "bounding_box_xyxy": bbox,
                "translation_xyz": T_k.tolist(),
                "quaternion_wxyz": q_k.tolist(),
                "rotation_matrix_3x3": R_k.tolist(),
                "linear_velocity_xyz": v_k.tolist(),
                "angular_velocity_body_xyz": w_k.tolist()
            })

        annotation = {
            "sequence_id": seq_name,
            "num_frames": frames_per_sequence,
            "inertia_properties": {
                "inertia_ratios": data["inertia_ratios"].tolist()
            },
            "frames": frames_meta
        }
        with open(os.path.join(seq_dir, "annotations.json"), "w") as f:
            json.dump(annotation, f, indent=2)

    print(f"Generated synthetic verification dataset with {num_sequences} sequences in '{output_dir}'.")
