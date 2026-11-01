"""
Stage 1 Training Script: FastPose-ViT Spacecraft Pose Estimator.

Trains Vision Transformer to directly regress 6DoF pose from cropped images
using apparent rotation and normalized translation targets, hardened against
space illumination and stray light.

References:
- FastPose-ViT: A Vision Transformer for Real-Time Spacecraft Pose Estimation (2025).
"""

import os
import argparse
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fastpose_vit import FastPoseViT
from models.apparent_rotation import compute_pose_errors
from data.dataset_loader import SpacecraftPoseDataset, generate_synthetic_verification_dataset
from simulation.space_augmentation import SpaceAugmentor


def train_fastpose(
    data_dir: str = "data/synthetic_sample",
    crop_size: int = 224,
    embed_dim: int = 192,
    depth: int = 4,
    num_heads: int = 4,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-4,
    output_dir: str = "checkpoints"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training FastPose-ViT on device: {device}")

    # Discover sequence directories
    seq_dirs = sorted(glob.glob(os.path.join(data_dir, "seq_*")))
    if len(seq_dirs) == 0:
        print(f"No sequences found in '{data_dir}'. Generating synthetic verification dataset...")
        generate_synthetic_verification_dataset(output_dir=data_dir, num_sequences=3, frames_per_sequence=25)
        seq_dirs = sorted(glob.glob(os.path.join(data_dir, "seq_*")))

    # Train/Val split
    split_idx = max(1, int(0.75 * len(seq_dirs)))
    train_seqs = seq_dirs[:split_idx]
    val_seqs = seq_dirs[split_idx:] if split_idx < len(seq_dirs) else seq_dirs[:1]

    # Camera parameters (640x480 default for sample, or 1920x1200 for full dataset)
    full_w, full_h = 640.0, 480.0
    fx, fy, cx, cy = 640.0, 640.0, 320.0, 240.0
    alpha = 1.0

    augmentor = SpaceAugmentor(
        bbox_jitter_ratio=0.10,
        spatial_aug_prob=0.50,
        harsh_glare_prob=0.35,
        specular_sat_prob=0.30
    )

    train_ds = SpacecraftPoseDataset(
        train_seqs, crop_size=crop_size, full_w=full_w, full_h=full_h,
        fx=fx, fy=fy, cx=cx, cy=cy, alpha=alpha, augment=True, augmentor=augmentor
    )
    val_ds = SpacecraftPoseDataset(
        val_seqs, crop_size=crop_size, full_w=full_w, full_h=full_h,
        fx=fx, fy=fy, cx=cx, cy=cy, alpha=alpha, augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset loaded: {len(train_ds)} train crops, {len(val_ds)} val crops.")

    # Model architecture
    model = FastPoseViT(
        img_size=crop_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads
    ).to(device)

    # Optimizer: AdamW / Muon without weight decay per FastPose-ViT Section 4.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    weights_path = os.path.join(output_dir, "fastpose_best.pt")

    print("\nStarting FastPose-ViT training...")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_rot = 0.0
        train_trans = 0.0

        for batch in train_loader:
            crops = batch["crop"].to(device)
            R_prime_gt = batch["R_prime_gt"].to(device)
            U_gt = batch["U_gt"].to(device)

            optimizer.zero_grad()
            r12_pred, U_pred = model(crops)
            loss_dict = model.compute_loss(r12_pred, U_pred, R_prime_gt, U_gt)
            loss = loss_dict["loss_total"]
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(crops)
            train_rot += loss_dict["loss_rot"].item() * len(crops)
            train_trans += loss_dict["loss_trans"].item() * len(crops)

        train_loss /= len(train_loader.dataset)
        train_rot /= len(train_loader.dataset)
        train_trans /= len(train_loader.dataset)

        # Validation with ESA Pose Error metrics
        model.eval()
        val_loss = 0.0
        all_ET = []
        all_ER = []

        with torch.no_grad():
            for batch in val_loader:
                crops = batch["crop"].to(device)
                bboxes = batch["bbox"].to(device)
                R_prime_gt = batch["R_prime_gt"].to(device)
                U_gt = batch["U_gt"].to(device)
                T_gt = batch["T_gt"].to(device)
                q_gt = batch["q_gt"].to(device)

                r12_pred, U_pred = model(crops)
                loss_dict = model.compute_loss(r12_pred, U_pred, R_prime_gt, U_gt)
                val_loss += loss_dict["loss_total"].item() * len(crops)

                # Predict 6DoF pose
                preds = model.predict_6dof(
                    crops, bboxes, full_w=full_w, full_h=full_h,
                    fx=fx, fy=fy, cx=cx, cy=cy, alpha=alpha
                )
                ET, ER = compute_pose_errors(preds["T_pred"], T_gt, preds["q_pred"], q_gt)
                all_ET.extend(ET.cpu().numpy().tolist())
                all_ER.extend(ER.cpu().numpy().tolist())

        val_loss /= len(val_loader.dataset)
        mean_ET = np.mean(all_ET) if len(all_ET) > 0 else 0.0
        mean_ER_deg = np.degrees(np.mean(all_ER)) if len(all_ER) > 0 else 0.0

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)

        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} (Rot: {train_rot:.4f}, Trans: {train_trans:.4f}) | Val Loss: {val_loss:.4f} | Mean ET: {mean_ET:.3f}m | Mean ER: {mean_ER_deg:.2f}°")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.1f}s. Saved best model to '{weights_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FastPose-ViT 6DoF Pose Estimator")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_sample")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_fastpose(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
