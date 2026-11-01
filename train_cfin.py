"""
Stage 3 Training Script: Characteristic Frequency Input Network (CFIN).

Generates the cheap rendering-free dataset of quaternion sequences via Euler dynamics,
extracts 15-dim characteristic frequency features using DFT, normalizes them via
Cholesky decomposition, and trains the CFIN network to predict inertia ratios:
    y = [Jxx/Jzz, Jyy/Jzz]^T in (0, 1]

References:
- Ma et al., 'Robust attitude estimation of rotating space debris based on
  virtual observations of neural network', 2021. Section 3.2-3.4.
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from simulation.dynamics_generator import SpacecraftDynamicsSimulator
from tracking.dft_feature_extractor import QuaternionDFTFeatureExtractor
from models.cfin_net import CFINNet


def generate_and_extract_dataset(num_samples: int = 3000, seq_len: int = 300, dt: float = 0.1):
    print(f"Generating {num_samples} simulated tumbling sequences ({seq_len} steps each at {1/dt:.1f} Hz)...")
    sim = SpacecraftDynamicsSimulator(dt=dt)
    extractor = QuaternionDFTFeatureExtractor(dt=dt)

    X_list = []
    Y_list = []

    t0 = time.time()
    for i in range(num_samples):
        data = sim.simulate_sequence(num_frames=seq_len, measurement_noise_std=0.04)
        xc = extractor.extract_features_single(data["quaternions_noisy"])
        X_list.append(xc)
        Y_list.append(data["inertia_ratios"])

        if (i + 1) % 500 == 0 or (i + 1) == num_samples:
            elapsed = time.time() - t0
            print(f"  Generated {i + 1}/{num_samples} samples ({elapsed:.1f}s)")

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    return X, Y, extractor


def train_cfin(
    num_samples: int = 3000,
    seq_len: int = 300,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    output_dir: str = "checkpoints"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CFIN on device: {device}")

    # 1. Generate simulation sequences and extract 15-dim DFT features
    X_raw, Y, extractor = generate_and_extract_dataset(num_samples=num_samples, seq_len=seq_len)

    # 2. Fit whitening normalizer on training split (80% train, 10% val, 10% test)
    num_train = int(0.8 * num_samples)
    num_val = int(0.1 * num_samples)

    extractor.fit_normalizer(X_raw[:num_train])
    X_norm = extractor.normalize(X_raw)

    # Save normalizer statistics
    norm_path = os.path.join(output_dir, "cfin_norm.npz")
    extractor.save_stats(norm_path)
    print(f"Saved whitening normalization parameters to {norm_path}")

    # Create PyTorch datasets
    X_train, Y_train = torch.from_numpy(X_norm[:num_train]), torch.from_numpy(Y[:num_train])
    X_val, Y_val = torch.from_numpy(X_norm[num_train:num_train + num_val]), torch.from_numpy(Y[num_train:num_train + num_val])
    X_test, Y_test = torch.from_numpy(X_norm[num_train + num_val:]), torch.from_numpy(Y[num_train + num_val:])

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    # 3. Initialize CFIN model
    model = CFINNet(input_dim=15, hidden_dim=100, output_dim=2, activation="tanh").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    weights_path = os.path.join(output_dir, "cfin_weights.pt")

    print("\nStarting CFIN training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = model.compute_loss(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = model.compute_loss(pred, by)
                val_loss += loss.item() * len(bx)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | Best: {best_val_loss:.6f}")

    print(f"\nTraining completed! Saved best CFIN model weights to {weights_path}")

    # 4. Evaluation on test set
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test.to(device)).cpu().numpy()
        test_true = Y_test.numpy()
        errors = np.abs(test_pred - test_true)
        mean_err = np.mean(errors, axis=0)
        max_err = np.max(errors, axis=0)

    print("\n--- Test Set Evaluation Results ---")
    print(f"Mean Error [Jxx/Jzz, Jyy/Jzz]: [{mean_err[0]:.4f}, {mean_err[1]:.4f}]")
    print(f"Max Error  [Jxx/Jzz, Jyy/Jzz]: [{max_err[0]:.4f}, {max_err[1]:.4f}]")
    print(f"Within 0.05 error threshold: {np.mean(errors < 0.05) * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 3 CFIN Inertia Ratio Network")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of simulation sequences")
    parser.add_argument("--seq_len", type=int, default=250, help="Quaternion sequence length")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_cfin(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
