"""
Apparent Rotation & Normalized Translation Transformations for FastPose-ViT.

References:
- FastPose-ViT: A Vision Transformer for Real-Time Spacecraft Pose Estimation (2512.09792v1.pdf),
  Section 3.3 'Geometric Reformulation of Pose Regression for Cropped Images'.
"""

import torch
import torch.nn.functional as F
import numpy as np


def gram_schmidt_6d_to_matrix(r12: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation (first two 3D columns) to a valid SO(3)
    rotation matrix using Gram-Schmidt orthogonalization (Eq. 11-12 in FastPose-ViT).

    Args:
        r12: Tensor of shape (..., 6) where r12[..., 0:3] is r1_hat and r12[..., 3:6] is r2_hat.

    Returns:
        R: Tensor of shape (..., 3, 3) representing an orthonormal rotation matrix with det(R) = +1.
    """
    shape = r12.shape[:-1]
    r1_hat = r12[..., 0:3]
    r2_hat = r12[..., 3:6]

    # r1 = r1_hat / ||r1_hat||
    r1 = F.normalize(r1_hat, dim=-1, eps=1e-8)

    # proj = (r1^T * r2_hat) * r1
    proj = torch.sum(r1 * r2_hat, dim=-1, keepdim=True) * r1
    r2_ortho = r2_hat - proj
    r2 = F.normalize(r2_ortho, dim=-1, eps=1e-8)

    # r3 = r1 x r2
    r3 = torch.cross(r1, r2, dim=-1)

    # Stack as columns: R = [r1, r2, r3]
    R = torch.stack([r1, r2, r3], dim=-1)
    return R


def compute_corrective_rotation(T: torch.Tensor) -> torch.Tensor:
    """
    Computes the corrective rotation Delta_R(T) that maps true orientation to apparent orientation:
        R' = Delta_R * R_true  ==>  R_true = Delta_R^T * R'
    as defined in FastPose-ViT Eq. (8)-(9).

    Args:
        T: Translation vector(s) of shape (..., 3) [X, Y, Z] in camera frame.

    Returns:
        Delta_R: Tensor of shape (..., 3, 3) rotation matrices.
    """
    shape = T.shape[:-1]
    T_flat = T.view(-1, 3)
    batch_size = T_flat.shape[0]

    norm_T = torch.norm(T_flat, dim=-1, keepdim=True).clamp(min=1e-8)
    T_unit = T_flat / norm_T  # (B, 3)

    # Optical axis ez = [0, 0, 1]^T
    ez = torch.tensor([0.0, 0.0, 1.0], device=T.device, dtype=T.dtype).expand(batch_size, 3)

    # u = (T_unit x ez) / ||T_unit x ez||
    cross_prod = torch.cross(T_unit, ez, dim=-1)
    cross_norm = torch.norm(cross_prod, dim=-1, keepdim=True)

    # Check for collinearity (when target is on the optical axis, theta = 0, Delta_R = I)
    is_collinear = (cross_norm < 1e-6).squeeze(-1)
    cross_norm_safe = cross_norm.clamp(min=1e-8)
    u = cross_prod / cross_norm_safe

    # theta = arccos(T_unit^T * ez)
    dot_prod = torch.sum(T_unit * ez, dim=-1).clamp(min=-1.0, max=1.0)
    theta = torch.acos(dot_prod)  # (B,)

    # Skew-symmetric matrix [u]_x
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    zeros = torch.zeros_like(ux)
    u_skew = torch.stack([
        zeros, -uz, uy,
        uz, zeros, -ux,
        -uy, ux, zeros
    ], dim=-1).view(batch_size, 3, 3)

    u_skew_sq = torch.bmm(u_skew, u_skew)

    sin_theta = torch.sin(theta).view(batch_size, 1, 1)
    one_minus_cos = (1.0 - torch.cos(theta)).view(batch_size, 1, 1)

    eye = torch.eye(3, device=T.device, dtype=T.dtype).expand(batch_size, 3, 3)
    Delta_R = eye + sin_theta * u_skew + one_minus_cos * u_skew_sq

    # For collinear vectors (theta ~ 0 or pi), set to identity
    if is_collinear.any():
        Delta_R[is_collinear] = eye[is_collinear]

    return Delta_R.view(*shape, 3, 3)


def true_to_apparent_rotation(R_true: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    R' = Delta_R * R_true (Eq. 8).
    Used during dataset preparation/loss calculation.
    """
    Delta_R = compute_corrective_rotation(T)
    return torch.matmul(Delta_R, R_true)


def apparent_to_true_rotation(R_prime: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    R_true = Delta_R^T * R' (Eq. 10).
    Used during model inference.
    """
    Delta_R = compute_corrective_rotation(T)
    Delta_R_T = Delta_R.transpose(-1, -2)
    return torch.matmul(Delta_R_T, R_prime)


def compute_normalized_translation_targets(
    T: torch.Tensor,
    bbox: torch.Tensor,
    full_w: float,
    full_h: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    alpha: float = 1.6
) -> torch.Tensor:
    """
    Maps absolute 3D translation T = [X, Y, Z] to normalized crop targets U = (Ux, Uy, Uz)
    as defined in FastPose-ViT Eq. (3)-(6).

    Args:
        T: (..., 3) [X, Y, Z]
        bbox: (..., 4) [xmin, ymin, xmax, ymax]
        full_w, full_h: Dimensions of the full camera frame (W, H)
        fx, fy, cx, cy: Camera intrinsics
        alpha: Aspect ratio correction factor (1.6 for SPEED/SPEED+)

    Returns:
        U: (..., 3) [Ux, Uy, Uz]
    """
    X, Y, Z = T[..., 0], T[..., 1], T[..., 2]
    xmin, ymin, xmax, ymax = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    w = (xmax - xmin).clamp(min=1.0)
    h = (ymax - ymin).clamp(min=1.0)
    bx = (xmin + xmax) * 0.5
    by = (ymin + ymax) * 0.5

    sx = full_w / (alpha * w)
    sy = full_h / h

    # Uz = 0.5 * (1/sx + 1/sy) * Z  (Eq. 4)
    Uz = 0.5 * (1.0 / sx + 1.0 / sy) * Z

    # Ux = (1/w) * (fx * X / Z + cx - bx)  (Eq. 6)
    Ux = (1.0 / w) * (fx * (X / Z.clamp(min=1e-5)) + cx - bx)

    # Uy = (1/h) * (fy * Y / Z + cy - by)  (Eq. 6)
    Uy = (1.0 / h) * (fy * (Y / Z.clamp(min=1e-5)) + cy - by)

    return torch.stack([Ux, Uy, Uz], dim=-1)


def recover_absolute_translation(
    U: torch.Tensor,
    bbox: torch.Tensor,
    full_w: float,
    full_h: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    alpha: float = 1.6
) -> torch.Tensor:
    """
    Recovers absolute 3D translation T = [X, Y, Z] from predicted normalized targets U
    and bounding box using FastPose-ViT Eq. (4) and (7).

    Args:
        U: (..., 3) [Ux, Uy, Uz]
        bbox: (..., 4) [xmin, ymin, xmax, ymax]
        full_w, full_h: Dimensions of full frame
        fx, fy, cx, cy: Camera intrinsics
        alpha: Aspect ratio correction factor

    Returns:
        T: (..., 3) [X, Y, Z] in meters
    """
    Ux, Uy, Uz = U[..., 0], U[..., 1], U[..., 2]
    xmin, ymin, xmax, ymax = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    w = (xmax - xmin).clamp(min=1.0)
    h = (ymax - ymin).clamp(min=1.0)
    bx = (xmin + xmax) * 0.5
    by = (ymin + ymax) * 0.5

    sx = full_w / (alpha * w)
    sy = full_h / h

    scale_factor = 0.5 * (1.0 / sx + 1.0 / sy)
    Z = Uz / scale_factor.clamp(min=1e-8)

    X = (1.0 / fx) * (bx + Ux * w - cx) * Z
    Y = (1.0 / fy) * (by + Uy * h - cy) * Z

    return torch.stack([X, Y, Z], dim=-1)


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Converts 3x3 rotation matrices to unit quaternions q = [qw, qx, qy, qz]
    with positive scalar part qw >= 0 for sign-continuity.
    """
    R_flat = R.view(-1, 3, 3)
    batch_size = R_flat.shape[0]

    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)

    mask0 = trace > 0.0
    if mask0.any():
        s = torch.sqrt(trace[mask0] + 1.0) * 2.0
        q[mask0, 0] = 0.25 * s
        q[mask0, 1] = (R_flat[mask0, 2, 1] - R_flat[mask0, 1, 2]) / s
        q[mask0, 2] = (R_flat[mask0, 0, 2] - R_flat[mask0, 2, 0]) / s
        q[mask0, 3] = (R_flat[mask0, 1, 0] - R_flat[mask0, 0, 1]) / s

    mask1 = (~mask0) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    if mask1.any():
        s = torch.sqrt(1.0 + R_flat[mask1, 0, 0] - R_flat[mask1, 1, 1] - R_flat[mask1, 2, 2]) * 2.0
        q[mask1, 0] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s
        q[mask1, 1] = 0.25 * s
        q[mask1, 2] = (R_flat[mask1, 0, 1] + R_flat[mask1, 1, 0]) / s
        q[mask1, 3] = (R_flat[mask1, 0, 2] + R_flat[mask1, 2, 0]) / s

    mask2 = (~mask0) & (~mask1) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if mask2.any():
        s = torch.sqrt(1.0 + R_flat[mask2, 1, 1] - R_flat[mask2, 0, 0] - R_flat[mask2, 2, 2]) * 2.0
        q[mask2, 0] = (R_flat[mask2, 0, 2] - R_flat[mask2, 2, 0]) / s
        q[mask2, 1] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s
        q[mask2, 2] = 0.25 * s
        q[mask2, 3] = (R_flat[mask2, 1, 2] + R_flat[mask2, 2, 1]) / s

    mask3 = (~mask0) & (~mask1) & (~mask2)
    if mask3.any():
        s = torch.sqrt(1.0 + R_flat[mask3, 2, 2] - R_flat[mask3, 0, 0] - R_flat[mask3, 1, 1]) * 2.0
        q[mask3, 0] = (R_flat[mask3, 1, 0] - R_flat[mask3, 0, 1]) / s
        q[mask3, 1] = (R_flat[mask3, 0, 2] + R_flat[mask3, 2, 0]) / s
        q[mask3, 2] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s
        q[mask3, 3] = 0.25 * s

    q = F.normalize(q, dim=-1)
    # Enforce qw >= 0
    neg_mask = q[:, 0] < 0.0
    q[neg_mask] = -q[neg_mask]

    return q.view(*R.shape[:-2], 4)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Converts unit quaternions q = [qw, qx, qy, qz] to 3x3 rotation matrices.
    """
    q_norm = F.normalize(q, dim=-1)
    qw, qx, qy, qz = q_norm[..., 0], q_norm[..., 1], q_norm[..., 2], q_norm[..., 3]

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)

    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)

    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    R = torch.stack([
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22
    ], dim=-1).view(*q.shape[:-1], 3, 3)
    return R


def compute_pose_errors(
    T_pred: torch.Tensor,
    T_gt: torch.Tensor,
    q_pred: torch.Tensor,
    q_gt: torch.Tensor
):
    """
    ESA Pose Estimation Challenge Metrics (FastPose-ViT Eq. 13-14):
    ET = ||T_pred - T_gt||_2
    ER = 2 * arccos(|<q_pred, q_gt>|)
    """
    ET = torch.norm(T_pred - T_gt, dim=-1)
    inner_prod = torch.abs(torch.sum(q_pred * q_gt, dim=-1)).clamp(max=1.0)
    ER = 2.0 * torch.acos(inner_prod)
    return ET, ER
