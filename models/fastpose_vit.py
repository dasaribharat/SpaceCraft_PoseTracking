"""
FastPose-ViT: Vision Transformer for Real-Time Spacecraft Pose Estimation.

Direct 6DoF pose regression from cropped spacecraft images using decoupled
apparent rotation and normalized translation targets.

References:
- Pierre Ancey, Andrew Price, Saqib Javed, Mathieu Salzmann.
  'FastPose-ViT: A Vision Transformer for Real-Time Spacecraft Pose Estimation', 2025.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from models.apparent_rotation import (
    gram_schmidt_6d_to_matrix,
    apparent_to_true_rotation,
    recover_absolute_translation,
    matrix_to_quaternion
)


class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and projects to embedding dimension.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 192, num_heads: int = 4, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLPBlock(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 192, num_heads: int = 4, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, int(embed_dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FastPoseViT(nn.Module):
    """
    FastPose-ViT network:
    Input: Cropped spacecraft RGB image (B, 3, 224, 224)
    Output: 9 pose parameters:
      - 6 parameters for apparent rotation (r1_hat, r2_hat in R^3)
      - 3 parameters for normalized translation U = (Ux, Uy, Uz)
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Pose Regression MLP Head (Section 3.1)
        self.pose_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 9)  # 6 for rotation (r1, r2) + 3 for normalized translation U
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: (B, C, H, W) cropped satellite images
        Returns:
            r12_pred: (B, 6) apparent rotation 6D representation
            U_pred: (B, 3) normalized translation target (Ux, Uy, Uz)
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Extract [CLS] token representation
        cls_out = x[:, 0]
        raw_pose = self.pose_head(cls_out)

        r12_pred = raw_pose[:, :6]
        U_pred = raw_pose[:, 6:9]
        return r12_pred, U_pred

    def compute_loss(
        self,
        r12_pred: torch.Tensor,
        U_pred: torch.Tensor,
        R_prime_gt: torch.Tensor,
        U_gt: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the FastPose-ViT training loss (Eq. 15-17):
            L_rot = ||R'_pred,1:2 - R'_gt,1:2||_F^2
            L_trans = ||U_pred - U_gt||_2^2
            L_total = L_rot + L_trans
        """
        R_prime_pred = gram_schmidt_6d_to_matrix(r12_pred)
        # First two columns of apparent rotation
        R_prime_pred_12 = R_prime_pred[..., :, :2]
        R_prime_gt_12 = R_prime_gt[..., :, :2]

        loss_rot = torch.sum((R_prime_pred_12 - R_prime_gt_12) ** 2, dim=(-1, -2)).mean()
        loss_trans = torch.sum((U_pred - U_gt) ** 2, dim=-1).mean()
        loss_total = loss_rot + loss_trans

        return {
            "loss_total": loss_total,
            "loss_rot": loss_rot,
            "loss_trans": loss_trans
        }

    @torch.no_grad()
    def predict_6dof(
        self,
        crop: torch.Tensor,
        bbox: torch.Tensor,
        full_w: float = 1920.0,
        full_h: float = 1200.0,
        fx: float = 1920.0,
        fy: float = 1920.0,
        cx: float = 960.0,
        cy: float = 600.0,
        alpha: float = 1.6
    ) -> Dict[str, torch.Tensor]:
        """
        Inference method with full geometric recovery:
        1. Predict r12_pred and U_pred from crop
        2. Recover absolute translation T = [X, Y, Z]
        3. Recover apparent rotation R' via Gram-Schmidt
        4. Undo perspective distortion via Rodrigues correction: R_pred = Delta_R(T)^T * R'
        5. Convert R_pred to unit quaternion q_pred
        """
        self.eval()
        r12_pred, U_pred = self.forward(crop)

        # 1. Translation recovery
        T_pred = recover_absolute_translation(
            U_pred, bbox, full_w=full_w, full_h=full_h,
            fx=fx, fy=fy, cx=cx, cy=cy, alpha=alpha
        )

        # 2. Apparent rotation recovery
        R_prime_pred = gram_schmidt_6d_to_matrix(r12_pred)

        # 3. Perspective correction to true orientation
        R_true_pred = apparent_to_true_rotation(R_prime_pred, T_pred)

        # 4. Quaternion conversion
        q_pred = matrix_to_quaternion(R_true_pred)

        return {
            "T_pred": T_pred,
            "R_pred": R_true_pred,
            "q_pred": q_pred,
            "U_pred": U_pred,
            "R_prime_pred": R_prime_pred
        }
