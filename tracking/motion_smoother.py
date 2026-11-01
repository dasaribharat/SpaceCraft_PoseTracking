"""
Stage 2: Motion Estimation and Temporal Smoothing.

Turns sequential 6DoF pose estimates into continuous motion trajectories:
1. Enforces quaternion sign-continuity across time: q_k . q_{k-1} >= 0
2. Applies temporal smoothing (Savitzky-Golay / polynomial filtering)
3. Derives 3D linear velocity (v = dT/dt) via central differences
4. Derives 3D angular velocity (omega_b) via quaternion kinematics:
      [0, omega_b^T]^T = 2 * q^* \circ dq/dt
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Dict, Any, List, Optional


def enforce_quaternion_sign_continuity(quaternions: np.ndarray) -> np.ndarray:
    """
    Ensures that sequential quaternions don't flip antipodally on S^3.
    Args:
        quaternions: (N, 4) array of [qw, qx, qy, qz]
    Returns:
        continuous_quats: (N, 4) sign-continuous unit quaternions
    """
    N = len(quaternions)
    if N <= 1:
        return quaternions.copy()

    q_cont = np.zeros_like(quaternions)
    q_cont[0] = quaternions[0] / np.linalg.norm(quaternions[0])

    for k in range(1, N):
        qk = quaternions[k] / np.linalg.norm(quaternions[k])
        dot = np.dot(qk, q_cont[k - 1])
        if dot < 0.0:
            q_cont[k] = -qk
        else:
            q_cont[k] = qk

    return q_cont


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamiltonian quaternion product q = q1 \circ q2.
    Both q1, q2 are [qw, qx, qy, qz].
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Conjugate q* = [qw, -qx, -qy, -qz].
    """
    q_conj = q.copy()
    q_conj[..., 1:] = -q_conj[..., 1:]
    return q_conj


class PoseMotionSmoother:
    """
    Filter that turns raw frame-by-frame 6DoF pose detections into smoothed states
    and derived velocities.
    """
    def __init__(self, window_size: int = 7, poly_order: int = 2, dt: float = 0.1):
        self.window_size = window_size
        self.poly_order = poly_order
        self.dt = dt

    def process_sequence(
        self,
        translations: np.ndarray,
        quaternions: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Processes full sequence of poses:
        Args:
            translations: (N, 3) translations [X, Y, Z]
            quaternions: (N, 4) quaternions [qw, qx, qy, qz]
            timestamps: Optional (N,) time in seconds. If None, uses uniform dt.

        Returns:
            Dict containing:
              - 'smoothed_translations': (N, 3)
              - 'smoothed_quaternions': (N, 4)
              - 'linear_velocities': (N, 3) [vx, vy, vz]
              - 'angular_velocities_body': (N, 3) [wx, wy, wz]
              - 'angular_velocities_cam': (N, 3)
        """
        N = len(translations)
        assert len(quaternions) == N, "Translations and quaternions must have the same length"

        # 1. Enforce sign continuity
        q_clean = enforce_quaternion_sign_continuity(quaternions)

        # 2. Smooth trajectory
        w_len = min(self.window_size, N)
        if w_len % 2 == 0:
            w_len -= 1

        if w_len > self.poly_order + 1:
            T_smooth = savgol_filter(translations, window_length=w_len, polyorder=self.poly_order, axis=0)
            q_filt = savgol_filter(q_clean, window_length=w_len, polyorder=self.poly_order, axis=0)
            # Re-normalize quaternions to unit sphere S^3
            norms = np.linalg.norm(q_filt, axis=1, keepdims=True)
            q_smooth = q_filt / np.clip(norms, 1e-8, None)
        else:
            T_smooth = translations.copy()
            q_smooth = q_clean.copy()

        # 3. Derive linear velocities: v = dT/dt
        v_linear = np.zeros_like(T_smooth)
        if N > 1:
            if timestamps is not None and len(timestamps) == N:
                dt_arr = np.gradient(timestamps)
                v_linear[:, 0] = np.gradient(T_smooth[:, 0], timestamps)
                v_linear[:, 1] = np.gradient(T_smooth[:, 1], timestamps)
                v_linear[:, 2] = np.gradient(T_smooth[:, 2], timestamps)
            else:
                v_linear = np.gradient(T_smooth, self.dt, axis=0)

        # 4. Derive quaternion time derivative: dq/dt
        dq_dt = np.gradient(q_smooth, self.dt, axis=0)

        # 5. Derive body angular velocity: [0, omega_b^T]^T = 2 * q^* \circ dq/dt
        q_conj = quaternion_conjugate(q_smooth)
        prod_body = quaternion_multiply(q_conj, dq_dt)
        omega_body = 2.0 * prod_body[:, 1:4]

        # 6. Derive camera-frame angular velocity: [0, omega_c^T]^T = 2 * dq/dt \circ q^*
        prod_cam = quaternion_multiply(dq_dt, q_conj)
        omega_cam = 2.0 * prod_cam[:, 1:4]

        return {
            "smoothed_translations": T_smooth,
            "smoothed_quaternions": q_smooth,
            "linear_velocities": v_linear,
            "angular_velocities_body": omega_body,
            "angular_velocities_cam": omega_cam
        }
