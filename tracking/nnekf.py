"""
Stage 4: Neural-Network-Enhanced Kalman Filter (NNEKF) and Trajectory Forecaster.

Tracks spacecraft attitude (quaternion), angular velocity, and principal inertia ratios,
incorporating virtual observations from the Stage 3 CFIN network and forecasting
motion through camera dropouts (eclipses, occlusions, glare blind-outs).

References:
- Ma et al., 'Robust attitude estimation of rotating space debris based on
  virtual observations of neural network', Int. J. Robust Nonlinear Control, 2021.
  Section 4 'Nonlinear Filter for Parameter Estimation'.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from simulation.dynamics_generator import rk4_step, euler_rotational_dynamics, quaternion_derivative


class NNEKF:
    """
    Neural-Network-Enhanced Kalman Filter (NNEKF).
    State vector: x = [qv (3), omega (3), y (2)]^T in R^8
    Full quaternion q = [q0, qv^T]^T maintained in parallel with unit norm.
    """
    def __init__(
        self,
        dt: float = 0.1,
        q_init: Optional[np.ndarray] = None,
        omega_init: Optional[np.ndarray] = None,
        y_init: Optional[np.ndarray] = None,
        q_cov_attitude: float = 1e-5,
        q_cov_omega: float = 1e-4,
        q_cov_inertia: float = 1e-7,
        r_cov_attitude: float = 2.5e-3,
        r_cov_inertia_virt: float = 5.0e-3
    ):
        self.dt = dt

        # Initial state
        self.q = q_init.copy() if q_init is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.q /= np.linalg.norm(self.q)
        if self.q[0] < 0:
            self.q = -self.q

        self.omega = omega_init.copy() if omega_init is not None else np.zeros(3, dtype=np.float64)
        self.y = y_init.copy() if y_init is not None else np.array([0.5, 0.75], dtype=np.float64)

        # State error covariance P (8x8)
        self.P = np.diag([
            1e-3, 1e-3, 1e-3,       # qv
            1e-2, 1e-2, 1e-2,       # omega
            0.1, 0.1                # y (inertia ratios)
        ]).astype(np.float64)

        # Process noise covariance Q (8x8)
        self.Q = np.diag([
            q_cov_attitude, q_cov_attitude, q_cov_attitude,
            q_cov_omega, q_cov_omega, q_cov_omega,
            q_cov_inertia, q_cov_inertia
        ]).astype(np.float64)

        # Measurement noise covariances
        self.R_att = (np.eye(3) * r_cov_attitude).astype(np.float64)
        self.R_virt = (np.eye(2) * r_cov_inertia_virt).astype(np.float64)

        # Translation tracking states: [X, Y, Z, vx, vy, vz]^T
        self.T = np.array([0.0, 0.0, 15.0], dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.P_trans = np.eye(6, dtype=np.float64) * 0.1
        self.Q_trans = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]).astype(np.float64)
        self.R_trans = (np.eye(3) * 0.01).astype(np.float64)

    def _compute_state_transition_jacobian(self) -> np.ndarray:
        """
        Computes the continuous process Jacobian F and discrete state transition matrix Phi_k:
        Phi_k = I_8 + dt * F(x_hat) (Eq. 19-23 in Ma et al.).
        """
        wx, wy, wz = self.omega
        y1, y2 = self.y

        # Avoid divide-by-zero
        y1 = max(y1, 1e-4)
        y2 = max(y2, 1e-4)

        # -[omega x] skew-symmetric matrix (Eq. 21)
        w_cross_neg = np.array([
            [ 0.0,   wz, -wy],
            [-wz,   0.0,   wx],
            [ wy,   -wx,  0.0]
        ], dtype=np.float64)

        # Jacobian M = d(dw/dt) / d(omega) (Eq. 19)
        M = np.array([
            [0.0, ((y2 - 1.0) / y1) * wz, ((y2 - 1.0) / y1) * wy],
            [((1.0 - y1) / y2) * wz, 0.0, ((1.0 - y1) / y2) * wx],
            [(y1 - y2) * wy, (y1 - y2) * wx, 0.0]
        ], dtype=np.float64)

        # Jacobian N = d(dw/dt) / d(y) (Eq. 20)
        N = np.array([
            [-((y2 - 1.0) / (y1 ** 2)) * wy * wz, (1.0 / y1) * wy * wz],
            [-(1.0 / y2) * wx * wz, -((1.0 - y1) / (y2 ** 2)) * wx * wz],
            [wx * wy, -wx * wy]
        ], dtype=np.float64)

        # Assemble full 8x8 Jacobian F
        F = np.zeros((8, 8), dtype=np.float64)
        F[0:3, 0:3] = w_cross_neg
        F[0:3, 3:6] = 0.5 * np.eye(3)
        F[3:6, 3:6] = M
        F[3:6, 6:8] = N
        # F[6:8, :] remains 0 because inertia ratios are constant

        Phi = np.eye(8, dtype=np.float64) + self.dt * F
        return Phi

    def predict(self, dt: Optional[float] = None):
        """
        Estimate Propagation Step (Eq. 25-26):
        Propagates state forward using Euler rotational dynamics and attitude kinematics via RK4.
        """
        step_dt = dt if dt is not None else self.dt

        # 1. Propagate attitude and angular velocity via RK4
        self.q, self.omega = rk4_step(self.q, self.omega, self.y, step_dt)
        if self.q[0] < 0:
            self.q = -self.q

        # 2. Covariance propagation
        Phi = self._compute_state_transition_jacobian()
        self.P = Phi @ self.P @ Phi.T + self.Q

        # 3. Propagate translational state
        F_trans = np.eye(6, dtype=np.float64)
        F_trans[0:3, 3:6] = np.eye(3) * step_dt
        self.T = self.T + self.v * step_dt
        self.P_trans = F_trans @ self.P_trans @ F_trans.T + self.Q_trans

    def update_attitude_vision(self, q_meas: np.ndarray):
        """
        Vision Measurement Update Step (Eq. 27-34):
        Corrects attitude quaternion and angular velocity from detected camera pose.
        """
        # Ensure sign consistency with current estimate
        q_m = q_meas / np.linalg.norm(q_meas)
        if np.dot(q_m, self.q) < 0.0:
            q_m = -q_m

        # Observation matrix for quaternion vector part: H = [I_3, 0_3, 0_2] (3x8)
        H = np.zeros((3, 8), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        # Innovation: delta_z = q_m_v - q_v
        delta_z = q_m[1:4] - self.q[1:4]

        # Kalman Gain
        S = H @ self.P @ H.T + self.R_att
        K = self.P @ H.T @ np.linalg.inv(S)

        # Correction
        delta_x = K @ delta_z

        # Update Covariance
        I_8 = np.eye(8, dtype=np.float64)
        self.P = (I_8 - K @ H) @ self.P

        # Multiplicative quaternion error update (Eq. 30-31)
        delta_qv = delta_x[0:3]
        norm_sq = np.sum(delta_qv ** 2)
        delta_q0 = np.sqrt(max(0.0, 1.0 - norm_sq))
        delta_q = np.array([delta_q0, delta_qv[0], delta_qv[1], delta_qv[2]], dtype=np.float64)

        # Quaternion multiplication: q_new = q \circ delta_q
        w1, x1, y1, z1 = self.q
        w2, x2, y2, z2 = delta_q
        q_updated = np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ], dtype=np.float64)
        self.q = q_updated / np.linalg.norm(q_updated)
        if self.q[0] < 0:
            self.q = -self.q

        # Update angular velocity and inertia ratios
        self.omega = self.omega + delta_x[3:6]
        self.y = np.clip(self.y + delta_x[6:8], 0.05, 0.98)

    def update_inertia_virtual(self, y_cfin: np.ndarray):
        """
        Virtual Observation Update Step (Ma et al. Section 4.1):
        Incorporates the CFIN predicted inertia ratios y_cfin = [Jxx/Jzz, Jyy/Jzz].
        """
        # Observation matrix for inertia ratios: H = [0_2x3, 0_2x3, I_2] (2x8)
        H = np.zeros((2, 8), dtype=np.float64)
        H[0:2, 6:8] = np.eye(2)

        delta_z = y_cfin - self.y

        S = H @ self.P @ H.T + self.R_virt
        K = self.P @ H.T @ np.linalg.inv(S)

        delta_x = K @ delta_z
        I_8 = np.eye(8, dtype=np.float64)
        self.P = (I_8 - K @ H) @ self.P

        # Update state components
        self.y = np.clip(self.y + delta_x[6:8], 0.05, 0.98)
        self.omega = self.omega + delta_x[3:6]

    def update_translation_vision(self, T_meas: np.ndarray):
        """
        Updates translational position and linear velocity from camera detection.
        """
        H = np.zeros((3, 6), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        delta_z = T_meas - self.T
        S = H @ self.P_trans @ H.T + self.R_trans
        K = self.P_trans @ H.T @ np.linalg.inv(S)

        delta_x = K @ delta_z
        self.P_trans = (np.eye(6) - K @ H) @ self.P_trans

        self.T = self.T + delta_x[0:3]
        self.v = self.v + delta_x[3:6]

    def forecast_trajectory(self, duration_seconds: float, step_dt: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Propagates motion forward in time during camera dropouts (occlusion, eclipse, out of FOV).
        Uses current estimated state and inertia parameters.
        """
        num_steps = int(duration_seconds / step_dt)
        f_times = np.zeros(num_steps)
        f_quats = np.zeros((num_steps, 4))
        f_omegas = np.zeros((num_steps, 3))
        f_translations = np.zeros((num_steps, 3))

        curr_q = self.q.copy()
        curr_w = self.omega.copy()
        curr_T = self.T.copy()
        curr_v = self.v.copy()
        curr_y = self.y.copy()

        for step in range(num_steps):
            f_times[step] = (step + 1) * step_dt
            curr_q, curr_w = rk4_step(curr_q, curr_w, curr_y, step_dt)
            curr_T = curr_T + curr_v * step_dt

            f_quats[step] = curr_q
            f_omegas[step] = curr_w
            f_translations[step] = curr_T

        return {
            "forecast_times_s": f_times,
            "forecast_quaternions": f_quats,
            "forecast_angular_velocities": f_omegas,
            "forecast_translations": f_translations
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "quaternion": self.q.copy(),
            "angular_velocity": self.omega.copy(),
            "inertia_ratios": self.y.copy(),
            "translation": self.T.copy(),
            "velocity": self.v.copy(),
            "covariance_P": self.P.copy()
        }
