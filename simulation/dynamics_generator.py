"""
6DoF Rigid Body Tumbling Dynamics Generator.

Implements torque-free rotational Euler dynamics and orbital translation:
- Simulates realistic tumbling satellite/debris motion
- Generates the rendering-free layer of 10,000+ quaternion sequences to train
  the CFIN inertia network (Stage 3)
- Provides exact ground-truth for pose, velocities, and inertia ratios

References:
- Ma et al., 'Robust attitude estimation of rotating space debris based on
  virtual observations of neural network', 2021. Section 2 & 3.4.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional


def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Kinematic equation: dq/dt = 0.5 * q \circ [0, omega]^T
    q: [qw, qx, qy, qz]
    omega: [wx, wy, wz] in body frame
    """
    qw, qx, qy, qz = q
    wx, wy, wz = omega

    dq_w = 0.5 * (-qx * wx - qy * wy - qz * wz)
    dq_x = 0.5 * ( qw * wx + qy * wz - qz * wy)
    dq_y = 0.5 * ( qw * wy - qx * wz + qz * wx)
    dq_z = 0.5 * ( qw * wz + qx * wy - qy * wx)

    return np.array([dq_w, dq_x, dq_y, dq_z], dtype=np.float64)


def euler_rotational_dynamics(omega: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Torque-free Euler rotational dynamics (Eq. 16 in Ma et al.):
    y = [y1, y2] = [Jxx/Jzz, Jyy/Jzz] with Jzz = 1.0.

    dwx/dt = ((y2 - 1) / y1) * wy * wz
    dwy/dt = ((1 - y1) / y2) * wx * wz
    dwz/dt = (y1 - y2) * wx * wy
    """
    wx, wy, wz = omega
    y1, y2 = y

    dwx = ((y2 - 1.0) / y1) * wy * wz
    dwy = ((1.0 - y1) / y2) * wx * wz
    dwz = (y1 - y2) * wx * wy

    return np.array([dwx, dwy, dwz], dtype=np.float64)


def rk4_step(
    q: np.ndarray,
    omega: np.ndarray,
    y: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    4th-order Runge-Kutta integration step for coupled attitude dynamics and kinematics.
    """
    # k1
    k1_w = euler_rotational_dynamics(omega, y)
    k1_q = quaternion_derivative(q, omega)

    # k2
    w2 = omega + 0.5 * dt * k1_w
    q2 = q + 0.5 * dt * k1_q
    q2 /= np.linalg.norm(q2)
    k2_w = euler_rotational_dynamics(w2, y)
    k2_q = quaternion_derivative(q2, w2)

    # k3
    w3 = omega + 0.5 * dt * k2_w
    q3 = q + 0.5 * dt * k2_q
    q3 /= np.linalg.norm(q3)
    k3_w = euler_rotational_dynamics(w3, y)
    k3_q = quaternion_derivative(q3, w3)

    # k4
    w4 = omega + dt * k3_w
    q4 = q + dt * k3_q
    q4 /= np.linalg.norm(q4)
    k4_w = euler_rotational_dynamics(w4, y)
    k4_q = quaternion_derivative(q4, w4)

    # Update
    omega_next = omega + (dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)
    q_next = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
    q_next /= np.linalg.norm(q_next)

    return q_next, omega_next


class SpacecraftDynamicsSimulator:
    """
    Simulates complete 6DoF trajectory of a tumbling space debris target.
    """
    def __init__(self, dt: float = 0.1, internal_substeps: int = 10):
        self.dt = dt
        self.internal_substeps = internal_substeps
        self.sub_dt = dt / internal_substeps

    def generate_random_inertia_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates random inertia tensor J and ratios y = [Jxx/Jzz, Jyy/Jzz]
        per Ma et al. Section 3.4 (Eq. 9):
        m = [mx, my, mz] ~ U[-1, 1]
        Jxx = my^2 + mz^2, Jyy = mx^2 + mz^2, Jzz = mx^2 + my^2
        sorted such that Jxx < Jyy < Jzz.
        """
        while True:
            m = np.random.uniform(-1.0, 1.0, size=3)
            J_raw = np.array([
                m[1] ** 2 + m[2] ** 2,
                m[0] ** 2 + m[2] ** 2,
                m[0] ** 2 + m[1] ** 2
            ])
            # Ensure non-degenerate and distinct moments
            if np.min(J_raw) > 0.05 and (np.max(J_raw) - np.min(J_raw)) > 0.05:
                break

        J_sorted = np.sort(J_raw)
        Jxx, Jyy, Jzz = J_sorted[0], J_sorted[1], J_sorted[2]

        # Normalized inertia ratios: Jxx/Jzz and Jyy/Jzz
        y = np.array([Jxx / Jzz, Jyy / Jzz], dtype=np.float64)
        return J_sorted, y

    def simulate_sequence(
        self,
        num_frames: int = 500,
        y: Optional[np.ndarray] = None,
        omega0: Optional[np.ndarray] = None,
        q0: Optional[np.ndarray] = None,
        T0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None,
        measurement_noise_std: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Simulates a tumbling trajectory of num_frames steps.
        """
        if y is None:
            _, y = self.generate_random_inertia_sample()

        if omega0 is None:
            # Angular rate 0.3 - 0.6 rad/s (Ma et al. Section 3.4)
            w_rand = np.random.uniform(-1.0, 1.0, size=3)
            w_dir = w_rand / np.linalg.norm(w_rand)
            rate = np.random.uniform(0.3, 0.6)
            omega0 = rate * w_dir

        if q0 is None:
            q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            q0 = q0 / np.linalg.norm(q0)

        if T0 is None:
            T0 = np.array([0.0, 0.0, 15.0], dtype=np.float64)

        if v0 is None:
            v0 = np.array([0.02, -0.01, -0.05], dtype=np.float64)

        times = np.zeros(num_frames, dtype=np.float64)
        quats_true = np.zeros((num_frames, 4), dtype=np.float64)
        quats_noisy = np.zeros((num_frames, 4), dtype=np.float64)
        omegas_true = np.zeros((num_frames, 3), dtype=np.float64)
        translations_true = np.zeros((num_frames, 3), dtype=np.float64)
        velocities_true = np.zeros((num_frames, 3), dtype=np.float64)

        curr_q = q0.copy()
        curr_w = omega0.copy()
        curr_T = T0.copy()
        curr_v = v0.copy()

        for k in range(num_frames):
            times[k] = k * self.dt
            quats_true[k] = curr_q
            omegas_true[k] = curr_w
            translations_true[k] = curr_T
            velocities_true[k] = curr_v

            # Add Gaussian observation noise to quaternion
            noise = np.random.normal(0.0, measurement_noise_std, size=4)
            q_noisy = curr_q + noise
            q_noisy /= np.linalg.norm(q_noisy)
            # Enforce sign continuity relative to true
            if np.dot(q_noisy, curr_q) < 0.0:
                q_noisy = -q_noisy
            quats_noisy[k] = q_noisy

            # Substep RK4 integration for high accuracy
            for _ in range(self.internal_substeps):
                curr_q, curr_w = rk4_step(curr_q, curr_w, y, self.sub_dt)

            # Translation step (linear drift + subtle orbital relative motion)
            curr_T = curr_T + curr_v * self.dt

        return {
            "timestamps": times,
            "quaternions_true": quats_true,
            "quaternions_noisy": quats_noisy,
            "angular_velocities": omegas_true,
            "translations": translations_true,
            "linear_velocities": velocities_true,
            "inertia_ratios": y
        }


def generate_batch_quaternion_dataset(
    num_sequences: int = 10000,
    seq_length: int = 500,
    dt: float = 0.1,
    noise_std: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates 10,000+ quaternion sequences to train the CFIN inertia network (Stage 3).
    Returns:
        all_quats: (num_sequences, seq_length, 4)
        all_y: (num_sequences, 2)
    """
    sim = SpacecraftDynamicsSimulator(dt=dt)
    all_quats = np.zeros((num_sequences, seq_length, 4), dtype=np.float32)
    all_y = np.zeros((num_sequences, 2), dtype=np.float32)

    for i in range(num_sequences):
        data = sim.simulate_sequence(num_frames=seq_length, measurement_noise_std=noise_std)
        all_quats[i] = data["quaternions_noisy"].astype(np.float32)
        all_y[i] = data["inertia_ratios"].astype(np.float32)

    return all_quats, all_y
