# 6DoF Non-Cooperative Spacecraft Pose Estimation, Motion Tracking & Trajectory Forecasting

A complete end-to-end vision and dynamical estimation pipeline for autonomous rendezvous, capture, and tracking of uncooperative space targets (dead satellites, rocket bodies, and debris) tumbling in orbit under harsh space lighting.

---

## Architecture Overview

```
                      +---------------------------------------+
                      | Raw Space Imagery (Collimated Sun,   |
                      | Deep Shadows, Specular Foil, Stray)  |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Frontend Visual Saliency (Shi/Ulrich) |
                      |  DoG Filtering + Spectral Residual    |
                      +-------------------+-------------------+
                                          | Spacecraft Bounding Box
                                          v
                      +---------------------------------------+
                      | Stage 1: FastPose-ViT (Direct Reg.)   |
                      |  - Apparent Rotation R' Decoupling    |
                      |  - Normalized Translation U = (Ux,Uy,Uz)
                      |  - Rodrigues Perspective Correction   |
                      +-------------------+-------------------+
                                          | Raw Pose (T_k, q_k)
                                          v
                      +---------------------------------------+
                      | Stage 2: Motion Smoothing & Kinematics|
                      |  - Quaternion Sign-Continuity         |
                      |  - Savitzky-Golay / Polynomial Filter |
                      |  - Angular Velocity w_b & Linear Vel  |
                      +-------------------+-------------------+
                                          | Smoothed Poses & Velocities
                                          v
                      +---------------------------------------+
                      | Stage 3: Inertia Ratio Estim. (CFIN)  |
                      |  - DFT on Quaternion Time Series      |
                      |  - 3 Dominant Peak Frequencies        |
                      |  - 15-dim Characteristic Features     |
                      |  - Predicts y = [Jxx/Jzz, Jyy/Jzz]    |
                      +-------------------+-------------------+
                                          | Virtual Observation y_cfin
                                          v
                      +---------------------------------------+
                      | Stage 4: NNEKF State Estimation       |
                      |  - 8-State EKF: [qv(3), w(3), y(2)]   |
                      |  - Euler Dynamics Propagation         |
                      |  - Dual Vision & Virtual Updates      |
                      |  - Forward Prediction Through Dropouts|
                      +---------------------------------------+
```

---

## Mathematical Formulations

### Stage 1: FastPose-ViT Direct Regression
- **Normalized Translation**:
  $$s_x = \frac{W}{\alpha w}, \quad s_y = \frac{H}{h}$$
  $$U_z = \frac{1}{2}\left(\frac{1}{s_x} + \frac{1}{s_y}\right) Z \implies Z = \frac{U_z}{\frac{1}{2}(s_x^{-1} + s_y^{-1})}$$
  $$U_x = \frac{1}{w}\left(\frac{f_x X}{Z} + c_x - b_x\right), \quad U_y = \frac{1}{h}\left(\frac{f_y Y}{Z} + c_y - b_y\right)$$
- **Apparent Rotation Decoupling**:
  $$\vec{T} = \frac{T}{\|T\|}, \quad \mathbf{u} = \frac{\vec{T} \times \mathbf{e}_z}{\|\vec{T} \times \mathbf{e}_z\|}, \quad \theta = \arccos(\vec{T}^T \mathbf{e}_z)$$
  $$\Delta R(\vec{T}) = I + \sin\theta [\mathbf{u}]_\times + (1 - \cos\theta) [\mathbf{u}]_\times^2$$
  $$R_{pred} = \Delta R(\vec{T})^T R'_{pred}$$
- **Training Loss**:
  $$\mathcal{L} = \|R'_{pred, 1:2} - R'_{gt, 1:2}\|_F^2 + \|U_{pred} - U_{gt}\|_2^2$$

### Stage 2: Motion Smoother
- **Sign Continuity**: If $q_k \cdot q_{k-1} < 0 \implies q_k \leftarrow -q_k$.
- **Angular Velocity**:
  $$\begin{bmatrix} 0 \\ \boldsymbol{\omega}_b \end{bmatrix} = 2 q^* \circ \dot{q}$$

### Stage 3: CFIN Inertia Identification (Ma et al.)
- **15-dim Characteristic Frequency Vector**:
  $$x_c = [f_1, f_2, f_3, p_{01}, p_{02}, \dots, p_{33}]^T \in \mathbb{R}^{15}$$
  Extracted from 3 dominant DFT peaks of the 4 quaternion components.
- **CFIN Network**:
  $$\tilde{x}_c = \sigma^{-1}(x_c - \mu) \to \text{FC}(15, 100) \to \tanh / \sinh \to \text{FC}(100, 2) \to [J_{xx}/J_{zz}, J_{yy}/J_{zz}]^T$$

### Stage 4: Neural-Network-Enhanced Kalman Filter (NNEKF)
- **State Vector**: $\mathbf{x} = [\mathbf{q}_v^T, \boldsymbol{\omega}^T, \mathbf{y}^T]^T \in \mathbb{R}^8$.
- **Process Dynamics**:
  $$\dot{\mathbf{q}}_v = \frac{1}{2}(\mathbf{q}_0 \boldsymbol{\omega} + \boldsymbol{\omega} \times \mathbf{q}_v)$$
  $$\dot{\omega}_x = \frac{y_2 - 1}{y_1} \omega_y \omega_z, \quad \dot{\omega}_y = \frac{1 - y_1}{y_2} \omega_x \omega_z, \quad \dot{\omega}_z = (y_1 - y_2) \omega_x \omega_y$$
- **Dual Update**:
  1. Camera measurement: updates quaternion error vector $\Delta \mathbf{q}_v$ and $\boldsymbol{\omega}$.
  2. Virtual observation: updates inertia ratios $\mathbf{y}$ from CFIN.
- **Dropout Forecasting**: During camera outage (target lost in eclipse or sun glare), the filter propagates states forward via torque-free Euler dynamics, achieving ~1/4 the drift accumulation rate of a plain EKF.

---

## File Directory

- `DATASET_SPECIFICATION.md`: Precise specification for friend generating Blender simulation sequences.
- `configs/default_config.yaml`: Pipeline configuration and hyperparameters.
- `models/fastpose_vit.py`: Vision Transformer architecture with direct 6DoF regression.
- `models/apparent_rotation.py`: Analytical Rodrigues apparent rotation and crop-to-fullframe recovery.
- `models/frontend_saliency.py`: Shi & Ulrich grayscale saliency and background clutter rejection.
- `models/cfin_net.py`: Characteristic Frequency Input Network (Stage 3).
- `simulation/space_stray_light.py`: Sun et al. Aerospace 2023 stray-light models (Moon, Earth atmosphere, stars).
- `simulation/space_augmentation.py`: Space illumination augmentations and spatial in-plane rotations.
- `simulation/dynamics_generator.py`: Euler rotational integrator and 10k rendering-free CFIN dataset generator.
- `tracking/motion_smoother.py`: Stage 2 sign-continuity and kinematics velocity derivation.
- `tracking/dft_feature_extractor.py`: 15-dim FFT peak extraction and whitening.
- `tracking/nnekf.py`: Stage 4 NNEKF filter and dropout forecasting.
- `train_fastpose.py`: Stage 1 training script.
- `train_cfin.py`: Stage 3 CFIN training script.
- `evaluate.py`: End-to-end evaluation with camera dropout injection and benchmark comparisons.
- `tests/test_pipeline.py`: Unit and integration test suite.

---

## Quickstart Commands

```bash
# 1. Run unit test suite
python -m unittest tests/test_pipeline.py

# 2. Train Stage 3 CFIN Inertia Estimator (runs rendering-free dynamics generator)
python train_cfin.py --num_samples 2000 --seq_len 250 --epochs 30

# 3. Train Stage 1 FastPose-ViT (with space augmentations and apparent rotation)
python train_fastpose.py --epochs 15 --batch_size 8

# 4. Run End-to-End Evaluation & Dropout Benchmark
python evaluate.py --num_frames 150 --dropout_start 70 --dropout_end 110
```
