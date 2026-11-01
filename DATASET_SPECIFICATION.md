# Non-Cooperative Space Target Dataset Specification for Blender Generation

This document defines the complete specification for generating synthetic space rendezvous and debris tracking datasets in Blender (or equivalent orbital simulator) for 6DoF pose estimation, motion tracking, inertia identification, and trajectory forecasting through dropouts.

---

## 1. Directory Structure

The generated dataset should be organized into sequences and standardized train/validation/test splits:

```
dataset_root/
├── camera_intrinsics.json               # Global camera calibration parameters
├── train/
│   ├── seq_001/
│   │   ├── images/
│   │   │   ├── frame_000000.png
│   │   │   ├── frame_000001.png
│   │   │   └── ...
│   │   ├── annotations.json             # Ground truth per frame
│   │   └── metadata.json                # Inertia properties and simulation config
│   ├── seq_002/
│   └── ...
├── val/
│   └── ...
├── test_nominal/                        # Standard orbital lighting (deep space background)
│   └── ...
├── test_harsh_straylight/               # Stress test: Earth atmospheric glow, Moon glare, solar flares
│   └── ...
└── test_dropouts/                       # Sequences with injected camera dropouts (umbra eclipse, FOV loss)
    └── ...
```

---

## 2. Coordinate System Conventions

To ensure mathematical consistency across all pipeline stages:

### 2.1 Camera Frame ($\mathcal{F}_c$)
- Follows the standard OpenCV / Computer Vision convention:
  - **$+X_c$**: Points to the camera's **right**
  - **$+Y_c$**: Points **downward**
  - **$+Z_c$**: Points **forward** along the optical axis (viewing direction)
  - Origin $(0, 0, 0)$ is at the camera optical center.

### 2.2 Target Body Frame ($\mathcal{F}_b$)
- The origin is placed strictly at the **Center of Mass (CoM)** of the target spacecraft/debris.
- Axes are aligned with the target's **principal axes of inertia**:
  $$J = \begin{bmatrix} J_{xx} & 0 & 0 \\ 0 & J_{yy} & 0 \\ 0 & 0 & J_{zz} \end{bmatrix}$$
  where principal moments are ordered: $0 < J_{xx} \le J_{yy} \le J_{zz}$.

### 2.3 Pose Transformation
- The 6DoF pose maps points from the target body frame $\mathcal{F}_b$ into the camera frame $\mathcal{F}_c$:
  $$P_c = R_{b \to c} P_b + T_{b \to c}$$
  - $T = [X, Y, Z]^T$: Translation of the target CoM in camera coordinates (in meters).
  - $R$: $3 \times 3$ rotation matrix mapping vectors from $\mathcal{F}_b$ to $\mathcal{F}_c$.
  - Quaternion $q = [q_w, q_x, q_y, q_z]$ (scalar-first convention, unit norm $\|q\| = 1$).

---

## 3. Camera Parameters (`camera_intrinsics.json`)

The camera model is a standard pinhole camera:

```json
{
  "camera_name": "ServicingCraft_Navigation_Cam",
  "image_width": 1920,
  "image_height": 1200,
  "aspect_ratio_correction_factor": 1.6,
  "fx": 1920.0,
  "fy": 1920.0,
  "cx": 960.0,
  "cy": 600.0,
  "distortion_model": "none",
  "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
  "fps": 10.0,
  "exposure_time_ms": 15.0
}
```

*Note: For testing and lightweight training, images can also be rendered at $1024 \times 1024$ or $512 \times 512$ with corresponding $f_x, f_y, c_x, c_y$.*

---

## 4. Annotations Schema (`annotations.json`)

Each sequence must have an `annotations.json` file structured as follows:

```json
{
  "sequence_id": "seq_001",
  "num_frames": 500,
  "fps": 10.0,
  "inertia_properties": {
    "Jxx": 400.0,
    "Jyy": 500.0,
    "Jzz": 600.0,
    "inertia_ratios": [0.666667, 0.833333],
    "notes": "Jxx/Jzz and Jyy/Jzz ordered such that Jxx <= Jyy <= Jzz"
  },
  "frames": [
    {
      "frame_idx": 0,
      "timestamp_s": 0.0,
      "image_file": "images/frame_000000.png",
      "visible": true,
      "bounding_box_xywh": [450.2, 280.5, 320.0, 310.0],
      "bounding_box_xyxy": [450.2, 280.5, 770.2, 590.5],
      "translation_xyz": [0.45, -0.32, 12.85],
      "quaternion_wxyz": [0.92388, 0.38268, 0.0, 0.0],
      "rotation_matrix_3x3": [
        [1.0, 0.0, 0.0],
        [0.0, 0.7071, -0.7071],
        [0.0, 0.7071, 0.7071]
      ],
      "linear_velocity_xyz": [0.02, -0.01, -0.15],
      "angular_velocity_body_xyz": [0.35, 0.12, -0.28],
      "stray_light_conditions": {
        "earth_off_axis_angle_deg": 45.0,
        "moon_angle_deg": 65.0,
        "sun_glare_present": false,
        "in_eclipse": false
      }
    }
  ]
}
```

---

## 5. Physical Dynamics Requirements

To enable Stage 3 (Inertia Identification via CFIN) and Stage 4 (NNEKF Forecasting):

1. **Torque-Free Rotational Tumbling**:
   Motion should be generated via Euler's equations of motion for a rigid body in orbit:
   $$\dot{\omega}_x = \frac{J_{yy} - J_{zz}}{J_{xx}} \omega_y \omega_z$$
   $$\dot{\omega}_y = \frac{J_{zz} - J_{xx}}{J_{yy}} \omega_x \omega_z$$
   $$\dot{\omega}_z = \frac{J_{xx} - J_{yy}}{J_{zz}} \omega_x \omega_y$$
2. **Attitude Kinematics**:
   $$\dot{q} = \frac{1}{2} q \circ \begin{bmatrix} 0 \\ \omega_b \end{bmatrix}$$
3. **Simulation Step Size**:
   - Integrate dynamics at high frequency ($\Delta t \le 0.005$ s / $200$ Hz) using 4th-order Runge-Kutta (RK4), then downsample to the sensor frame rate (e.g., 10 Hz or 20 Hz).
4. **Tumbling Rates**:
   - Initial angular velocity magnitude $\|\omega_0\| \in [0.1, 0.6]$ rad/s (typical for tumbling debris and uncontrolled upper stages).
5. **Inertia Tensor Variations**:
   - Sample random masses and shapes to produce varied inertia ratios $y_1 = J_{xx}/J_{zz} \in [0.2, 0.95]$ and $y_2 = J_{yy}/J_{zz} \in [y_1, 0.98]$.
6. **Relative Orbital Translation**:
   - Chaser-to-target distance ranging from $5$ m to $50$ m.
   - Smooth relative drift (linear velocity $\|\mathbf{v}\| \le 0.5$ m/s).

---

## 6. Photorealistic Space Lighting & Rendering Specifications

Space imagery is characterized by extreme dynamic range and stray light. In Blender (Cycles or EEVEE-Next):

### 6.1 Direct Sunlight
- Sun lamp modeled as a distant directional collimated source:
  - Energy: $>1360 \text{ W/m}^2$.
  - Color temperature: $5778\text{ K}$ (crisp white).
  - Hard shadow termination with near-zero ambient fill in deep space.

### 6.2 Target Materials
- **Multi-Layer Insulation (MLI)**: Specular golden and silver foil with normal map ripples and micro-wrinkles, causing specular spikes and glare glints.
- **Solar Panels**: Dark blue/black specular silicon cells with reflective metallic busbars.
- **Spacecraft Body**: Matte and semi-gloss white/gray thermal coatings.

### 6.3 Stray Light Environments (Sun et al. Aerospace 2023)
- **Moonlight Simulation**:
  - When the Moon is within off-axis angle $\sigma < 30^\circ$, inject a bright Gaussian halo / glare blooming into the sensor.
- **Earth Atmospheric Limb & Albedo**:
  - When looking towards the Earth horizon (off-axis angle $\theta < 22^\circ$), inject soft blue/white atmospheric glow, and Earth albedo illuminating the shadow side of the satellite.
- **Background Star Field**:
  - Star point sources rendered with optical system Point Spread Function (diffraction spikes / Gaussian blur).

### 6.4 Dropouts and Occlusions (for Stage 4 Testing)
- **Umbra / Penumbra Eclipse**: Spacecraft passing into Earth's shadow (illumination rapidly drops to 0).
- **Sun Blinding**: Sun directly entering camera field of view, causing saturated blowout.
- **Field of View Exits**: Target drifting out of frame for 5–30 seconds.

---

## 7. Recommended Target Models

- **Tango / PRISMA** (Standard ESA satellite model from SPEED/SPEED+ benchmark).
- **Envisat** (Large ESA Earth observation satellite, classic space debris scenario).
- **Ariane / Centaur Upper Stage** (Cylindrical rocket body).
- **CubeSat 3U/6U** with deployable solar panels.

---

*This specification guarantees full compatibility with the FastPose-ViT pose estimator, Stage 2 motion smoother, Stage 3 CFIN inertia network, and Stage 4 NNEKF forecasting filter.*
