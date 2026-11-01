"""
Unit and Integration Tests for Non-Cooperative Space Target Tracking Pipeline.
"""

import unittest
import numpy as np
import torch

from models.apparent_rotation import (
    gram_schmidt_6d_to_matrix,
    compute_corrective_rotation,
    true_to_apparent_rotation,
    apparent_to_true_rotation,
    compute_normalized_translation_targets,
    recover_absolute_translation,
    matrix_to_quaternion,
    quaternion_to_matrix,
    compute_pose_errors
)
from models.fastpose_vit import FastPoseViT
from models.frontend_saliency import SaliencyFrontEnd
from models.cfin_net import CFINNet
from simulation.space_stray_light import SpaceStrayLightSimulator
from simulation.space_augmentation import SpaceAugmentor
from simulation.dynamics_generator import SpacecraftDynamicsSimulator
from tracking.motion_smoother import PoseMotionSmoother, enforce_quaternion_sign_continuity
from tracking.dft_feature_extractor import QuaternionDFTFeatureExtractor
from tracking.nnekf import NNEKF


class TestPipeline(unittest.TestCase):

    def test_apparent_rotation_invertibility(self):
        T = torch.tensor([[2.0, -1.5, 12.0]], dtype=torch.float32)
        R_true = quaternion_to_matrix(torch.tensor([[0.8, 0.2, 0.4, 0.4]], dtype=torch.float32))

        # R' = Delta_R * R_true
        R_prime = true_to_apparent_rotation(R_true, T)
        # R_recovered = Delta_R^T * R'
        R_recovered = apparent_to_true_rotation(R_prime, T)

        self.assertTrue(torch.allclose(R_true, R_recovered, atol=1e-5))

    def test_translation_normalization_invertibility(self):
        T = torch.tensor([[1.2, -0.8, 14.5]], dtype=torch.float32)
        bbox = torch.tensor([[450.0, 320.0, 750.0, 620.0]], dtype=torch.float32)

        U = compute_normalized_translation_targets(T, bbox, 1920, 1200, 1920, 1920, 960, 600, 1.6)
        T_rec = recover_absolute_translation(U, bbox, 1920, 1200, 1920, 1920, 960, 600, 1.6)

        self.assertTrue(torch.allclose(T, T_rec, atol=1e-5))

    def test_gram_schmidt_orthogonality(self):
        r12 = torch.randn(4, 6)
        R = gram_schmidt_6d_to_matrix(r12)
        eye = torch.eye(3).unsqueeze(0).expand(4, 3, 3)
        R_RT = torch.bmm(R, R.transpose(1, 2))
        self.assertTrue(torch.allclose(R_RT, eye, atol=1e-5))

    def test_fastpose_vit_forward(self):
        model = FastPoseViT(img_size=224, embed_dim=192, depth=2, num_heads=4)
        x = torch.randn(2, 3, 224, 224)
        r12, U = model(x)
        self.assertEqual(r12.shape, (2, 6))
        self.assertEqual(U.shape, (2, 3))

    def test_spatial_rotation_analytical_update(self):
        aug = SpaceAugmentor(spatial_aug_prob=1.0)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = np.array([200, 150, 400, 350], dtype=np.float32)
        R = np.eye(3, dtype=np.float32)
        T = np.array([0.0, 0.0, 10.0], dtype=np.float32)

        rot_img, rot_bbox, R_new, T_new = aug.apply_spatial_inplane_rotation(img, bbox, R, T)
        # Z translation along optical axis should be invariant to Z-axis rotation
        self.assertAlmostEqual(T_new[2], 10.0, places=4)

    def test_stray_light_models(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 20
        moon = SpaceStrayLightSimulator.apply_moonlight_glare(img, moon_angle_deg=10.0)
        earth = SpaceStrayLightSimulator.apply_earth_atmosphere_radiation(img, off_axis_angle_deg=12.0)
        stars = SpaceStrayLightSimulator.apply_starlight_psf(img, num_stars=10)

        self.assertEqual(moon.shape, (100, 100, 3))
        self.assertEqual(earth.shape, (100, 100, 3))
        self.assertEqual(stars.shape, (100, 100, 3))

    def test_motion_smoother_sign_continuity(self):
        quats = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],  # Sign flip
            [1.0, 0.0, 0.0, 0.0]
        ])
        clean = enforce_quaternion_sign_continuity(quats)
        for i in range(len(clean) - 1):
            self.assertGreaterEqual(np.dot(clean[i], clean[i + 1]), 0.0)

    def test_dft_feature_extraction(self):
        sim = SpacecraftDynamicsSimulator(dt=0.1)
        data = sim.simulate_sequence(num_frames=100)
        extractor = QuaternionDFTFeatureExtractor(dt=0.1)
        xc = extractor.extract_features_single(data["quaternions_noisy"])
        self.assertEqual(len(xc), 15)

    def test_cfin_forward(self):
        cfin = CFINNet()
        dummy_in = torch.randn(3, 15)
        out = cfin(dummy_in)
        self.assertEqual(out.shape, (3, 2))
        # Verify physical ordering: 0 < Jxx <= Jyy <= 1
        y1, y2 = out[:, 0], out[:, 1]
        self.assertTrue((y1 > 0).all())
        self.assertTrue((y2 >= y1).all())
        self.assertTrue((y2 <= 1).all())

    def test_nnekf_predict_update_forecast(self):
        nnekf = NNEKF(dt=0.1)
        nnekf.predict()
        nnekf.update_attitude_vision(np.array([1.0, 0.0, 0.0, 0.0]))
        nnekf.update_inertia_virtual(np.array([0.6, 0.8]))

        forecast = nnekf.forecast_trajectory(duration_seconds=5.0, step_dt=0.1)
        self.assertEqual(forecast["forecast_quaternions"].shape, (50, 4))
        self.assertEqual(forecast["forecast_translations"].shape, (50, 3))


if __name__ == "__main__":
    unittest.main()
