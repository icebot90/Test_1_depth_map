"""
Unit and integration tests for DepthAnythingV2Estimator using unittest.
"""

from pathlib import Path
import unittest
import cv2
import numpy as np
import torch

from integration.config import DepthConfig, get_project_root
from integration.depth_estimator import DepthAnythingV2Estimator


class TestDepthAnythingV2Estimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.estimator = DepthAnythingV2Estimator()

    def test_depth_estimator_initialization(self):
        self.assertIsNotNone(self.estimator.model)

    def test_depth_estimator_missing_checkpoint(self):
        config = DepthConfig(checkpoint_path="non_existent_depth.pth")
        with self.assertRaises(FileNotFoundError):
            DepthAnythingV2Estimator(config=config)

    def test_depth_estimator_dummy_inference(self):
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        depth = self.estimator.estimate_depth(dummy, input_size=392)

        self.assertIsInstance(depth, np.ndarray)
        self.assertEqual(depth.shape, (240, 320))
        self.assertEqual(depth.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(depth)))
        self.assertGreater(float(depth.min()), 0.0)

    def test_depth_estimator_real_image(self):
        sample_path = get_project_root() / "Depth-Anything-V2" / "assets" / "examples" / "demo01.jpg"
        self.assertTrue(sample_path.is_file(), f"Test image missing at {sample_path}")

        image = cv2.imread(str(sample_path))
        h, w = image.shape[:2]
        depth = self.estimator.estimate_depth(image, input_size=518)

        self.assertEqual(depth.shape, (h, w))
        self.assertEqual(depth.dtype, np.float32)
        self.assertGreater(float(depth.min()), 0.0)
        self.assertLessEqual(float(depth.max()), 80.0)


if __name__ == "__main__":
    unittest.main()
