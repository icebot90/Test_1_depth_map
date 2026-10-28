"""
Unit tests for PersonDistanceEstimator using unittest.
Uses controlled synthetic depth maps to verify geometric sampling,
filtering, median calculation, and edge case handling.
"""

import math
import unittest
import numpy as np

from integration.config import DistanceConfig
from integration.distance_estimator import DistanceResult, PersonDistanceEstimator
from integration.human_detector import Detection


class TestPersonDistanceEstimator(unittest.TestCase):

    def setUp(self):
        self.estimator = PersonDistanceEstimator(DistanceConfig(crop_ratio_x=0.5, crop_ratio_y=0.5))

    def test_distance_estimator_ideal_box(self):
        # 100x100 synthetic depth map with constant 5.0 meters
        depth_map = np.full((100, 100), 5.0, dtype=np.float32)
        det = Detection(box=(20.0, 20.0, 60.0, 60.0), confidence=0.9, class_id=0)

        result = self.estimator.estimate_person_depth(det, depth_map)
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.depth_m, 5.0, delta=0.05)
        self.assertIsNotNone(result.distance_m)
        self.assertGreaterEqual(result.distance_m, result.depth_m)
        self.assertEqual(result.status_message, "OK")
        self.assertGreater(result.valid_pixels, 0)

    def test_distance_estimator_central_sampling_filters_background(self):
        """
        Simulate a person at 3.0 meters surrounded by a background wall at 10.0 meters.
        The outer edges of the box contain the background (10.0m),
        while the center contains the person (3.0m).
        Central ROI sampling should extract only the person's depth.
        """
        # Create 100x100 background at 10m
        depth_map = np.full((100, 100), 10.0, dtype=np.float32)
        # Put person at 3m in central 30..70
        depth_map[30:70, 30:70] = 3.0

        # Person bounding box covers 20..80 (which includes some 10m background)
        det = Detection(box=(20.0, 20.0, 80.0, 80.0), confidence=0.95, class_id=0)

        result = self.estimator.estimate_person_depth(det, depth_map)
        self.assertTrue(result.is_valid)
        # The central 50% of 20..80 is 35..65, which is purely inside 30..70 (3.0m)
        self.assertAlmostEqual(result.depth_m, 3.0, delta=0.05)

    def test_distance_estimator_clipping_and_out_of_bounds(self):
        depth_map = np.full((100, 100), 4.2, dtype=np.float32)
        det = Detection(box=(-50.0, -20.0, 50.0, 50.0), confidence=0.8, class_id=0)
        result = self.estimator.estimate_person_depth(det, depth_map)
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.depth_m, 4.2, delta=0.05)

    def test_distance_estimator_nan_and_inf_handling(self):
        depth_map = np.full((100, 100), np.nan, dtype=np.float32)
        depth_map[40:60, 40:60] = 6.5

        det = Detection(box=(20.0, 20.0, 80.0, 80.0), confidence=0.85, class_id=0)
        result = self.estimator.estimate_person_depth(det, depth_map)
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.depth_m, 6.5, delta=0.05)

    def test_distance_estimator_all_invalid_depths(self):
        depth_map = np.full((100, 100), np.nan, dtype=np.float32)
        det = Detection(box=(20.0, 20.0, 80.0, 80.0), confidence=0.85, class_id=0)

        result = self.estimator.estimate_person_depth(det, depth_map)
        self.assertFalse(result.is_valid)
        self.assertIsNone(result.depth_m)
        self.assertIn("Insufficient valid pixels", result.status_message)

    def test_distance_estimator_tiny_box_rejected(self):
        estimator = PersonDistanceEstimator(DistanceConfig(min_box_area=100.0))
        depth_map = np.full((100, 100), 5.0, dtype=np.float32)

        det = Detection(box=(20.0, 20.0, 24.0, 24.0), confidence=0.7, class_id=0)
        result = estimator.estimate_person_depth(det, depth_map)
        self.assertFalse(result.is_valid)
        self.assertIn("too small", result.status_message)

    def test_distance_estimator_invalid_depth_map(self):
        det = Detection(box=(20.0, 20.0, 80.0, 80.0), confidence=0.85, class_id=0)
        result = self.estimator.estimate_person_depth(det, None)
        self.assertFalse(result.is_valid)
        self.assertIsNone(result.depth_m)
        self.assertIsNone(result.distance_m)

    def test_distance_estimator_euclidean_distance_greater_than_depth(self):
        """
        An off-center object at (u != cx, v != cy) should have Euclidean distance R > Z.
        """
        depth_map = np.full((480, 640), 10.0, dtype=np.float32)
        # Person placed at far right corner of the frame: x=(500, 600), y=(350, 450)
        det = Detection(box=(500.0, 350.0, 600.0, 450.0), confidence=0.9, class_id=0)
        result = self.estimator.estimate_person_depth(det, depth_map)
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.depth_m, 10.0, delta=0.05)
        self.assertGreater(result.distance_m, result.depth_m)
        self.assertIsNotNone(result.coords_3d)
        x, y, z = result.coords_3d
        expected_r = math.sqrt(x*x + y*y + z*z)
        self.assertAlmostEqual(result.distance_m, expected_r, delta=0.05)


if __name__ == "__main__":
    unittest.main()
