"""
Unit and integration tests for DFineDetector using unittest.
"""

from pathlib import Path
import unittest
import cv2
import numpy as np
import torch

from integration.config import DetectorConfig, get_project_root
from integration.human_detector import DFineDetector, Detection


class TestDFineDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = DFineDetector()

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.model)
        self.assertIsInstance(self.detector.device, torch.device)

    def test_detector_missing_checkpoint(self):
        config = DetectorConfig(checkpoint_path="non_existent_model.pt")
        with self.assertRaises(FileNotFoundError):
            DFineDetector(config=config)

    def test_detector_dummy_frame(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(dummy_frame)
        self.assertIsInstance(detections, list)
        self.assertEqual(len(detections), 0)

    def test_detector_real_image_detection(self):
        sample_path = get_project_root() / "Depth-Anything-V2" / "assets" / "examples" / "demo01.jpg"
        self.assertTrue(sample_path.is_file(), f"Test image missing at {sample_path}")

        image = cv2.imread(str(sample_path))
        h, w = image.shape[:2]
        detections = self.detector.detect(image, confidence=0.4)

        self.assertGreater(len(detections), 0, "Expected at least 1 person detected in demo01.jpg")

        for det in detections:
            self.assertIsInstance(det, Detection)
            self.assertEqual(det.class_id, 0)
            self.assertEqual(det.label, "person")
            self.assertGreaterEqual(det.confidence, 0.4)
            self.assertLessEqual(det.confidence, 1.0)
            self.assertGreaterEqual(det.x1, 0.0)
            self.assertLess(det.x1, det.x2)
            self.assertLessEqual(det.x2, float(w))
            self.assertGreaterEqual(det.y1, 0.0)
            self.assertLess(det.y1, det.y2)
            self.assertLessEqual(det.y2, float(h))
            self.assertGreater(det.area, 0.0)


if __name__ == "__main__":
    unittest.main()
