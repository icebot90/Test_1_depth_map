"""
Integration tests for CombinedInferencePipeline using unittest.
"""

from pathlib import Path
import unittest
import cv2
import numpy as np

from integration.combined_inference import CombinedInferencePipeline
from integration.config import PipelineConfig, get_project_root


class TestCombinedPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config = PipelineConfig(no_display=True)
        cls.pipeline = CombinedInferencePipeline(config)

    def test_pipeline_dummy_frame(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        combined, depth, detections, distances, stats = self.pipeline.process_frame(dummy_frame)

        self.assertEqual(combined.shape, (480, 1280, 3))
        self.assertEqual(depth.shape, (480, 640))
        self.assertEqual(len(detections), 0)
        self.assertEqual(len(distances), 0)

        self.assertIn("det_ms", stats)
        self.assertIn("depth_ms", stats)
        self.assertIn("total_ms", stats)
        self.assertIn("fps", stats)
        self.assertGreater(stats["fps"], 0.0)

    def test_pipeline_real_image(self):
        sample_path = get_project_root() / "Depth-Anything-V2" / "assets" / "examples" / "demo01.jpg"
        image = cv2.imread(str(sample_path))
        h, w = image.shape[:2]

        combined, depth, detections, distances, stats = self.pipeline.process_frame(image)

        self.assertEqual(combined.shape, (h, 2 * w, 3))
        self.assertEqual(depth.shape, (h, w))
        self.assertGreater(len(detections), 0)
        self.assertEqual(len(detections), len(distances))

        for det, dist in zip(detections, distances):
            self.assertEqual(det.label, "person")
            if dist.is_valid:
                self.assertIsNotNone(dist.depth_m)
                self.assertGreater(dist.depth_m, 0.0)
                self.assertLessEqual(dist.depth_m, 80.0)


if __name__ == "__main__":
    unittest.main()
