"""
Unified test runner for D-FINE Human Detection + Depth Anything V2 Integration.
Discovers and executes all unit and integration tests.
"""

import sys
import unittest
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(project_root / "tests"), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(0 if result.wasSuccessful() else 1)
