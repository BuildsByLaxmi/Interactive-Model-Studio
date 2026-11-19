"""
Unit Tests for Reproducibility Verification Engine (Person B)
"""

import unittest
import numpy as np
import pandas as pd
from ml_engine.pipeline_executor import execute_pipeline
from ml_engine.reproduce import verify_reproducibility


class TestReproducibility(unittest.TestCase):

    def setUp(self):
        """Creates sample dataset for reproducibility tests."""
        np.random.seed(100)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(2, 1, n)
        y = (x1 + x2 > 2).astype(int)

        self.df = pd.DataFrame({"x1": x1, "x2": x2, "label": y})

        self.config = {
            "task_type": "classification",
            "target_column": "label",
            "recipe": [
                {"step": "split", "params": {"test_size": 0.2, "random_state": 42}},
                {"step": "scale", "params": {"columns": ["x1", "x2"], "method": "standard"}}
            ],
            "model_config": {
                "algorithm": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 30, "random_state": 42}
            }
        }

    def test_reproducibility_success(self):
        """Runs pipeline once, then verifies that verify_reproducibility passes with original metrics."""
        first_run = execute_pipeline(self.df, self.config, use_cache=False)
        original_metrics = first_run["metrics"]

        check = verify_reproducibility(self.df, self.config, original_metrics=original_metrics, tolerance=1e-4)
        self.assertTrue(check["is_reproducible"])

    def test_reproducibility_detects_mismatch(self):
        """Altered metrics should cause verify_reproducibility to report failure."""
        first_run = execute_pipeline(self.df, self.config, use_cache=False)
        tampered_metrics = first_run["metrics"].copy()
        tampered_metrics["accuracy"] = 0.01

        check = verify_reproducibility(self.df, self.config, original_metrics=tampered_metrics, tolerance=1e-4)
        self.assertFalse(check["is_reproducible"])


if __name__ == "__main__":
    unittest.main()
