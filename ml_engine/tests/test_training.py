"""
Unit Tests for ML Engine Training and Execution Pipeline (Person B)
"""

import unittest
import numpy as np
import pandas as pd
from ml_engine.pipeline_executor import execute_pipeline
from ml_engine.training import train_model


class TestTraining(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(5, 2, n)
        y = (x1 + x2 > 5).astype(int)
        self.df = pd.DataFrame({"feature1": x1, "feature2": x2, "target": y})

    def test_train_model_basic(self):
        X = self.df[["feature1", "feature2"]]
        y = self.df["target"]
        model, train_time = train_model(X, y, "RandomForestClassifier", {"n_estimators": 20})
        self.assertIsNotNone(model)
        self.assertGreater(train_time, 0.0)

    def test_pipeline_execution(self):
        config = {
            "task_type": "classification",
            "target_column": "target",
            "recipe": [
                {"step": "scale", "params": {"columns": ["feature1", "feature2"], "method": "standard"}},
                {"step": "split", "params": {"test_size": 0.2, "random_state": 42}}
            ],
            "model_config": {
                "algorithm": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 25, "random_state": 42}
            }
        }
        res = execute_pipeline(self.df, config, use_cache=False)
        self.assertIn("metrics", res)
        self.assertIn("accuracy", res["metrics"])
        self.assertIn("confusion_matrix", res)


if __name__ == "__main__":
    unittest.main()
