"""
Reproducibility Verification Module (Person B)
Re-runs an ML experiment using stored pipeline configurations
and verifies whether outputs match historical records within numerical tolerance.
"""

from typing import Any, Dict, Optional
import pandas as pd
from ml_engine.pipeline_executor import execute_pipeline


def verify_reproducibility(
    df: pd.DataFrame,
    original_config: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    original_metrics: Optional[Dict[str, float]] = None,
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """
    Re-runs an experiment from raw configuration and verifies output reproducibility.
    """
    active_config = original_config or config
    if not active_config:
        raise ValueError("Must provide original_config to verify reproducibility.")

    active_metrics = original_metrics or {}

    # Force uncached execution to test true reproducibility
    reproduced_result = execute_pipeline(df=df, config=active_config, use_cache=False)
    new_metrics = reproduced_result.get("metrics", {})

    discrepancies = {}
    is_reproducible = True

    for metric_name, expected_val in active_metrics.items():
        if metric_name not in new_metrics:
            is_reproducible = False
            discrepancies[metric_name] = {
                "expected": expected_val,
                "reproduced": None,
                "difference": None,
                "status": "MISSING_IN_NEW_RUN",
            }
            continue

        reproduced_val = new_metrics[metric_name]
        diff = abs(float(reproduced_val) - float(expected_val))
        passed = diff <= tolerance

        if not passed:
            is_reproducible = False

        discrepancies[metric_name] = {
            "expected": expected_val,
            "reproduced": reproduced_val,
            "difference": round(diff, 6),
            "status": "PASS" if passed else "FAIL",
        }

    return {
        "is_reproducible": is_reproducible,
        "discrepancies": discrepancies,
        "new_metrics": new_metrics,
        "new_confusion_matrix": reproduced_result.get("confusion_matrix"),
        "new_train_time_sec": reproduced_result.get("train_time_sec"),
    }
