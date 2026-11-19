"""
Training Module (Person B)
Instantiates and fits machine learning models, tracking precise training execution times.
"""

import time
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from ml_engine.registry import get_model_class, get_model_defaults


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None,
) -> Tuple[Any, float]:
    """
    Instantiates and trains a model using provided features and labels.

    Args:
        X_train: Preprocessed training features DataFrame.
        y_train: Training target Series.
        model_name: Algorithm key defined in registry (e.g. 'RandomForestClassifier').
        hyperparameters: Dictionary of hyperparameters to override defaults.
        task_type: 'classification' or 'regression'.

    Returns:
        fitted_model: Trained scikit-learn / XGBoost estimator object.
        train_time (float): Training duration in seconds.
    """
    model_cls = get_model_class(model_name, task_type=task_type)

    params = get_model_defaults(model_name)
    if hyperparameters:
        params.update(hyperparameters)

    # Clean out non-model params if passed by mistake
    clean_params = {k: v for k, v in params.items() if not k.startswith("_")}

    try:
        estimator = model_cls(**clean_params)
    except TypeError:
        # Fall back with basic defaults if hyperparameter mismatch occurs
        estimator = model_cls(random_state=42) if "random_state" in model_cls.__init__.__code__.co_varnames else model_cls()

    start_time = time.perf_counter()
    estimator.fit(X_train, y_train)
    elapsed_time = time.perf_counter() - start_time

    return estimator, round(elapsed_time, 4)
