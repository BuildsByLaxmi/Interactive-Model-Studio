"""
Pipeline Executor Module (Person B)
The central orchestrator of the ML engine:
Chains Recipe Runner -> Model Training -> Evaluation.
Implements an in-memory transform cache to avoid recomputing preprocessing on repeat runs.
"""

import hashlib
import json
from typing import Any, Dict, Optional, Tuple, List
import pandas as pd

from ml_engine.recipe_runner import run_recipe
from ml_engine.training import train_model
from ml_engine.evaluation import evaluate_model

# Global cache mapping cache_key -> (X_train, X_test, y_train, y_test, fitted_transformers)
_TRANSFORM_CACHE: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]] = {}


def _generate_cache_key(df: pd.DataFrame, target_column: str, recipe: Any) -> str:
    """Generates a stable SHA-256 hash for dataset content + recipe configuration."""
    hasher = hashlib.sha256()
    meta_str = f"{len(df)}_{list(df.columns)}_{target_column}_"
    hasher.update(meta_str.encode("utf-8"))

    sample_bytes = pd.concat([df.head(10), df.tail(10)]).to_csv(index=False).encode("utf-8")
    hasher.update(sample_bytes)

    recipe_str = json.dumps(recipe, sort_keys=True) if isinstance(recipe, (list, dict)) else str(recipe)
    hasher.update(recipe_str.encode("utf-8"))

    return hasher.hexdigest()


def clear_pipeline_cache():
    """Clears the preprocessing transform cache."""
    global _TRANSFORM_CACHE
    _TRANSFORM_CACHE.clear()


def execute_pipeline(
    df: pd.DataFrame,
    config: Dict[str, Any],
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Executes an end-to-end ML pipeline run:
    1. Preprocessing Recipe Execution (with caching)
    2. Model Training with timing profiling
    3. Performance Evaluation (accuracy/f1/rmse, confusion matrix, feature importances)

    Args:
        df: Input raw pandas DataFrame.
        config: Dictionary containing target_column, task_type, recipe, and model_config.
        use_cache: Whether to use/populate preprocessed data cache.

    Returns:
        Dict with metrics, confusion_matrix, feature_importances, train_time_sec,
        fitted_model, and fitted_transformers.
    """
    target_column = config.get("target_column") or config.get("target_col")
    if not target_column:
        raise ValueError("Missing 'target_column' in pipeline configuration.")

    task_type = config.get("task_type", "classification")
    recipe = config.get("recipe", [])
    model_config = config.get("model_config", {})

    algorithm = model_config.get("algorithm") or model_config.get("model_name", "RandomForestClassifier")
    hyperparameters = model_config.get("hyperparameters", {})

    # 1. Check or populate transform cache
    cache_key = _generate_cache_key(df, target_column, recipe)
    cache_hit = False

    if use_cache and cache_key in _TRANSFORM_CACHE:
        X_train, X_test, y_train, y_test, fitted_transformers = _TRANSFORM_CACHE[cache_key]
        cache_hit = True
    else:
        X_train, X_test, y_train, y_test, fitted_transformers = run_recipe(
            df=df,
            recipe=recipe,
            target_column=target_column,
        )
        if use_cache:
            _TRANSFORM_CACHE[cache_key] = (X_train, X_test, y_train, y_test, fitted_transformers)

    # 2. Train model
    fitted_model, train_time_sec = train_model(
        X_train=X_train,
        y_train=y_train,
        model_name=algorithm,
        hyperparameters=hyperparameters,
        task_type=task_type,
    )

    # 3. Evaluate model
    eval_result = evaluate_model(
        fitted_model=fitted_model,
        X_test=X_test,
        y_test=y_test,
        task_type=task_type,
    )

    return {
        "metrics": eval_result["metrics"],
        "confusion_matrix": eval_result["confusion_matrix"],
        "feature_importances": eval_result["feature_importances"],
        "train_time_sec": train_time_sec,
        "fitted_model": fitted_model,
        "fitted_transformers": fitted_transformers,
        "cache_hit": cache_hit,
        "feature_names": list(X_train.columns),
    }
