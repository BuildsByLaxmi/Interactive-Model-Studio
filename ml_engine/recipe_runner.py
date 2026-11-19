"""
Recipe Runner Module (Person B)
Executes ordered preprocessing steps deterministically on a dataset.
Prevents data leakage by fitting transformers strictly on training data.
Supports:
- Data Cleaning (Imputation, Drop Columns)
- Outlier & Noise Handling (IQR Clipping, Variance Threshold)
- Transformations (Scaling, Encoding, Math/Power Transforms, Binning)
- Dimension Reduction (PCA, Feature Selection)
- Data Splitting
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from ml_engine.preprocessing import (
    apply_split,
    apply_train_val_test_split,
    apply_imputation,
    apply_outlier_clipping,
    apply_scaling,
    apply_encoding,
    apply_power_transform,
    apply_log1p_transform,
    apply_binning,
    apply_pca_reduction,
    apply_feature_selection,
    apply_variance_threshold,
)


def run_recipe(
    df: pd.DataFrame,
    recipe: Union[List[Dict[str, Any]], Dict[str, Any]],
    target_column: Optional[str] = None,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[Dict[str, Any]]]:
    """
    Executes a sequence of preprocessing steps on a raw DataFrame without data leakage.

    Supports both list-based recipes:
    [
        {"step": "impute", "params": {"strategy": "mean", "columns": [...]}},
        {"step": "outlier_clip", "params": {"threshold": 1.5, "columns": [...]}},
        {"step": "scale", "params": {"method": "standard", "columns": [...]}},
        {"step": "pca", "params": {"n_components": 2}},
        {"step": "split", "params": {"test_size": 0.2, "random_state": 42}}
    ]

    And chained-dict recipes:
    {
        "chains": {
            "stage_1_cleaning": [...],
            "stage_2_transformation": [...]
        }
    }
    """
    resolved_target = target_column or target_col
    if not resolved_target:
        raise ValueError("Target column must be specified.")

    if resolved_target not in df.columns:
        raise ValueError(f"Target column '{resolved_target}' not found in DataFrame.")

    # Flatten chained recipe if necessary
    flat_steps = []
    if isinstance(recipe, dict) and "chains" in recipe:
        for chain_key in ["stage_1_cleaning", "stage_2_transformation", "stage_3_reduction"]:
            flat_steps.extend(recipe["chains"].get(chain_key, []))
    elif isinstance(recipe, list):
        flat_steps = recipe
    else:
        flat_steps = []

    # 1. Split identification (default 80/20)
    split_params = {"test_size": 0.2, "random_state": 42, "stratify": False}
    feature_steps = []

    for step_cfg in flat_steps:
        step_name = step_cfg.get("step", "").lower()
        if step_name == "split":
            split_params.update(step_cfg.get("params", step_cfg))
        else:
            feature_steps.append(step_cfg)

    # 2. Initial Data Split to prevent leakage across all steps
    X_train, X_test, y_train, y_test = apply_split(
        df=df,
        target_column=resolved_target,
        test_size=float(split_params.get("test_size", 0.2)),
        random_state=int(split_params.get("random_state", 42)),
        stratify=bool(split_params.get("stratify", False)),
    )

    fitted_transformers: List[Dict[str, Any]] = []

    # 3. Sequentially apply pipeline steps
    for step_cfg in feature_steps:
        step_name = step_cfg.get("step", "").lower()
        params = step_cfg.get("params", step_cfg)
        columns = params.get("columns", [])

        # Drop columns
        if step_name in ["drop", "drop_columns"]:
            cols_to_drop = [c for c in columns if c in X_train.columns]
            if cols_to_drop:
                X_train = X_train.drop(columns=cols_to_drop)
                X_test = X_test.drop(columns=cols_to_drop)
                fitted_transformers.append({"step": "drop", "columns": cols_to_drop})

        # Impute missing values
        elif step_name == "impute":
            strategy = params.get("strategy", "mean")
            fill_value = params.get("fill_value", None)
            X_train, X_test, transformer = apply_imputation(
                X_train, X_test, columns=columns, strategy=strategy, fill_value=fill_value
            )
            fitted_transformers.append({
                "step": "impute",
                "columns": columns,
                "strategy": strategy,
                "transformer": transformer,
            })

        # Outlier clipping & winsorization
        elif step_name in ["outlier_clip", "iqr_outlier_clipping", "outliers"]:
            method = params.get("method", "iqr")
            threshold = float(params.get("threshold", 1.5))
            X_train, X_test, bounds = apply_outlier_clipping(
                X_train, X_test, columns=columns, method=method, threshold=threshold
            )
            fitted_transformers.append({
                "step": "outlier_clip",
                "columns": columns,
                "bounds": bounds,
            })

        # Scaling
        elif step_name == "scale":
            method = params.get("method", "standard")
            X_train, X_test, transformer = apply_scaling(
                X_train, X_test, columns=columns, method=method
            )
            fitted_transformers.append({
                "step": "scale",
                "columns": columns,
                "method": method,
                "transformer": transformer,
            })

        # Encoding
        elif step_name == "encode":
            method = params.get("method", "onehot")
            drop = params.get("drop", None)
            X_train, X_test, transformer = apply_encoding(
                X_train, X_test, columns=columns, method=method, drop=drop
            )
            fitted_transformers.append({
                "step": "encode",
                "columns": columns,
                "method": method,
                "transformer": transformer,
            })

        # Power Transforms
        elif step_name in ["power_transform", "yeo_johnson", "math_transform"]:
            method = params.get("method", "yeo-johnson")
            if method == "log1p":
                X_train, X_test, transformer = apply_log1p_transform(X_train, X_test, columns=columns)
            else:
                X_train, X_test, transformer = apply_power_transform(X_train, X_test, columns=columns, method="yeo-johnson")
            fitted_transformers.append({
                "step": "power_transform",
                "columns": columns,
                "transformer": transformer,
            })

        # Binning / Discretization
        elif step_name in ["binning", "discretize"]:
            n_bins = int(params.get("n_bins", 5))
            strategy = params.get("strategy", "quantile")
            X_train, X_test, transformer = apply_binning(
                X_train, X_test, columns=columns, n_bins=n_bins, strategy=strategy
            )
            fitted_transformers.append({
                "step": "binning",
                "columns": columns,
                "transformer": transformer,
            })

        # Dimension Reduction: PCA
        elif step_name in ["pca", "dimension_reduction"]:
            n_components = int(params.get("n_components", 2))
            X_train, X_test, transformer = apply_pca_reduction(
                X_train, X_test, n_components=n_components
            )
            fitted_transformers.append({
                "step": "pca",
                "n_components": n_components,
                "transformer": transformer,
            })

        # Feature Selection
        elif step_name in ["feature_selection", "select_k_best"]:
            k = int(params.get("k", 5))
            X_train, X_test, transformer = apply_feature_selection(
                X_train, X_test, y_train, k=k
            )
            fitted_transformers.append({
                "step": "feature_selection",
                "k": k,
                "transformer": transformer,
            })

        # Variance Threshold (Noise Removal)
        elif step_name == "variance_threshold":
            thresh = float(params.get("threshold", 0.0))
            X_train, X_test, transformer = apply_variance_threshold(
                X_train, X_test, threshold=thresh
            )
            fitted_transformers.append({
                "step": "variance_threshold",
                "transformer": transformer,
            })

    # Convert column labels to string and reset indices
    X_train.columns = [str(c) for c in X_train.columns]
    X_test.columns = [str(c) for c in X_test.columns]

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        fitted_transformers,
    )
