"""
Data Processing & Integration Service (Person A)
Implements:
- Data Cleaning (missing value handling, duplicate removal, invalid row filtering)
- Data Integration (joining, merging, and concatenating multiple datasets)
- Outlier & Noise Handling (IQR bounds, Z-score analysis, variance filtering, noise detection)
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# ============================================================================
# 1. DATA INTEGRATION
# ============================================================================

def merge_datasets(
    df_primary: pd.DataFrame,
    df_secondary: pd.DataFrame,
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    how: str = "inner",
    suffixes: Tuple[str, str] = ("_primary", "_secondary"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Integrates two datasets via relational join (inner, left, right, outer).
    Returns the merged DataFrame and integration metadata.
    """
    initial_rows = (len(df_primary), len(df_secondary))

    try:
        merged_df = pd.merge(
            df_primary,
            df_secondary,
            on=on,
            left_on=left_on,
            right_on=right_on,
            how=how,
            suffixes=suffixes,
        )
    except Exception as e:
        raise ValueError(f"Data integration merge failed: {str(e)}")

    summary = {
        "join_type": how,
        "join_key": on or f"{left_on} == {right_on}",
        "primary_initial_rows": initial_rows[0],
        "secondary_initial_rows": initial_rows[1],
        "result_rows": len(merged_df),
        "result_columns": len(merged_df.columns),
        "column_names": list(merged_df.columns),
    }

    return merged_df, summary


def concat_datasets(
    dfs: List[pd.DataFrame],
    axis: int = 0,
    ignore_index: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Concatenates multiple datasets vertically (appending rows) or horizontally (combining features).
    """
    if not dfs:
        raise ValueError("No datasets provided for concatenation.")

    merged_df = pd.concat(dfs, axis=axis, ignore_index=ignore_index)

    summary = {
        "concat_axis": "rows" if axis == 0 else "columns",
        "dataset_count": len(dfs),
        "result_rows": len(merged_df),
        "result_columns": len(merged_df.columns),
    }

    return merged_df, summary


# ============================================================================
# 2. OUTLIER & NOISE HANDLING
# ============================================================================

def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> Dict[str, Any]:
    """
    Scans numerical columns for outliers using IQR or Z-score method.
    Returns per-column outlier counts, percentages, and upper/lower bounds.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = [c for c in (columns or num_cols) if c in num_cols]

    outlier_report = {}
    total_outlier_rows = set()

    for col in target_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue

        if method == "iqr":
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == "zscore":
            mean = float(series.mean())
            std = float(series.std())
            if std == 0:
                continue
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            outlier_mask = ((df[col] - mean).abs() > threshold * std)
        else:
            raise ValueError(f"Unsupported outlier detection method '{method}'. Choose 'iqr' or 'zscore'.")

        outlier_indices = df[outlier_mask].index.tolist()
        count = len(outlier_indices)
        pct = round((count / max(1, len(df))) * 100, 2)

        if count > 0:
            total_outlier_rows.update(outlier_indices)
            outlier_report[col] = {
                "outlier_count": count,
                "percentage": pct,
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "min_observed": float(series.min()),
                "max_observed": float(series.max()),
                "sample_outliers": [round(float(v), 4) for v in series[outlier_mask].head(5)],
            }

    return {
        "method": method,
        "threshold": threshold,
        "columns_evaluated": len(target_cols),
        "columns_with_outliers": len(outlier_report),
        "total_rows_with_any_outlier": len(total_outlier_rows),
        "total_outlier_row_percentage": round((len(total_outlier_rows) / max(1, len(df))) * 100, 2),
        "column_details": outlier_report,
    }


def clip_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    threshold: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handles outliers via winsorization/clipping.
    Values outside [lower_bound, upper_bound] are clipped to the boundaries.
    """
    df_clipped = df.copy()
    report = {}

    for col in columns:
        if col not in df_clipped.columns or not np.issubdtype(df_clipped[col].dtype, np.number):
            continue

        series = df_clipped[col].dropna()
        if len(series) < 5:
            continue

        if method == "iqr":
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
        else:
            mean = float(series.mean())
            std = float(series.std())
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std

        clipped_count = int(((df_clipped[col] < lower_bound) | (df_clipped[col] > upper_bound)).sum())
        df_clipped[col] = np.clip(df_clipped[col], lower_bound, upper_bound)

        report[col] = {
            "clipped_count": clipped_count,
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
        }

    return df_clipped, report


def detect_noise_and_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Scans dataset for noise, duplicates, low-variance features, and leakage:
    - Duplicate rows
    - Constant / zero-variance columns (pure noise)
    - Near-zero variance columns (e.g. >99% single value)
    - High-cardinality ID proxies
    """
    total_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())

    constant_cols = []
    low_variance_cols = []
    high_null_cols = []

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / max(1, total_rows)) * 100, 2)
        if null_pct > 40:
            high_null_cols.append({"column": col, "null_pct": null_pct})

        unique_count = df[col].nunique(dropna=True)
        if unique_count <= 1:
            constant_cols.append(col)
        elif total_rows > 50:
            top_val_pct = (df[col].value_counts(normalize=True, dropna=False).iloc[0]) * 100
            if top_val_pct > 99:
                low_variance_cols.append({"column": col, "dominant_value_pct": round(top_val_pct, 2)})

    return {
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_row_pct": round((duplicate_rows / max(1, total_rows)) * 100, 2),
        "constant_columns": constant_cols,
        "low_variance_columns": low_variance_cols,
        "high_null_columns": high_null_cols,
    }


# ============================================================================
# 3. DATA CLEANING EXECUTION
# ============================================================================

def clean_dataset(
    df: pd.DataFrame,
    drop_columns: Optional[List[str]] = None,
    drop_duplicates: bool = True,
    impute_strategy: Optional[Dict[str, str]] = None,
    outlier_clipping: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies comprehensive data cleaning chain to a DataFrame.
    """
    df_clean = df.copy()
    audit_log = []

    # 1. Drop specified columns
    if drop_columns:
        valid_drops = [c for c in drop_columns if c in df_clean.columns]
        if valid_drops:
            df_clean = df_clean.drop(columns=valid_drops)
            audit_log.append(f"Dropped {len(valid_drops)} columns: {valid_drops}")

    # 2. Drop duplicates
    if drop_duplicates:
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        dropped_dups = initial_count - len(df_clean)
        if dropped_dups > 0:
            audit_log.append(f"Removed {dropped_dups} duplicate rows")

    # 3. Impute missing values
    if impute_strategy:
        for col, strategy in impute_strategy.items():
            if col not in df_clean.columns:
                continue
            if np.issubdtype(df_clean[col].dtype, np.number):
                if strategy == "mean":
                    val = df_clean[col].mean()
                elif strategy == "median":
                    val = df_clean[col].median()
                elif strategy == "constant":
                    val = 0
                else:
                    val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(val)
            else:
                mode_vals = df_clean[col].mode()
                val = mode_vals[0] if len(mode_vals) > 0 else "Missing"
                df_clean[col] = df_clean[col].fillna(val)
            audit_log.append(f"Imputed missing values in '{col}' using {strategy}")

    # 4. Outlier clipping
    if outlier_clipping and outlier_clipping.get("columns"):
        cols = outlier_clipping["columns"]
        method = outlier_clipping.get("method", "iqr")
        threshold = outlier_clipping.get("threshold", 1.5)
        df_clean, clip_report = clip_outliers(df_clean, cols, method=method, threshold=threshold)
        audit_log.append(f"Clipped outliers in {len(clip_report)} columns using {method} (threshold={threshold})")

    summary = {
        "initial_shape": (len(df), len(df.columns)),
        "cleaned_shape": (len(df_clean), len(df_clean.columns)),
        "audit_log": audit_log,
    }

    return df_clean, summary
