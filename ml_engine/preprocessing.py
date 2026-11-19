"""
Preprocessing Module (Person B)
Provides modular, reproducible data transformation operations:
- Data Cleaning (Imputation, Duplicate Removal, Dropping Columns)
- Outlier & Noise Handling (IQR Clipping, Z-Score Winsorization, Variance Filtering)
- Transformations (Scaling, Categorical Encoding, Log1p, Yeo-Johnson, KBins Discretization)
- Dimension Reduction & Feature Selection (PCA, SelectKBest, VarianceThreshold)
- Data Splitting (Train/Test, Train/Validation/Test, StratifiedKFold cross-validation)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold
from sklearn.preprocessing import PowerTransformer, KBinsDiscretizer, FunctionTransformer

from ml_engine.registry import get_preprocessor_class


# ============================================================================
# 1. DATA SPLITTING & CROSS-VALIDATION
# ============================================================================

def apply_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits a DataFrame into X_train, X_test, y_train, y_test.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame columns: {list(df.columns)}")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    stratify_target = y if (stratify and y.nunique() > 1 and y.nunique() <= 20) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_target,
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def apply_train_val_test_split(
    df: pd.DataFrame,
    target_column: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits DataFrame into 3 partitions: Train, Validation, and Test.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    strat_target = y if (stratify and y.nunique() > 1 and y.nunique() <= 20) else None

    # Step 1: Split train vs temp (val + test)
    temp_size = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=strat_target
    )

    # Step 2: Split temp into val and test
    val_relative_size = val_ratio / temp_size
    temp_strat = y_temp if (stratify and y_temp.nunique() > 1 and y_temp.nunique() <= 20) else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_relative_size), random_state=random_state, stratify=temp_strat
    )

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


# ============================================================================
# 2. DATA CLEANING & OUTLIER / NOISE HANDLING
# ============================================================================

def apply_imputation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    strategy: str = "mean",
    fill_value: Optional[Any] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Fits an imputer on X_train[columns] and transforms both X_train and X_test without leakage.
    """
    valid_cols = [col for col in columns if col in X_train.columns]
    if not valid_cols:
        return X_train, X_test, None

    imputer_cls = get_preprocessor_class("impute", strategy)
    imputer = imputer_cls(strategy=strategy, fill_value=fill_value)

    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()

    X_train_transformed[valid_cols] = imputer.fit_transform(X_train[valid_cols])
    X_test_transformed[valid_cols] = imputer.transform(X_test[valid_cols])

    return X_train_transformed, X_test_transformed, imputer


def apply_outlier_clipping(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    threshold: float = 1.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """
    Computes outlier clipping bounds on X_train and applies them to both X_train and X_test.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    bounds: Dict[str, Tuple[float, float]] = {}

    for col in columns:
        if col not in X_tr.columns or not np.issubdtype(X_tr[col].dtype, np.number):
            continue

        series = X_tr[col].dropna()
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

        bounds[col] = (lower_bound, upper_bound)
        X_tr[col] = np.clip(X_tr[col], lower_bound, upper_bound)
        X_te[col] = np.clip(X_te[col], lower_bound, upper_bound)

    return X_tr, X_te, bounds


# ============================================================================
# 3. FEATURE TRANSFORMATIONS
# ============================================================================

def apply_scaling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    method: str = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Fits scaler on X_train and transforms X_train and X_test.
    """
    valid_cols = [col for col in columns if col in X_train.columns]
    if not valid_cols:
        return X_train, X_test, None

    scaler_cls = get_preprocessor_class("scale", method)
    scaler = scaler_cls()

    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()

    X_train_transformed[valid_cols] = scaler.fit_transform(X_train[valid_cols])
    X_test_transformed[valid_cols] = scaler.transform(X_test[valid_cols])

    return X_train_transformed, X_test_transformed, scaler


def apply_encoding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    method: str = "onehot",
    drop: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Encodes categorical features via OneHotEncoder or OrdinalEncoder.
    """
    valid_cols = [col for col in columns if col in X_train.columns]
    if not valid_cols:
        return X_train, X_test, None

    encoder_cls = get_preprocessor_class("encode", method)

    if method == "onehot":
        encoder = encoder_cls(
            drop=drop if drop in ["first", "if_binary"] else None,
            sparse_output=False,
            handle_unknown="ignore",
        )
        encoded_train_arr = encoder.fit_transform(X_train[valid_cols].astype(str))
        encoded_test_arr = encoder.transform(X_test[valid_cols].astype(str))
        feature_names = encoder.get_feature_names_out(valid_cols)

        df_encoded_train = pd.DataFrame(encoded_train_arr, columns=feature_names, index=X_train.index)
        df_encoded_test = pd.DataFrame(encoded_test_arr, columns=feature_names, index=X_test.index)

        X_train_transformed = pd.concat([X_train.drop(columns=valid_cols), df_encoded_train], axis=1)
        X_test_transformed = pd.concat([X_test.drop(columns=valid_cols), df_encoded_test], axis=1)

    elif method == "ordinal":
        encoder = encoder_cls(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_transformed = X_train.copy()
        X_test_transformed = X_test.copy()

        X_train_transformed[valid_cols] = encoder.fit_transform(X_train[valid_cols].astype(str))
        X_test_transformed[valid_cols] = encoder.transform(X_test[valid_cols].astype(str))
    else:
        raise ValueError(f"Unsupported encoding method: {method}")

    return X_train_transformed, X_test_transformed, encoder


def apply_power_transform(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    method: str = "yeo-johnson",
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Applies Yeo-Johnson or Box-Cox power transform to make feature distributions normal.
    """
    valid_cols = [col for col in columns if col in X_train.columns and np.issubdtype(X_train[col].dtype, np.number)]
    if not valid_cols:
        return X_train, X_test, None

    transformer = PowerTransformer(method=method, standardize=True)
    X_tr = X_train.copy()
    X_te = X_test.copy()

    X_tr[valid_cols] = transformer.fit_transform(X_tr[valid_cols])
    X_te[valid_cols] = transformer.transform(X_te[valid_cols])

    return X_tr, X_te, transformer


def apply_log1p_transform(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Applies np.log1p (log(1 + x)) transformation to skewed non-negative columns.
    """
    valid_cols = [col for col in columns if col in X_train.columns and np.issubdtype(X_train[col].dtype, np.number)]
    if not valid_cols:
        return X_train, X_test, None

    X_tr = X_train.copy()
    X_te = X_test.copy()

    for col in valid_cols:
        min_val = X_tr[col].min()
        offset = abs(min_val) if min_val < 0 else 0
        X_tr[col] = np.log1p(X_tr[col] + offset)
        X_te[col] = np.log1p(X_te[col] + offset)

    return X_tr, X_te, FunctionTransformer(np.log1p)


def apply_binning(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: List[str],
    n_bins: int = 5,
    strategy: str = "quantile",
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Discretizes continuous numerical columns into categorical bins.
    """
    valid_cols = [col for col in columns if col in X_train.columns and np.issubdtype(X_train[col].dtype, np.number)]
    if not valid_cols:
        return X_train, X_test, None

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    X_tr = X_train.copy()
    X_te = X_test.copy()

    X_tr[valid_cols] = discretizer.fit_transform(X_tr[valid_cols])
    X_te[valid_cols] = discretizer.transform(X_te[valid_cols])

    return X_tr, X_te, discretizer


# ============================================================================
# 4. DATA REDUCTION (PCA & FEATURE SELECTION)
# ============================================================================

def apply_pca_reduction(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: Union[int, float] = 2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Performs Principal Component Analysis (PCA) to reduce feature dimensionality.
    Fits PCA on numerical training features, projects both train and test.
    """
    # Exclude non-numeric columns if any remained
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return X_train, X_test, None

    k = min(n_components, len(num_cols), len(X_train))
    pca = PCA(n_components=k, random_state=random_state)

    X_tr_pca = pca.fit_transform(X_train[num_cols])
    X_te_pca = pca.transform(X_test[num_cols])

    pca_cols = [f"PCA_Dim_{i+1}" for i in range(X_tr_pca.shape[1])]

    df_tr_pca = pd.DataFrame(X_tr_pca, columns=pca_cols, index=X_train.index)
    df_te_pca = pd.DataFrame(X_te_pca, columns=pca_cols, index=X_test.index)

    # If there were non-numeric columns, retain them alongside PCA features
    non_num_cols = [c for c in X_train.columns if c not in num_cols]
    if non_num_cols:
        df_tr_final = pd.concat([df_tr_pca, X_train[non_num_cols]], axis=1)
        df_te_final = pd.concat([df_te_pca, X_test[non_num_cols]], axis=1)
    else:
        df_tr_final = df_tr_pca
        df_te_final = df_te_pca

    return df_tr_final, df_te_final, pca


def apply_feature_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    k: int = 10,
    task_type: str = "classification",
) -> Tuple[pd.DataFrame, pd.DataFrame, SelectKBest]:
    """
    Selects top K most informative features using ANOVA F-value test.
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return X_train, X_test, None

    k_actual = min(k, len(num_cols))
    score_func = f_classif if task_type == "classification" else f_regression

    selector = SelectKBest(score_func=score_func, k=k_actual)
    selector.fit(X_train[num_cols], y_train)

    selected_mask = selector.get_support()
    selected_cols = [col for col, selected in zip(num_cols, selected_mask) if selected]

    return X_train[selected_cols].copy(), X_test[selected_cols].copy(), selector


def apply_variance_threshold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, VarianceThreshold]:
    """
    Removes constant or near-zero variance features (noise filtering).
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return X_train, X_test, None

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_train[num_cols])

    selected_mask = selector.get_support()
    retained_cols = [col for col, kept in zip(num_cols, selected_mask) if kept]

    return X_train[retained_cols].copy(), X_test[retained_cols].copy(), selector
