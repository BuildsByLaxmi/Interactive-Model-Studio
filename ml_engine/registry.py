"""
ML Engine Registry (Person B)
Single source of truth mapping step names, preprocessing methods,
and algorithm names to scikit-learn and XGBoost classes.
Used by both the execution engine and the code generator.
"""

from typing import Any, Dict, List, Optional, Type
import importlib

# Supported Preprocessor Mapping
def get_preprocessor_class(step_name: str, method_or_strategy: str = "default"):
    """
    Dynamically resolves a preprocessing transformer class.
    Deferred imports ensure fast startup and clean error handling.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        StandardScaler,
        MinMaxScaler,
        RobustScaler,
        OneHotEncoder,
        OrdinalEncoder,
        PowerTransformer,
        KBinsDiscretizer,
    )
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, VarianceThreshold

    mapping = {
        # Imputation
        ("impute", "mean"): SimpleImputer,
        ("impute", "median"): SimpleImputer,
        ("impute", "most_frequent"): SimpleImputer,
        ("impute", "constant"): SimpleImputer,
        # Scaling
        ("scale", "standard"): StandardScaler,
        ("scale", "minmax"): MinMaxScaler,
        ("scale", "robust"): RobustScaler,
        # Encoding
        ("encode", "onehot"): OneHotEncoder,
        ("encode", "ordinal"): OrdinalEncoder,
        # Transformations
        ("transform", "yeo_johnson"): PowerTransformer,
        ("transform", "box_cox"): PowerTransformer,
        ("transform", "binning"): KBinsDiscretizer,
        # Dimension Reduction & Feature Selection
        ("reduction", "pca"): PCA,
        ("reduction", "select_k_best"): SelectKBest,
        ("reduction", "variance_threshold"): VarianceThreshold,
    }

    key = (step_name.lower(), method_or_strategy.lower())
    if key not in mapping:
        raise ValueError(
            f"Unknown preprocessor: step='{step_name}', method/strategy='{method_or_strategy}'. "
            f"Supported combinations: {list(mapping.keys())}"
        )
    return mapping[key]


def get_transformer(step_type: str, method_name: str):
    """Alias for backwards compatibility with legacy callers."""
    return get_preprocessor_class(step_type, method_name)


# Model Registry & Metadata
MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "RandomForestClassifier": {
        "task_type": "classification",
        "library": "sklearn",
        "default_params": {"n_estimators": 100, "max_depth": 8, "random_state": 42},
        "description": "Bagging ensemble of decision trees with feature randomization."
    },
    "LogisticRegression": {
        "task_type": "classification",
        "library": "sklearn",
        "default_params": {"C": 1.0, "max_iter": 1000, "random_state": 42},
        "description": "Linear classifier using logistic loss."
    },
    "DecisionTreeClassifier": {
        "task_type": "classification",
        "library": "sklearn",
        "default_params": {"max_depth": 5, "random_state": 42},
        "description": "Single decision tree with entropy / gini criterion."
    },
    "GradientBoostingClassifier": {
        "task_type": "classification",
        "library": "sklearn",
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        "description": "Additive gradient boosting ensemble."
    },
    "XGBClassifier": {
        "task_type": "classification",
        "library": "xgboost",
        "default_params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        "description": "Extreme Gradient Boosting classifier."
    },
    "RandomForestRegressor": {
        "task_type": "regression",
        "library": "sklearn",
        "default_params": {"n_estimators": 100, "max_depth": None, "random_state": 42},
        "description": "Random forest regression ensemble."
    },
    "LinearRegression": {
        "task_type": "regression",
        "library": "sklearn",
        "default_params": {"fit_intercept": True},
        "description": "Ordinary least squares linear regression."
    },
}


def get_model_class(model_name: str, task_type: Optional[str] = None) -> Type[Any]:
    """
    Returns the estimator class for a given model name.
    Supports scikit-learn and XGBoost estimators.
    """
    model_map: Dict[str, str] = {
        "RandomForestClassifier": "sklearn.ensemble.RandomForestClassifier",
        "LogisticRegression": "sklearn.linear_model.LogisticRegression",
        "DecisionTreeClassifier": "sklearn.tree.DecisionTreeClassifier",
        "GradientBoostingClassifier": "sklearn.ensemble.GradientBoostingClassifier",
        "XGBClassifier": "xgboost.XGBClassifier",
        "XGBoostClassifier": "xgboost.XGBClassifier",
        "RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
        "LinearRegression": "sklearn.linear_model.LinearRegression",
    }

    if model_name not in model_map:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {list(model_map.keys())}"
        )

    module_path, class_name = model_map[model_name].rsplit(".", 1)

    if module_path == "xgboost":
        try:
            import xgboost as xgb
            return getattr(xgb, class_name)
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def list_supported_models(task_type: Optional[str] = None) -> List[str]:
    """Returns a list of model names, optionally filtered by task type."""
    if task_type is None:
        return list(MODEL_METADATA.keys())
    return [name for name, meta in MODEL_METADATA.items() if meta["task_type"] == task_type]


def get_model_defaults(model_name: str) -> Dict[str, Any]:
    """Returns the default hyperparameters for a given model."""
    if model_name not in MODEL_METADATA:
        return {"random_state": 42}
    return MODEL_METADATA[model_name]["default_params"].copy()
