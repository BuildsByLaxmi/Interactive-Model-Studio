"""
Evaluation Module (Person B)
Calculates performance metrics, confusion matrices, and feature importances
for classification and regression tasks.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def extract_feature_importances(
    fitted_model: Any,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Extracts and ranks feature importances or coefficients from a fitted model.
    """
    importances = []

    # Tree-based models (Random Forest, Decision Tree, Gradient Boosting, XGBoost)
    if hasattr(fitted_model, "feature_importances_"):
        raw_importances = fitted_model.feature_importances_
        for name, val in zip(feature_names, raw_importances):
            importances.append({"feature": name, "importance": round(float(val), 5)})

    # Linear models (Logistic Regression, Linear Regression)
    elif hasattr(fitted_model, "coef_"):
        raw_coefs = fitted_model.coef_
        if raw_coefs.ndim > 1:
            mean_coefs = np.mean(np.abs(raw_coefs), axis=0)
        else:
            mean_coefs = np.abs(raw_coefs)

        total = np.sum(mean_coefs)
        norm_coefs = mean_coefs / (total if total > 0 else 1.0)
        for name, val in zip(feature_names, norm_coefs):
            importances.append({"feature": name, "importance": round(float(val), 5)})

    # Sort descending by importance
    importances.sort(key=lambda x: x["importance"], reverse=True)
    return importances


def evaluate_model(
    fitted_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: str = "classification",
) -> Dict[str, Any]:
    """
    Evaluates model performance and returns metrics, confusion matrix, and feature importances.
    """
    y_pred = fitted_model.predict(X_test)
    metrics: Dict[str, float] = {}
    cm_payload: Optional[Dict[str, Any]] = None

    if task_type == "classification":
        metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
        metrics["balanced_accuracy"] = round(float(balanced_accuracy_score(y_test, y_pred)), 4)

        # Average precision / recall / f1
        average = "binary" if y_test.nunique() == 2 else "weighted"
        metrics["precision"] = round(float(precision_score(y_test, y_pred, average=average, zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_test, y_pred, average=average, zero_division=0)), 4)
        metrics["f1_score"] = round(float(f1_score(y_test, y_pred, average=average, zero_division=0)), 4)

        # ROC-AUC if probabilities are supported
        if hasattr(fitted_model, "predict_proba"):
            try:
                y_probs = fitted_model.predict_proba(X_test)
                if y_test.nunique() == 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_probs[:, 1])), 4)
                else:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_probs, multi_class="ovr")), 4)
            except Exception:
                pass

        # Confusion Matrix
        labels = sorted(list(set(y_test.unique()) | set(pd.Series(y_pred).unique())))
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        if len(labels) == 2:
            tn, fp, fn, tp = cm.ravel()
            cm_payload = {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "labels": [str(l) for l in labels],
                "matrix": cm.tolist(),
            }
        else:
            cm_payload = {
                "labels": [str(l) for l in labels],
                "matrix": cm.tolist(),
            }

    else:
        # Regression metrics
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_test, y_pred)), 4)
        metrics["r2_score"] = round(float(r2_score(y_test, y_pred)), 4)

    # Feature Importances
    feature_names = list(X_test.columns)
    importances = extract_feature_importances(fitted_model, feature_names)

    return {
        "metrics": metrics,
        "confusion_matrix": cm_payload,
        "feature_importances": importances,
    }
