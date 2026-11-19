# ML Engine Architecture & Edge Deployment Specification

**Author:** Hiral Gupta (`Hiral25-og`)  
**Component:** ML Engine Core, Pipeline Execution, Dimension Reduction & Edge Deployment  
**Role:** Person B  

---

## 1. Executive Summary

The **ML Engine** serves as the deterministic, reproducible machine learning execution core of the Interactive Model Studio. It encapsulates four core responsibilities:
1. **Dynamic Preprocessing & Transformation**: Modular feature engineering, power transforms, binning, and outlier clipping.
2. **Dimension Reduction & Feature Selection**: Orthogonal decomposition via Principal Component Analysis (PCA) and univariate statistical feature selection.
3. **Data Splitting & Leakage-Free Pipeline Execution**: Strict fit/transform lifecycle isolation ensuring zero test set contamination.
4. **Edge Deployment & ONNX Export**: Compiling trained pipelines to open neural network exchange (ONNX) and lightweight Edge JSON for sub-millisecond offline in-browser inference.

---

## 2. Preprocessing & Feature Engineering Architecture

### 2.1 Transformation Pipeline
Every dataset passes through a strictly ordered multi-stage transformation sequence configured via the `RecipeRunner`:

```
Input DataFrame
      │
      ▼
Stage 1: Outlier Handling & Value Clipping (Z-Score / IQR bounds)
      │
      ▼
Stage 2: Power Transforms & Discretization (Log1p, Yeo-Johnson, KBins)
      │
      ▼
Stage 3: Imputation & Scaling (Mean/Median/Mode Imputer, StandardScaler, MinMaxScaler, RobustScaler)
      │
      ▼
Stage 4: Categorical Encoding (OneHotEncoder, OrdinalEncoder)
      │
      ▼
Stage 5: Dimension Reduction (PCA / SelectKBest / VarianceThreshold)
      │
      ▼
Transformed Feature Matrix (X_train, X_val, X_test)
```

### 2.2 Dimension Reduction with PCA
High-dimensional datasets frequently suffer from multicollinearity and curse of dimensionality. The ML Engine integrates scikit-learn's `PCA`:
- **Variance Retention**: Allows selecting top $k$ principal components or explaining up to $95\%$ of cumulative variance.
- **Transformation Pipeline**: PCA is fitted **strictly on the training split** and applied identically to validation and test splits:
  $$\mathbf{Z} = (\mathbf{X} - \boldsymbol{\mu})\mathbf{W}$$
  where $\mathbf{W}$ comprises the top $k$ eigenvectors of the covariance matrix.

---

## 3. Zero-Leakage Pipeline Execution

A critical flaw in standard data science workflows is *data leakage*—computing statistics (e.g., standard deviation, mean, PCA components, encoding frequencies) across the full dataset prior to train/test splitting.

The `ml_engine` enforces zero leakage:
1. **Split First**: The dataset is partitioned into Train / Validation / Test sets using stratified or random splitting (`ml_engine/preprocessing.py:apply_train_val_test_split`).
2. **Fit on Train Only**: Transformers (imputers, scalers, encoders, PCA) compute statistics **only** on the training partition:
   ```python
   transformer.fit(X_train)
   X_train_trans = transformer.transform(X_train)
   X_test_trans = transformer.transform(X_test)
   ```
3. **Reproducibility**: The `RecipeRunner` hashes all hyperparameter configurations and random seeds (`random_state=42`) ensuring bit-for-bit reproducibility of experiment runs.

---

## 4. Edge Deployment & ONNX Export Pipeline

To enable zero-latency client-side scoring without relying on remote cloud servers, the studio provides dual export formats:

### 4.1 ONNX (Open Neural Network Exchange)
- Trained scikit-learn pipelines are converted into `.onnx` graph binaries using `skl2onnx`.
- Standardized tensor inputs (`[None, num_features]`) allow seamless deployment across edge environments, C++ runtimes, and mobile devices.

### 4.2 Edge JSON Model Format
For instant in-browser offline inference without bulky WebAssembly binaries, linear models and tree ensembles can be exported into an optimized Edge JSON schema:
```json
{
  "format": "interactive-studio-edge-v1",
  "model_type": "LogisticRegression",
  "feature_names": ["age", "income", "credit_score", "pca_0", "pca_1"],
  "coefficients": [0.034, -0.0012, -0.045, 0.12, -0.08],
  "intercept": 0.452,
  "classes": [0, 1]
}
```
Inference is executed offline directly inside the browser using dot-product vectorized evaluation:
$$z = \sum_{j=1}^p w_j x_j + b, \quad P(y=1) = \frac{1}{1 + e^{-z}}$$

Measured inference latency: **< 1 millisecond** with zero network round trips.

---

## 5. Verification & Unit Testing

The ML Engine includes automated test suites covering:
- `test_training.py`: Model fitting, metric evaluation (Accuracy, F1, ROC-AUC, RMSE, MAE).
- `test_reproduce.py`: Cache hits and deterministic metric verification across identical runs.
- `export_onnx.py`: Serializing pipelines to ONNX and Edge JSON formats and validating parity between Python `predict_proba` and edge scoring.
