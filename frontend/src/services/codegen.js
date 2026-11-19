/**
 * Comprehensive Python Code Generator Service (Person C)
 * Converts complete 4-stage visual pipeline configuration into clean,
 * standalone, executable scikit-learn & pandas Python scripts.
 */

export function generatePythonScript({
  datasetName = "customer_churn.csv",
  targetColumn = "churn",
  splitConfig = { train: 0.7, val: 0.15, test: 0.15, stratify: true, kfold: false, n_splits: 5, randomState: 42 },
  cleaningConfigs = {}, // { [col]: { action: 'keep'|'drop', imputer: string, outlierMethod: string, handleOutliers: boolean } }
  transformConfigs = {}, // { [col]: { mathTransform: string, scaler: string, encoder: string, binning: string } }
  reductionConfig = { enabled: true, method: "pca", n_components: 2, k_best: 10, variance_threshold: 0.0 },
  modelType = "RandomForestClassifier",
  hyperparameters = { n_estimators: 100, max_depth: 8, random_state: 42 },
  exportOnnx = true
}) {
  const droppedCols = Object.entries(cleaningConfigs)
    .filter(([_, cfg]) => cfg.action === "drop")
    .map(([col]) => col);

  const outlierCols = Object.entries(cleaningConfigs)
    .filter(([_, cfg]) => cfg.handleOutliers && cfg.action !== "drop")
    .map(([col]) => col);

  // Categorize remaining features
  const numericCols = [];
  const categoricalCols = [];
  const logTransformCols = [];
  const powerTransformCols = [];

  Object.entries(transformConfigs).forEach(([col, cfg]) => {
    if (cleaningConfigs[col]?.action === "drop" || col === targetColumn) return;

    if (cfg.mathTransform === "log1p") {
      logTransformCols.push(col);
    } else if (cfg.mathTransform === "yeo_johnson" || cfg.mathTransform === "box_cox") {
      powerTransformCols.push(col);
    }

    if (cfg.type === "categorical" || cfg.encoder) {
      categoricalCols.push(col);
    } else {
      numericCols.push(col);
    }
  });

  const activeScaler = Object.values(transformConfigs).find(c => c.scaler && c.scaler !== "none")?.scaler || "StandardScaler";
  const activeEncoder = Object.values(transformConfigs).find(c => c.encoder && c.encoder !== "none")?.encoder || "OneHotEncoder";

  const hyperparamsStr = Object.entries(hyperparameters)
    .map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`)
    .join(", ");

  let script = `"""
Interactive ML Model Studio — End-to-End Reproducible Pipeline
Dataset: ${datasetName} | Target: ${targetColumn} | Estimator: ${modelType}
Features: Data Cleaning -> Outlier Clipping -> Transformations -> Dimension Reduction (${reductionConfig.method.toUpperCase()}) -> Splitting -> Estimator
Generated on: ${new Date().toISOString()}
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler, 
    OneHotEncoder, 
    OrdinalEncoder, 
    PowerTransformer, 
    FunctionTransformer,
    KBinsDiscretizer
)
`;

  if (reductionConfig.enabled) {
    if (reductionConfig.method === "pca") {
      script += `from sklearn.decomposition import PCA\n`;
    } else if (reductionConfig.method === "select_k_best") {
      script += `from sklearn.feature_selection import SelectKBest, f_classif\n`;
    } else if (reductionConfig.method === "variance_threshold") {
      script += `from sklearn.feature_selection import VarianceThreshold\n`;
    }
  }

  if (modelType.includes("RandomForest")) {
    script += `from sklearn.ensemble import RandomForestClassifier\n`;
  } else if (modelType.includes("XGBoost") || modelType.includes("XGB")) {
    script += `from xgboost import XGBClassifier\n`;
  } else if (modelType.includes("Logistic")) {
    script += `from sklearn.linear_model import LogisticRegression\n`;
  } else {
    script += `from sklearn.tree import DecisionTreeClassifier\n`;
  }

  script += `from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ==============================================================================
# 1. DATA INGESTION & QUALITY INSPECTION
# ==============================================================================
print("[1/6] Ingesting dataset: ${datasetName}...")
df = pd.read_csv("${datasetName}")
print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

# ==============================================================================
# 2. STAGE 1: DATA CLEANING & OUTLIER / NOISE HANDLING
# ==============================================================================
print("[2/6] Executing Stage 1: Data Cleaning & Noise/Outlier Handling...")

# Drop duplicates if present
initial_len = len(df)
df = df.drop_duplicates()
if len(df) < initial_len:
    print(f"Removed {initial_len - len(df)} duplicate records.")
`;

  if (droppedCols.length > 0) {
    script += `# Drop irrelevant ID, noisy, or proxy leakage columns\n`;
    script += `drop_cols = ${JSON.stringify(droppedCols)}\n`;
    script += `df = df.drop(columns=[col for col in drop_cols if col in df.columns])\n`;
  }

  if (outlierCols.length > 0) {
    script += `\n# Outlier Handling: IQR Winsorization on skewed numerical features\n`;
    script += `outlier_features = ${JSON.stringify(outlierCols)}\n`;
    script += `for col in outlier_features:\n`;
    script += `    if col in df.columns and np.issubdtype(df[col].dtype, np.number):\n`;
    script += `        Q1 = df[col].quantile(0.25)\n`;
    script += `        Q3 = df[col].quantile(0.75)\n`;
    script += `        IQR = Q3 - Q1\n`;
    script += `        lower_bound = Q1 - 1.5 * IQR\n`;
    script += `        upper_bound = Q3 + 1.5 * IQR\n`;
    script += `        df[col] = np.clip(df[col], lower_bound, upper_bound)\n`;
    script += `        print(f"Clipped outliers for '{col}' to [{lower_bound:.2f}, {upper_bound:.2f}]")\n`;
  }

  script += `\n# ==============================================================================
# 3. STAGE 2: DATA SPLITTING & STRATIFICATION
# ==============================================================================
print("[3/6] Partitioning data into train and test splits...")
target_col = "${targetColumn}"
X = df.drop(columns=[target_col])
y = df[target_col]

test_ratio = ${splitConfig.test || 0.2}
random_state = ${splitConfig.randomState || 42}
stratify_param = y if (${splitConfig.stratify ? "True" : "False"} and y.nunique() <= 20) else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio, random_state=random_state, stratify=stratify_param
)
print(f"Training split: {X_train.shape[0]} samples | Testing split: {X_test.shape[0]} samples")

# ==============================================================================
# 4. STAGE 3: FEATURE TRANSFORMATION PIPELINE
# ==============================================================================
print("[4/6] Building Preprocessing Transformers...")

numeric_features = [c for c in ${JSON.stringify(numericCols)} if c in X_train.columns]
categorical_features = [c for c in ${JSON.stringify(categoricalCols)} if c in X_train.columns]

numeric_pipeline = Pipeline(steps=[\n`;
  script += `    ('imputer', SimpleImputer(strategy='median')),\n`;

  if (logTransformCols.length > 0) {
    script += `    ('log_transform', FunctionTransformer(np.log1p, validate=True)),\n`;
  } else if (powerTransformCols.length > 0) {
    script += `    ('yeo_johnson', PowerTransformer(method='yeo-johnson')),\n`;
  }

  if (activeScaler === "MinMaxScaler") {
    script += `    ('scaler', MinMaxScaler())\n`;
  } else if (activeScaler === "RobustScaler") {
    script += `    ('scaler', RobustScaler())\n`;
  } else {
    script += `    ('scaler', StandardScaler())\n`;
  }
  script += `])\n\n`;

  script += `categorical_pipeline = Pipeline(steps=[\n`;
  script += `    ('imputer', SimpleImputer(strategy='most_frequent')),\n`;
  if (activeEncoder === "OrdinalEncoder") {
    script += `    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))\n`;
  } else {
    script += `    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n`;
  }
  script += `])\n\n`;

  script += `preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='passthrough'
)\n`;

  script += `\n# ==============================================================================
# 5. STAGE 4: DATA & DIMENSION REDUCTION
# ==============================================================================
`;

  const pipelineSteps = [`('preprocessor', preprocessor)`];

  if (reductionConfig.enabled) {
    if (reductionConfig.method === "pca") {
      script += `print("[5/6] Applying PCA Dimension Reduction (n_components=${reductionConfig.n_components})...")\n`;
      script += `pca_reducer = PCA(n_components=${reductionConfig.n_components}, random_state=${splitConfig.randomState || 42})\n`;
      pipelineSteps.push(`('pca_reduction', pca_reducer)`);
    } else if (reductionConfig.method === "select_k_best") {
      script += `print("[5/6] Applying SelectKBest Feature Selection (k=${reductionConfig.k_best})...")\n`;
      script += `k_best_selector = SelectKBest(score_func=f_classif, k=${reductionConfig.k_best})\n`;
      pipelineSteps.push(`('feature_selection', k_best_selector)`);
    } else if (reductionConfig.method === "variance_threshold") {
      script += `print("[5/6] Applying VarianceThreshold noise filtering...")\n`;
      script += `var_selector = VarianceThreshold(threshold=${reductionConfig.variance_threshold})\n`;
      pipelineSteps.push(`('variance_filter', var_selector)`);
    }
  } else {
    script += `print("[5/6] Full feature set retained (Dimension reduction bypassed)...")\n`;
  }

  script += `\n# Model Estimator
model = ${modelType}(${hyperparamsStr})
`;
  pipelineSteps.push(`('classifier', model)`);

  script += `\nfull_pipeline = Pipeline(steps=[\n    ${pipelineSteps.join(",\n    ")}\n])

# ==============================================================================
# 6. EXECUTION, CROSS-VALIDATION & EVALUATION
# ==============================================================================
print("[6/6] Fitting full pipeline and scoring...")
full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted' if y_test.nunique() > 2 else 'binary')

print(f"\\n Model Evaluation Results:")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"F1-Score:  {f1:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
`;

  if (exportOnnx) {
    script += `\n# ==============================================================================
# 7. EDGE DEPLOYMENT EXPORT (ONNX / Edge JSON)
# ==============================================================================
print("\\nExporting model for edge deployment...")
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]])))]
    # Export classifier step
    onnx_model = convert_sklearn(full_pipeline.named_steps['classifier'], initial_types=initial_type)
    with open("model_edge.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Exported: model_edge.onnx (Ready for in-browser inference via onnxruntime-web)")
except Exception as e:
    print(f"ONNX export notice: {e}")
    print("Pipeline ready for standard Python / FastAPI production deployment.")
`;
  }

  return script;
}
