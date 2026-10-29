/**
 * Python Code Generator Service
 * Converts visual pipeline configuration into clean, standalone, executable scikit-learn Python scripts.
 */

export function generatePythonScript({
  datasetName = "customer_churn.csv",
  targetColumn = "churn",
  testSize = 0.2,
  randomState = 42,
  cleaningConfigs = {}, // { [col]: { action: 'keep'|'drop', imputer: string, handleOutliers: boolean } }
  transformConfigs = {}, // { [col]: { mathTransform: string, scaler: string, encoder: string, binning: string } }
  modelType = "RandomForestClassifier",
  hyperparameters = { n_estimators: 100, max_depth: 8, random_state: 42 }
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
    } else if (cfg.mathTransform === "yeo_johnson") {
      powerTransformCols.push(col);
    }

    if (cfg.type === "categorical" || cfg.encoder) {
      categoricalCols.push(col);
    } else {
      numericCols.push(col);
    }
  });

  // Extract common scalers and encoders
  const activeScaler = Object.values(transformConfigs).find(c => c.scaler && c.scaler !== "none")?.scaler || "StandardScaler";
  const activeEncoder = Object.values(transformConfigs).find(c => c.encoder && c.encoder !== "none")?.encoder || "OneHotEncoder";

  const hyperparamsStr = Object.entries(hyperparameters)
    .map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`)
    .join(", ");

  let script = `"""
Interactive ML Model Studio — Generated Reproducible Pipeline
Dataset: ${datasetName} | Target: ${targetColumn} | Model: ${modelType}
Generated on: ${new Date().toISOString()}
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer, FunctionTransformer
`;

  if (modelType.includes("RandomForest")) {
    script += `from sklearn.ensemble import RandomForestClassifier\n`;
  } else if (modelType.includes("XGBoost") || modelType.includes("XGB")) {
    script += `from xgboost import XGBClassifier\n`;
  } else if (modelType.includes("Logistic")) {
    script += `from sklearn.linear_model import LogisticRegression\n`;
  } else {
    script += `from sklearn.tree import DecisionTreeClassifier\n`;
  }

  script += `from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("[1/5] Loading dataset...")
df = pd.read_csv("${datasetName}")
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

# ==========================================
# 2. STAGE 1: DATA CLEANING CHAIN
# ==========================================
print("[2/5] Applying Data Cleaning Chain...")
`;

  if (droppedCols.length > 0) {
    script += `# Drop irrelevant, high-null, or proxy ID features\n`;
    script += `drop_cols = ${JSON.stringify(droppedCols)}\n`;
    script += `df = df.drop(columns=[col for col in drop_cols if col in df.columns])\n`;
  } else {
    script += `# No columns marked for dropping\n`;
  }

  if (outlierCols.length > 0) {
    script += `\n# Outlier Handling: IQR 1.5x Clipping on skewed features\n`;
    script += `outlier_features = ${JSON.stringify(outlierCols)}\n`;
    script += `for col in outlier_features:\n`;
    script += `    if col in df.columns and np.issubdtype(df[col].dtype, np.number):\n`;
    script += `        Q1 = df[col].quantile(0.25)\n`;
    script += `        Q3 = df[col].quantile(0.75)\n`;
    script += `        IQR = Q3 - Q1\n`;
    script += `        lower_bound = Q1 - 1.5 * IQR\n`;
    script += `        upper_bound = Q3 + 1.5 * IQR\n`;
    script += `        df[col] = np.clip(df[col], lower_bound, upper_bound)\n`;
  }

  script += `\n# Separate features (X) and target (y)\n`;
  script += `target_col = "${targetColumn}"\n`;
  script += `X = df.drop(columns=[target_col])\n`;
  script += `y = df[target_col]\n`;
  script += `\n# Train/Test Split\n`;
  script += `X_train, X_test, y_train, y_test = train_test_split(\n`;
  script += `    X, y, test_size=${testSize}, random_state=${randomState}, stratify=y if y.nunique() <= 10 else None\n`;
  script += `)\n`;
  script += `print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")\n`;

  script += `\n# ==========================================
# 3. STAGE 2: FEATURE TRANSFORMATION CHAIN
# ==========================================
print("[3/5] Building Feature Transformation Transformers...")

numeric_features = ${JSON.stringify(numericCols)}
categorical_features = ${JSON.stringify(categoricalCols)}

# Numeric Preprocessing Pipeline (Impute -> Math Transform -> Scaling)
numeric_transformer = Pipeline(steps=[\n`;
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

  script += `# Categorical Preprocessing Pipeline (Impute -> Encoding)
categorical_transformer = Pipeline(steps=[\n`;
  script += `    ('imputer', SimpleImputer(strategy='most_frequent')),\n`;
  if (activeEncoder === "OrdinalEncoder") {
    script += `    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))\n`;
  } else {
    script += `    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n`;
  }
  script += `])\n\n`;

  script += `# Composite Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, [c for c in numeric_features if c in X.columns]),
        ('cat', categorical_transformer, [c for c in categorical_features if c in X.columns])
    ],
    remainder='passthrough'
)\n`;

  script += `\n# ==========================================
# 4. MODEL ESTIMATOR PIPELINE
# ==========================================
print("[4/5] Instantiating ${modelType}...")
model = ${modelType}(${hyperparamsStr})

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# ==========================================
# 5. EXECUTION & EVALUATION
# ==========================================
print("[5/5] Fitting full pipeline and evaluating...")
full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\\n Model Performance:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\\nPipeline execution complete! Model ready for deployment.")
`;

  return script;
}
