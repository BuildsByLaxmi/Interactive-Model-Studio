# Frontend Architecture & UI/UX Design Specification

**Author:** Laxmi Singla (`Laxmi_singla`)  
**Component:** Frontend Application, Visual Pipeline DAG, Comparison Dashboard & Codegen Engine  
**Role:** Person C  

---

## 1. Executive Summary

The **Frontend Architecture** provides an intuitive, high-performance web interface for the Interactive Model Studio. It bridges non-technical users and advanced machine learning practitioners by abstracting complex scikit-learn preprocessing and model architectures into interactive visual components while simultaneously providing an exportable, standalone Python pipeline.

Key responsibilities include:
1. **4-Stage Chained Preprocessing Workspace**: Modular control tabs for Cleaning & Outlier Clipping, Power Transformations & Scaling, Dimension Reduction (PCA), and Train/Val/Test Splitting.
2. **Interactive Visual Pipeline DAG**: Direct acyclic graph visualizing feature flows from Raw Ingestion to Edge Deployment.
3. **Experiment Comparison Dashboard**: Multi-run metric analysis, parameter diffing, and one-click experiment restoration.
4. **Offline Edge Model Simulator**: Sub-millisecond in-browser client scoring simulator for exported ONNX and Edge JSON models.
5. **Deterministic Codegen Engine**: Real-time generation of standalone, dependency-minimal scikit-learn Python scripts with inline PCA and ColumnTransformer integration.

---

## 2. Component Hierarchy & Navigation

```
App Shell (Navbar, Connection Monitor, Status Badge)
 │
 ├── /upload ────────────── Ingestion, Dataset Registry & CSV Diagnostics
 ├── /dataset/:id ───────── Schema Inspector, Null Ratios & Actionable Diagnostics Detail
 ├── /preprocessing ─────── 4-Stage Preprocessing Workspace (Cleaning, Transforms, PCA, Split)
 ├── /models ────────────── Estimator Configuration (RandomForest, LogisticRegression, GradientBoosting)
 ├── /experiments ───────── Historical Run Log & Status Tracker
 ├── /pipeline ──────────── Interactive DAG Canvas & Standalone Codegen Preview
 ├── /dashboard ─────────── Model Performance Radar, Confusion Matrix & Experiment Comparison
 └── /edge-export ───────── Offline In-Browser Edge Inference Sandbox
```

---

## 3. Core Frontend Subsystems

### 3.1 4-Stage Preprocessing Workspace (`Preprocessing.jsx`)
Organized into four dedicated tabs with persistent state synchronization:
1. **Cleaning & Outliers**: Imputation strategies (mean, median, most_frequent), outlier clipping thresholds (IQR 1.5x / Z-Score 3.0), and noise filtering.
2. **Transformations & Scaling**: Power transforms (`log1p`, `yeo-johnson`), binning (`quantile`, `uniform`), and feature scalers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`).
3. **Dimension Reduction (PCA)**: Toggleable Principal Component Analysis slider with live component counts ($k \in [2, 10]$), variance thresholding, and univariate feature selection (`SelectKBest`).
4. **Data Splitting**: Visual slider allocating Train / Validation / Test percentages (e.g. 70/15/15) with stratified sampling options to prevent class imbalance issues.

### 3.2 Visual Pipeline DAG (`PipelineCanvas.jsx`)
Visualizes the end-to-end data processing workflow through a responsive 6-node acyclic graph:
- **Node 1: Raw Ingest** (Source CSV & Schema Parser)
- **Node 2: Stage 1 - Clean** (Imputation & Outlier Clipping)
- **Node 3: Stage 2 - Transform** (Scaling & One-Hot Encoding)
- **Node 4: Stage 3 - Dimension Reduction** (PCA Orthogonal Decomposition)
- **Node 5: Stage 4 - Data Split** (Train / Validation / Test Separation)
- **Node 6: Estimator & Export** (Trained Model & Edge Artifacts)

### 3.3 Standalone Code Generator (`codegen.js` & `CodePreview.jsx`)
The codegen engine converts the UI state into production-grade Python code that users can run independently:
- Generates pure `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer`.
- Emits proper `PCA(n_components=k)` reduction steps.
- Enforces strict fit on training set and transform on test set to prevent leakage.
- Includes optional ONNX export routine (`skl2onnx`) for direct export.

---

## 4. Design System & Styling Architecture

- **Color Palette**: Enterprise slate (`#0f172a`, `#334155`), indigo primary (`#4f46e5`, `#6366f1`), emerald success (`#10b981`), and amber warning (`#f59e0b`).
- **CSS Architecture**: Scoped, modular CSS located in `src/App.css` ensuring cross-browser compatibility without build-time CSS compilation issues.
- **Responsiveness**: Flexbox and CSS Grid layout accommodating mobile, laptop, and ultra-wide monitor resolutions.
