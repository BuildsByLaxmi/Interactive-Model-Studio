import { useState, useEffect } from "react";
import { useSearchParams, useNavigate, Link } from "react-router-dom";
import { 
  Sliders, 
  Code2, 
  Copy, 
  Check, 
  ArrowRight, 
  ArrowLeft, 
  Wand2, 
  RotateCcw,
  Brush,
  Zap,
  Layers,
  TrendingDown,
  Split,
  ShieldAlert,
  Percent,
  Play
} from "lucide-react";
import api from "../api/client";
import { getMockDatasetById, mockDatasets } from "../api/mockData";

export default function Preprocessing() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const datasetId = searchParams.get("datasetId") || mockDatasets[0].id;
  const initialTarget = searchParams.get("target");

  const [dataset, setDataset] = useState(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState("cleaning"); // "cleaning" | "transform" | "reduction" | "splitting"

  // Stage 1: Data Cleaning & Noise / Outlier configs
  const [cleaningConfigs, setCleaningConfigs] = useState({});
  const [dropDuplicates, setDropDuplicates] = useState(true);

  // Stage 2: Feature Transformation configs
  const [transformConfigs, setTransformConfigs] = useState({});

  // Stage 3: Dimension Reduction configs
  const [reductionConfig, setReductionConfig] = useState({
    enabled: true,
    method: "pca", // "pca" | "select_k_best" | "variance_threshold"
    n_components: 2,
    k_best: 5,
    variance_threshold: 0.01,
  });

  // Stage 4: Data Splitting & Cross-Validation configs
  const [splitConfig, setSplitConfig] = useState({
    splitMode: "train_test", // "train_test" | "train_val_test" | "kfold"
    trainRatio: 80,
    valRatio: 10,
    testRatio: 20,
    stratify: true,
    n_splits: 5,
    randomState: 42,
  });

  useEffect(() => {
    api.getDatasets()
      .then((data) => {
        const found = Array.isArray(data) ? data.find((d) => d.id === datasetId) : null;
        setupDataset(found || getMockDatasetById(datasetId));
      })
      .catch(() => {
        setupDataset(getMockDatasetById(datasetId));
      });
  }, [datasetId]);

  const setupDataset = (data) => {
    setDataset(data);
    const target = initialTarget || data.target_column || data.schema[data.schema.length - 1]?.name;
    setTargetColumn(target);

    const initialCleaning = {};
    const initialTransform = {};

    (data.schema || []).forEach((col) => {
      if (col.name === target) return;

      const isNumeric = col.type === "numerical";
      const isIdOrProxy = col.name.toLowerCase().includes("id") || col.name.toLowerCase().includes("proxy");

      // Stage 1: Cleaning defaults
      initialCleaning[col.name] = {
        action: isIdOrProxy ? "drop" : "keep",
        imputer: isNumeric ? (col.null_pct > 0 ? "median" : "none") : (col.null_pct > 0 ? "most_frequent" : "none"),
        handleOutliers: isNumeric && (col.name.toLowerCase().includes("charge") || col.name.toLowerCase().includes("price") || col.name.toLowerCase().includes("tenure")),
        outlierMethod: "iqr", // "iqr" | "zscore"
      };

      // Stage 2: Transformation defaults
      initialTransform[col.name] = {
        type: col.type,
        mathTransform: isNumeric && (col.name.toLowerCase().includes("charge") || col.name.toLowerCase().includes("price")) ? "log1p" : "none",
        scaler: isNumeric ? "StandardScaler" : "none",
        encoder: !isNumeric ? "OneHotEncoder" : "none",
        binning: "none",
      };
    });

    setCleaningConfigs(initialCleaning);
    setTransformConfigs(initialTransform);
  };

  const updateCleaning = (colName, key, value) => {
    setCleaningConfigs((prev) => ({
      ...prev,
      [colName]: { ...prev[colName], [key]: value },
    }));
  };

  const updateTransform = (colName, key, value) => {
    setTransformConfigs((prev) => ({
      ...prev,
      [colName]: { ...prev[colName], [key]: value },
    }));
  };

  const applyPreset = (presetName) => {
    if (!dataset) return;
    const newCleaning = {};
    const newTransform = {};

    dataset.schema.forEach((col) => {
      if (col.name === targetColumn) return;
      const isNumeric = col.type === "numerical";
      const isId = col.name.toLowerCase().includes("id");

      if (presetName === "recommended") {
        newCleaning[col.name] = {
          action: isId ? "drop" : "keep",
          imputer: isNumeric ? "median" : "most_frequent",
          handleOutliers: isNumeric,
          outlierMethod: "iqr",
        };
        newTransform[col.name] = {
          type: col.type,
          mathTransform: isNumeric && (col.name.toLowerCase().includes("charge") || col.name.toLowerCase().includes("price")) ? "log1p" : "none",
          scaler: isNumeric ? "StandardScaler" : "none",
          encoder: !isNumeric ? (col.cardinality > 10 ? "OrdinalEncoder" : "OneHotEncoder") : "none",
          binning: "none",
        };
      } else {
        newCleaning[col.name] = { action: "keep", imputer: "none", handleOutliers: false, outlierMethod: "iqr" };
        newTransform[col.name] = { type: col.type, mathTransform: "none", scaler: "none", encoder: "none", binning: "none" };
      }
    });

    setCleaningConfigs(newCleaning);
    setTransformConfigs(newTransform);
    setReductionConfig({ enabled: true, method: "pca", n_components: 2, k_best: 5, variance_threshold: 0.01 });
  };

  // Generate full 4-stage chained recipe JSON
  const generateRecipeJson = () => {
    const droppedCols = [];
    const imputeSteps = {};
    const outlierCols = [];
    const mathSteps = {};
    const scaleSteps = {};
    const encodeSteps = {};
    const binSteps = {};

    // 1. Cleaning steps
    Object.entries(cleaningConfigs).forEach(([col, cfg]) => {
      if (cfg.action === "drop") {
        droppedCols.push(col);
        return;
      }
      if (cfg.imputer && cfg.imputer !== "none") {
        if (!imputeSteps[cfg.imputer]) imputeSteps[cfg.imputer] = [];
        imputeSteps[cfg.imputer].push(col);
      }
      if (cfg.handleOutliers) {
        outlierCols.push(col);
      }
    });

    // 2. Transformation steps
    Object.entries(transformConfigs).forEach(([col, cfg]) => {
      if (cleaningConfigs[col]?.action === "drop") return;

      if (cfg.mathTransform && cfg.mathTransform !== "none") {
        if (!mathSteps[cfg.mathTransform]) mathSteps[cfg.mathTransform] = [];
        mathSteps[cfg.mathTransform].push(col);
      }
      if (cfg.scaler && cfg.scaler !== "none") {
        if (!scaleSteps[cfg.scaler]) scaleSteps[cfg.scaler] = [];
        scaleSteps[cfg.scaler].push(col);
      }
      if (cfg.encoder && cfg.encoder !== "none") {
        if (!encodeSteps[cfg.encoder]) encodeSteps[cfg.encoder] = [];
        encodeSteps[cfg.encoder].push(col);
      }
      if (cfg.binning && cfg.binning !== "none") {
        if (!binSteps[cfg.binning]) binSteps[cfg.binning] = [];
        binSteps[cfg.binning].push(col);
      }
    });

    const stage_1_cleaning = [];
    if (dropDuplicates) stage_1_cleaning.push({ step: "drop_duplicates" });
    if (droppedCols.length > 0) stage_1_cleaning.push({ step: "drop_columns", columns: droppedCols });
    Object.entries(imputeSteps).forEach(([strategy, cols]) => stage_1_cleaning.push({ step: "impute", strategy, columns: cols }));
    if (outlierCols.length > 0) stage_1_cleaning.push({ step: "outlier_clip", columns: outlierCols, method: "iqr", threshold: 1.5 });

    const stage_2_transformation = [];
    Object.entries(mathSteps).forEach(([method, cols]) => stage_2_transformation.push({ step: "math_transform", method, columns: cols }));
    Object.entries(encodeSteps).forEach(([encoder, cols]) => stage_2_transformation.push({ step: "encode", method: encoder, columns: cols }));
    Object.entries(scaleSteps).forEach(([scaler, cols]) => stage_2_transformation.push({ step: "scale", method: scaler, columns: cols }));
    Object.entries(binSteps).forEach(([bins, cols]) => stage_2_transformation.push({ step: "binning", bins, columns: cols }));

    const stage_3_reduction = [];
    if (reductionConfig.enabled) {
      if (reductionConfig.method === "pca") {
        stage_3_reduction.push({ step: "pca", n_components: Number(reductionConfig.n_components) });
      } else if (reductionConfig.method === "select_k_best") {
        stage_3_reduction.push({ step: "select_k_best", k: Number(reductionConfig.k_best) });
      } else {
        stage_3_reduction.push({ step: "variance_threshold", threshold: Number(reductionConfig.variance_threshold) });
      }
    }

    const stage_4_splitting = {
      mode: splitConfig.splitMode,
      train_ratio: splitConfig.trainRatio / 100,
      test_ratio: splitConfig.testRatio / 100,
      val_ratio: splitConfig.splitMode === "train_val_test" ? splitConfig.valRatio / 100 : 0.0,
      stratify: splitConfig.stratify,
      kfold: splitConfig.splitMode === "kfold",
      n_splits: splitConfig.n_splits,
      random_state: splitConfig.randomState,
    };

    return {
      dataset_id: dataset?.id || datasetId,
      target_column: targetColumn,
      chains: {
        stage_1_cleaning,
        stage_2_transformation,
        stage_3_reduction,
        stage_4_splitting,
      }
    };
  };

  const recipe = generateRecipeJson();
  const recipeJsonString = JSON.stringify(recipe, null, 2);

  const copyRecipe = () => {
    navigator.clipboard.writeText(recipeJsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!dataset) {
    return (
      <div className="page-container flex justify-center items-center py-20">
        <div className="text-center text-muted">Loading dataset preprocessing setup...</div>
      </div>
    );
  }

  const eligibleCols = dataset.schema.filter((col) => col.name !== targetColumn);

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header flex justify-between items-start flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Link to={`/dataset/${dataset.id}`} className="text-muted hover:text-primary flex items-center gap-1 text-sm font-medium">
              <ArrowLeft size={15} /> Back to Schema
            </Link>
          </div>
          <h1 className="page-title">End-to-End Data Preparation & Reduction Studio</h1>
          <p className="page-subtitle">
            Configure the 4-stage pipeline: <strong>Data Cleaning & Outliers</strong> ➔ <strong>Transformations</strong> ➔ <strong>Dimension Reduction (PCA)</strong> ➔ <strong>Splitting & Cross-Validation</strong>.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button 
            className="btn btn-secondary flex items-center gap-1.5"
            onClick={() => applyPreset("recommended")}
          >
            <Wand2 size={15} className="text-primary" />
            <span>Auto-Recommend Pipeline</span>
          </button>
          <button 
            className="btn btn-secondary flex items-center gap-1.5"
            onClick={() => applyPreset("reset")}
          >
            <RotateCcw size={15} />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Target Column Selector Bar */}
      <div className="studio-card mb-6 bg-slate-50">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <label className="text-xs font-bold uppercase tracking-wider text-muted">Target Variable (y):</label>
            <select 
              value={targetColumn} 
              onChange={(e) => setTargetColumn(e.target.value)}
              className="form-select font-bold text-primary"
              style={{ width: "auto", minWidth: "200px" }}
            >
              {dataset.schema.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name} ({col.type})
                </option>
              ))}
            </select>
          </div>
          <div className="text-xs text-muted">
            Predictive Model Goal: Classification of <strong>{targetColumn}</strong>
          </div>
        </div>
      </div>

      {/* 4-Stage Navigation Tabs */}
      <div className="chain-tabs-nav mb-4 flex flex-wrap gap-2">
        <button 
          className={`chain-tab-btn ${activeTab === "cleaning" ? "active" : ""}`}
          onClick={() => setActiveTab("cleaning")}
        >
          <div className="chain-step-badge">1</div>
          <Brush size={16} />
          <span>Stage 1: Cleaning & Outliers</span>
          <span className="tab-pill-count">{recipe.chains.stage_1_cleaning.length}</span>
        </button>

        <div className="chain-tab-divider">➔</div>

        <button 
          className={`chain-tab-btn ${activeTab === "transform" ? "active" : ""}`}
          onClick={() => setActiveTab("transform")}
        >
          <div className="chain-step-badge">2</div>
          <Zap size={16} />
          <span>Stage 2: Transformations</span>
          <span className="tab-pill-count">{recipe.chains.stage_2_transformation.length}</span>
        </button>

        <div className="chain-tab-divider">➔</div>

        <button 
          className={`chain-tab-btn ${activeTab === "reduction" ? "active" : ""}`}
          onClick={() => setActiveTab("reduction")}
        >
          <div className="chain-step-badge" style={{ background: activeTab === "reduction" ? "#e11d48" : "#cbd5e1" }}>3</div>
          <TrendingDown size={16} />
          <span>Stage 3: Dimension Reduction</span>
          <span className="tab-pill-count">{reductionConfig.enabled ? "Active" : "Off"}</span>
        </button>

        <div className="chain-tab-divider">➔</div>

        <button 
          className={`chain-tab-btn ${activeTab === "splitting" ? "active" : ""}`}
          onClick={() => setActiveTab("splitting")}
        >
          <div className="chain-step-badge" style={{ background: activeTab === "splitting" ? "#0d9488" : "#cbd5e1" }}>4</div>
          <Split size={16} />
          <span>Stage 4: Data Splitting</span>
          <span className="tab-pill-count">{splitConfig.splitMode}</span>
        </button>
      </div>

      {/* Main Grid: Active Stage Configuration + Live JSON Preview */}
      <div className="recipe-grid">
        {/* Stage 1: Data Cleaning & Noise / Outlier Handling */}
        {activeTab === "cleaning" && (
          <div className="studio-card">
            <div className="flex justify-between items-center mb-3">
              <div>
                <h2 className="card-heading flex items-center gap-2">
                  <Brush size={18} className="text-primary" />
                  Stage 1: Data Cleaning, Noise & Outlier Handling
                </h2>
                <p className="text-xs text-muted">
                  Remove noisy/proxy columns, impute null values, eliminate duplicates, and clip extreme outliers.
                </p>
              </div>
              <label className="flex items-center gap-2 text-xs font-semibold cursor-pointer">
                <input 
                  type="checkbox"
                  checked={dropDuplicates}
                  onChange={(e) => setDropDuplicates(e.target.checked)}
                  className="accent-indigo-600 rounded"
                />
                <span>Drop Duplicate Rows</span>
              </label>
            </div>

            <div className="table-responsive">
              <table className="studio-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Type</th>
                    <th>Action</th>
                    <th>Missing Imputer</th>
                    <th>Outlier Handling (Winsorize)</th>
                  </tr>
                </thead>
                <tbody>
                  {eligibleCols.map((col) => {
                    const cfg = cleaningConfigs[col.name] || { action: "keep", imputer: "none", handleOutliers: false, outlierMethod: "iqr" };
                    const isNumeric = col.type === "numerical";
                    const isDropped = cfg.action === "drop";

                    return (
                      <tr key={col.name} className={isDropped ? "opacity-50" : ""}>
                        <td>
                          <strong>{col.name}</strong>
                          {col.null_pct > 0 && (
                            <span className="text-xs block text-muted">
                              {col.null_pct.toFixed(0)}% nulls
                            </span>
                          )}
                        </td>
                        <td>
                          <span className={`type-tag type-${col.type}`}>{col.type}</span>
                        </td>
                        <td>
                          <button
                            className={`btn-action-drop ${isDropped ? "dropped" : ""}`}
                            onClick={() => updateCleaning(col.name, "action", isDropped ? "keep" : "drop")}
                            title={isDropped ? "Restore feature" : "Drop feature"}
                          >
                            {isDropped ? "Dropped" : "Keep"}
                          </button>
                        </td>
                        <td>
                          {isDropped ? (
                            <span className="text-muted text-xs">—</span>
                          ) : isNumeric ? (
                            <select
                              value={cfg.imputer}
                              onChange={(e) => updateCleaning(col.name, "imputer", e.target.value)}
                              className="form-select-sm"
                            >
                              <option value="none">None (No Nulls)</option>
                              <option value="median">Median Impute</option>
                              <option value="mean">Mean Impute</option>
                              <option value="constant">Constant (0)</option>
                            </select>
                          ) : (
                            <select
                              value={cfg.imputer}
                              onChange={(e) => updateCleaning(col.name, "imputer", e.target.value)}
                              className="form-select-sm"
                            >
                              <option value="none">None (No Nulls)</option>
                              <option value="most_frequent">Most Frequent (Mode)</option>
                              <option value="constant">Constant ("Missing")</option>
                            </select>
                          )}
                        </td>
                        <td>
                          {isDropped || !isNumeric ? (
                            <span className="text-muted text-xs">N/A</span>
                          ) : (
                            <label className="flex items-center gap-2 cursor-pointer text-xs font-semibold">
                              <input 
                                type="checkbox"
                                checked={cfg.handleOutliers || false}
                                onChange={(e) => updateCleaning(col.name, "handleOutliers", e.target.checked)}
                                className="accent-indigo-600 rounded"
                              />
                              <span>1.5× IQR Clip</span>
                            </label>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Stage 2: Feature Transformation Chain */}
        {activeTab === "transform" && (
          <div className="studio-card">
            <div className="flex justify-between items-center mb-3">
              <div>
                <h2 className="card-heading flex items-center gap-2">
                  <Zap size={18} className="text-accent" />
                  Stage 2: Feature Transformation Chain
                </h2>
                <p className="text-xs text-muted">
                  Apply mathematical power transforms, categorical encoders, and feature scalers.
                </p>
              </div>
            </div>

            <div className="table-responsive">
              <table className="studio-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Type</th>
                    <th>Math / Power Transform</th>
                    <th>Scaling Method</th>
                    <th>Categorical Encoding</th>
                  </tr>
                </thead>
                <tbody>
                  {eligibleCols.map((col) => {
                    const isDropped = cleaningConfigs[col.name]?.action === "drop";
                    const cfg = transformConfigs[col.name] || { mathTransform: "none", scaler: "none", encoder: "none" };
                    const isNumeric = col.type === "numerical";

                    return (
                      <tr key={col.name} className={isDropped ? "opacity-50" : ""}>
                        <td>
                          <strong>{col.name}</strong>
                          {isDropped && <span className="text-xs block text-danger">(Dropped in Stage 1)</span>}
                        </td>
                        <td>
                          <span className={`type-tag type-${col.type}`}>{col.type}</span>
                        </td>
                        <td>
                          {isDropped || !isNumeric ? (
                            <span className="text-muted text-xs">—</span>
                          ) : (
                            <select
                              value={cfg.mathTransform}
                              onChange={(e) => updateTransform(col.name, "mathTransform", e.target.value)}
                              className="form-select-sm"
                            >
                              <option value="none">None (Raw)</option>
                              <option value="log1p">Log1p (np.log1p)</option>
                              <option value="yeo_johnson">Yeo-Johnson Power</option>
                              <option value="sqrt">Square Root</option>
                            </select>
                          )}
                        </td>
                        <td>
                          {isDropped || !isNumeric ? (
                            <span className="text-muted text-xs">—</span>
                          ) : (
                            <select
                              value={cfg.scaler}
                              onChange={(e) => updateTransform(col.name, "scaler", e.target.value)}
                              className="form-select-sm"
                            >
                              <option value="none">None (Unscaled)</option>
                              <option value="StandardScaler">StandardScaler (z-score)</option>
                              <option value="MinMaxScaler">MinMaxScaler [0, 1]</option>
                              <option value="RobustScaler">RobustScaler (IQR)</option>
                            </select>
                          )}
                        </td>
                        <td>
                          {isDropped || isNumeric ? (
                            <span className="text-muted text-xs">—</span>
                          ) : (
                            <select
                              value={cfg.encoder}
                              onChange={(e) => updateTransform(col.name, "encoder", e.target.value)}
                              className="form-select-sm"
                            >
                              <option value="none">None</option>
                              <option value="OneHotEncoder">OneHotEncoder</option>
                              <option value="OrdinalEncoder">OrdinalEncoder</option>
                            </select>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Stage 3: Dimension Reduction & Feature Selection */}
        {activeTab === "reduction" && (
          <div className="studio-card">
            <div className="flex justify-between items-center mb-3">
              <div>
                <h2 className="card-heading flex items-center gap-2 text-rose-600">
                  <TrendingDown size={18} />
                  Stage 3: Data Reduction & Dimension Reduction
                </h2>
                <p className="text-xs text-muted">
                  Compress high-dimensional feature spaces, remove multicollinear noise, or select top informative features.
                </p>
              </div>
              <label className="flex items-center gap-2 text-xs font-semibold cursor-pointer">
                <input 
                  type="checkbox"
                  checked={reductionConfig.enabled}
                  onChange={(e) => setReductionConfig({ ...reductionConfig, enabled: e.target.checked })}
                  className="accent-rose-600 rounded"
                />
                <span>Enable Dimension Reduction</span>
              </label>
            </div>

            {reductionConfig.enabled ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                  <label className="form-label font-bold text-sm">Reduction Technique</label>
                  <select
                    value={reductionConfig.method}
                    onChange={(e) => setReductionConfig({ ...reductionConfig, method: e.target.value })}
                    className="form-select mt-1"
                  >
                    <option value="pca">PCA (Principal Component Analysis - Unsupervised)</option>
                    <option value="select_k_best">SelectKBest (ANOVA F-Score - Supervised)</option>
                    <option value="variance_threshold">Variance Threshold (Zero/Low Variance Noise Filter)</option>
                  </select>
                  <p className="text-xs text-muted mt-2">
                    {reductionConfig.method === "pca" && "Projects features into orthogonal components maximizing explained variance."}
                    {reductionConfig.method === "select_k_best" && "Computes ANOVA F-statistic between each feature and the target variable."}
                    {reductionConfig.method === "variance_threshold" && "Removes all features whose variance does not meet the specified threshold."}
                  </p>
                </div>

                <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                  {reductionConfig.method === "pca" && (
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <label className="form-label font-bold text-sm">Number of Principal Components</label>
                        <span className="text-xs font-bold text-rose-600">{reductionConfig.n_components} components</span>
                      </div>
                      <input 
                        type="range"
                        min="1"
                        max={Math.min(eligibleCols.length, 10)}
                        step="1"
                        value={reductionConfig.n_components}
                        onChange={(e) => setReductionConfig({ ...reductionConfig, n_components: parseInt(e.target.value) })}
                        className="w-full accent-rose-600 mt-2"
                      />
                      <span className="text-xs text-muted mt-1 block">Reduces {eligibleCols.length} raw features down to {reductionConfig.n_components} orthogonal dimensions.</span>
                    </div>
                  )}

                  {reductionConfig.method === "select_k_best" && (
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <label className="form-label font-bold text-sm">Top K Best Features</label>
                        <span className="text-xs font-bold text-rose-600">Top {reductionConfig.k_best}</span>
                      </div>
                      <input 
                        type="range"
                        min="1"
                        max={eligibleCols.length}
                        step="1"
                        value={reductionConfig.k_best}
                        onChange={(e) => setReductionConfig({ ...reductionConfig, k_best: parseInt(e.target.value) })}
                        className="w-full accent-rose-600 mt-2"
                      />
                    </div>
                  )}

                  {reductionConfig.method === "variance_threshold" && (
                    <div>
                      <label className="form-label font-bold text-sm">Minimum Variance Threshold</label>
                      <input 
                        type="number"
                        step="0.01"
                        min="0"
                        value={reductionConfig.variance_threshold}
                        onChange={(e) => setReductionConfig({ ...reductionConfig, variance_threshold: parseFloat(e.target.value) || 0 })}
                        className="form-input mt-1"
                      />
                      <span className="text-xs text-muted mt-1 block">Drops all features with variance ≤ {reductionConfig.variance_threshold}.</span>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-muted border border-dashed rounded-lg mt-4">
                Dimension reduction is currently disabled. All cleaned & transformed features will be passed directly to the model estimator.
              </div>
            )}
          </div>
        )}

        {/* Stage 4: Data Splitting & Cross-Validation */}
        {activeTab === "splitting" && (
          <div className="studio-card">
            <div className="flex justify-between items-center mb-3">
              <div>
                <h2 className="card-heading flex items-center gap-2 text-teal-600">
                  <Split size={18} />
                  Stage 4: Data Splitting & Validation Strategy
                </h2>
                <p className="text-xs text-muted">
                  Configure training, validation, and testing partitions with stratification to handle imbalanced datasets.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                <label className="form-label font-bold text-sm">Partitioning Strategy</label>
                <select
                  value={splitConfig.splitMode}
                  onChange={(e) => setSplitConfig({ ...splitConfig, splitMode: e.target.value })}
                  className="form-select mt-1"
                >
                  <option value="train_test">Train / Test Split (e.g. 80% / 20%)</option>
                  <option value="train_val_test">Train / Validation / Test Split (e.g. 70% / 15% / 15%)</option>
                  <option value="kfold">K-Fold Cross-Validation (5-Fold / 10-Fold)</option>
                </select>

                <div className="mt-4">
                  <label className="flex items-center gap-2 text-xs font-semibold cursor-pointer">
                    <input 
                      type="checkbox"
                      checked={splitConfig.stratify}
                      onChange={(e) => setSplitConfig({ ...splitConfig, stratify: e.target.checked })}
                      className="accent-teal-600 rounded"
                    />
                    <span>Stratified Split (Preserve class balance in all splits)</span>
                  </label>
                </div>
              </div>

              <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                {splitConfig.splitMode === "train_test" && (
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="form-label font-bold text-sm">Test Split Size</label>
                      <span className="text-xs font-bold text-teal-600">{splitConfig.testRatio}% Test / {100 - splitConfig.testRatio}% Train</span>
                    </div>
                    <input 
                      type="range"
                      min="10"
                      max="40"
                      step="5"
                      value={splitConfig.testRatio}
                      onChange={(e) => setSplitConfig({ ...splitConfig, testRatio: parseInt(e.target.value), trainRatio: 100 - parseInt(e.target.value) })}
                      className="w-full accent-teal-600 mt-2"
                    />
                  </div>
                )}

                {splitConfig.splitMode === "train_val_test" && (
                  <div>
                    <label className="form-label font-bold text-sm mb-2 block">Train / Val / Test Ratios</label>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 p-2 bg-white rounded border text-center">
                        <span className="text-xs text-muted block">Train</span>
                        <strong className="text-sm text-teal-700">70%</strong>
                      </div>
                      <div className="flex-1 p-2 bg-white rounded border text-center">
                        <span className="text-xs text-muted block">Validation</span>
                        <strong className="text-sm text-teal-700">15%</strong>
                      </div>
                      <div className="flex-1 p-2 bg-white rounded border text-center">
                        <span className="text-xs text-muted block">Test</span>
                        <strong className="text-sm text-teal-700">15%</strong>
                      </div>
                    </div>
                  </div>
                )}

                {splitConfig.splitMode === "kfold" && (
                  <div>
                    <label className="form-label font-bold text-sm">Number of Folds (K)</label>
                    <select
                      value={splitConfig.n_splits}
                      onChange={(e) => setSplitConfig({ ...splitConfig, n_splits: parseInt(e.target.value) })}
                      className="form-select mt-1"
                    >
                      <option value="5">5-Fold Cross-Validation</option>
                      <option value="10">10-Fold Cross-Validation</option>
                    </select>
                  </div>
                )}

                <div className="mt-4">
                  <label className="form-label font-bold text-xs text-muted">Reproducibility Seed</label>
                  <input 
                    type="number"
                    value={splitConfig.randomState}
                    onChange={(e) => setSplitConfig({ ...splitConfig, randomState: parseInt(e.target.value) || 42 })}
                    className="form-input text-xs"
                    style={{ width: "100px" }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Real-Time Chained Recipe JSON Panel */}
        <div className="studio-card recipe-preview-card">
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center gap-2">
              <Code2 size={18} className="text-primary" />
              <h3 className="font-bold text-sm">4-Stage Pipeline Recipe JSON</h3>
            </div>
            <button 
              className="btn btn-secondary btn-sm flex items-center gap-1"
              onClick={copyRecipe}
            >
              {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
              <span>{copied ? "Copied!" : "Copy JSON"}</span>
            </button>
          </div>
          <p className="text-xs text-muted mb-3">
            Standardized execution specification passed to <code>recipe_runner.py</code> and Python codegen.
          </p>

          <pre className="json-box" style={{ maxHeight: "380px" }}>
            <code>{recipeJsonString}</code>
          </pre>

          <div className="mt-4 flex flex-col gap-2">
            <div className="flex justify-between items-center gap-2">
              <Link 
                to={`/pipeline?datasetId=${dataset?.id || datasetId}`}
                className="btn btn-secondary btn-sm flex-1 flex items-center justify-center gap-1"
              >
                <Layers size={14} />
                <span>Visual Studio</span>
              </Link>
              <Link 
                to={`/models?datasetId=${dataset?.id || datasetId}`}
                className="btn btn-primary btn-sm flex-1 flex items-center justify-center gap-1"
              >
                <span>Proceed to Models</span>
                <ArrowRight size={14} />
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
