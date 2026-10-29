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
  Sparkles,
  HelpCircle,
  Trash2,
  Brush,
  Zap,
  Layers,
  ShieldAlert,
  Percent
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
  const [testSize, setTestSize] = useState(0.2);
  const [randomState, setRandomState] = useState(42);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState("cleaning"); // "cleaning" | "transform"

  // Stage 1: Data Cleaning Configs per column: { action: 'keep'|'drop', imputer: string, handleOutliers: boolean }
  const [cleaningConfigs, setCleaningConfigs] = useState({});

  // Stage 2: Feature Transformation Configs per column: { mathTransform: string, scaler: string, encoder: string, binning: string }
  const [transformConfigs, setTransformConfigs] = useState({});

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
        handleOutliers: isNumeric && col.name.toLowerCase().includes("charge"),
      };

      // Stage 2: Transformation defaults
      initialTransform[col.name] = {
        type: col.type,
        mathTransform: isNumeric && col.name.toLowerCase().includes("charge") ? "log1p" : "none",
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
        };
        newTransform[col.name] = {
          type: col.type,
          mathTransform: isNumeric && (col.name.toLowerCase().includes("charge") || col.name.toLowerCase().includes("price")) ? "log1p" : "none",
          scaler: isNumeric ? "StandardScaler" : "none",
          encoder: !isNumeric ? (col.cardinality > 10 ? "OrdinalEncoder" : "OneHotEncoder") : "none",
          binning: "none",
        };
      } else {
        newCleaning[col.name] = { action: "keep", imputer: "none", handleOutliers: false };
        newTransform[col.name] = { type: col.type, mathTransform: "none", scaler: "none", encoder: "none", binning: "none" };
      }
    });

    setCleaningConfigs(newCleaning);
    setTransformConfigs(newTransform);
  };

  // Structured chained recipe JSON
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

    const cleaning_chain = [];
    if (droppedCols.length > 0) cleaning_chain.push({ step: "drop_columns", columns: droppedCols });
    Object.entries(imputeSteps).forEach(([strategy, cols]) => cleaning_chain.push({ step: "impute", strategy, columns: cols }));
    if (outlierCols.length > 0) cleaning_chain.push({ step: "iqr_outlier_clipping", columns: outlierCols, threshold: 1.5 });

    const transformation_chain = [];
    Object.entries(mathSteps).forEach(([method, cols]) => transformation_chain.push({ step: "math_transform", method, columns: cols }));
    Object.entries(encodeSteps).forEach(([encoder, cols]) => transformation_chain.push({ step: "encode", method: encoder, columns: cols }));
    Object.entries(scaleSteps).forEach(([scaler, cols]) => transformation_chain.push({ step: "scale", method: scaler, columns: cols }));
    Object.entries(binSteps).forEach(([bins, cols]) => transformation_chain.push({ step: "binning", bins, columns: cols }));

    return {
      dataset_id: dataset?.id || datasetId,
      target_column: targetColumn,
      split: { test_size: Number(testSize), random_state: Number(randomState) },
      chains: {
        stage_1_cleaning: cleaning_chain,
        stage_2_transformation: transformation_chain,
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
          <h1 className="page-title">Chained Data Cleaning & Feature Transformation</h1>
          <p className="page-subtitle">
            Configure data preparation in two sequential stages: <strong>Stage 1: Cleaning</strong> (impute nulls & handle outliers) followed by <strong>Stage 2: Transformation</strong> (power transforms, encoding & scaling).
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button 
            className="btn btn-secondary flex items-center gap-1.5"
            onClick={() => applyPreset("recommended")}
          >
            <Wand2 size={15} className="text-primary" />
            <span>Auto-Recommend Recipe</span>
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

      {/* Train / Test & Target Settings Bar */}
      <div className="studio-card mb-6 bg-slate-50">
        <h3 className="text-sm font-bold uppercase tracking-wider text-muted mb-4">
          Global Split & Target Setup
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="form-label">Target Column (y)</label>
            <select 
              value={targetColumn} 
              onChange={(e) => setTargetColumn(e.target.value)}
              className="form-select"
            >
              {dataset.schema.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name} ({col.type})
                </option>
              ))}
            </select>
          </div>

          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="form-label">Test Split Ratio</label>
              <span className="text-xs font-bold text-primary">{(testSize * 100).toFixed(0)}% Test / {((1 - testSize) * 100).toFixed(0)}% Train</span>
            </div>
            <input 
              type="range" 
              min="0.1" 
              max="0.4" 
              step="0.05"
              value={testSize} 
              onChange={(e) => setTestSize(parseFloat(e.target.value))}
              className="w-full accent-indigo-600"
            />
          </div>

          <div>
            <label className="form-label">Random Seed</label>
            <input 
              type="number" 
              value={randomState} 
              onChange={(e) => setRandomState(parseInt(e.target.value) || 42)}
              className="form-input"
            />
          </div>
        </div>
      </div>

      {/* Chained Stages Navigation Bar */}
      <div className="chain-tabs-nav mb-4">
        <button 
          className={`chain-tab-btn ${activeTab === "cleaning" ? "active" : ""}`}
          onClick={() => setActiveTab("cleaning")}
        >
          <div className="chain-step-badge">1</div>
          <Brush size={16} />
          <span>Stage 1: Data Cleaning Chain</span>
          <span className="tab-pill-count">
            {recipe.chains.stage_1_cleaning.length} steps
          </span>
        </button>

        <div className="chain-tab-divider">➔</div>

        <button 
          className={`chain-tab-btn ${activeTab === "transform" ? "active" : ""}`}
          onClick={() => setActiveTab("transform")}
        >
          <div className="chain-step-badge">2</div>
          <Zap size={16} />
          <span>Stage 2: Feature Transformation Chain</span>
          <span className="tab-pill-count">
            {recipe.chains.stage_2_transformation.length} steps
          </span>
        </button>
      </div>

      {/* Main Grid: Chained Controls Table + Real-Time JSON Preview */}
      <div className="recipe-grid">
        {/* Stage 1: Data Cleaning Chain View */}
        {activeTab === "cleaning" && (
          <div className="studio-card">
            <div className="flex justify-between items-center mb-3">
              <div>
                <h2 className="card-heading flex items-center gap-2">
                  <Brush size={18} className="text-primary" />
                  Stage 1: Data Cleaning Chain
                </h2>
                <p className="text-xs text-muted">
                  Remove proxy/ID columns, impute missing values, and handle extreme outliers.
                </p>
              </div>
            </div>

            <div className="table-responsive">
              <table className="studio-table">
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Type</th>
                    <th>Action</th>
                    <th>Missing Imputer</th>
                    <th>Outlier Handling (IQR)</th>
                  </tr>
                </thead>
                <tbody>
                  {eligibleCols.map((col) => {
                    const cfg = cleaningConfigs[col.name] || { action: "keep", imputer: "none", handleOutliers: false };
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

        {/* Stage 2: Feature Transformation Chain View */}
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

        {/* Real-Time Chained Recipe JSON Panel */}
        <div className="studio-card recipe-preview-card">
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center gap-2">
              <Code2 size={18} className="text-primary" />
              <h3 className="font-bold text-sm">Chained Recipe JSON</h3>
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
            Deterministic 2-stage configuration passed to <code>recipe_runner.py</code> and the Python Codegen engine.
          </p>

          <pre className="json-box">
            <code>{recipeJsonString}</code>
          </pre>

          <div className="mt-4 flex flex-col gap-2">
            <div className="flex justify-between text-xs text-muted mb-1">
              <span>Stage 1 (Cleaning): <strong>{recipe.chains.stage_1_cleaning.length}</strong> steps</span>
              <span>Stage 2 (Transforms): <strong>{recipe.chains.stage_2_transformation.length}</strong> steps</span>
            </div>
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
