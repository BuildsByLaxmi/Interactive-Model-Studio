import { useState, useEffect } from "react";
import { useSearchParams, useNavigate, Link } from "react-router-dom";
import { 
  Code2, 
  Copy, 
  Check, 
  Download, 
  Play, 
  Sliders, 
  ArrowRight, 
  Layers, 
  FileCode, 
  Settings2,
  Sparkles,
  ExternalLink,
  RefreshCw
} from "lucide-react";
import api from "../api/client";
import { getMockDatasetById, mockDatasets } from "../api/mockData";
import PipelineCanvas from "../components/PipelineCanvas";
import { generatePythonScript } from "../services/codegen";

export default function Pipeline() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const datasetId = searchParams.get("datasetId") || mockDatasets[0].id;

  const [dataset, setDataset] = useState(null);
  const [selectedModel, setSelectedModel] = useState("RandomForestClassifier");
  const [activeCodeTab, setActiveCodeTab] = useState("python"); // "python" | "json"
  const [copied, setCopied] = useState(false);

  // Hyperparameters
  const [hyperparams, setHyperparams] = useState({
    n_estimators: 100,
    max_depth: 8,
    random_state: 42
  });

  useEffect(() => {
    api.getDatasets()
      .then((data) => {
        const found = Array.isArray(data) ? data.find((d) => d.id === datasetId) : null;
        setDataset(found || getMockDatasetById(datasetId));
      })
      .catch(() => {
        setDataset(getMockDatasetById(datasetId));
      });
  }, [datasetId]);

  // Default cleaning configuration for the current dataset
  const cleaningConfigs = {
    customerID: { action: "drop", imputer: "none", handleOutliers: false },
    TotalCharges: { action: "keep", imputer: "median", handleOutliers: true },
    MonthlyCharges: { action: "keep", imputer: "none", handleOutliers: false },
    tenure: { action: "keep", imputer: "none", handleOutliers: false },
  };

  // Default transformation configuration
  const transformConfigs = {
    TotalCharges: { type: "numerical", mathTransform: "log1p", scaler: "StandardScaler" },
    MonthlyCharges: { type: "numerical", mathTransform: "none", scaler: "StandardScaler" },
    tenure: { type: "numerical", mathTransform: "none", scaler: "MinMaxScaler" },
    Contract: { type: "categorical", encoder: "OneHotEncoder" },
    PaymentMethod: { type: "categorical", encoder: "OneHotEncoder" },
    InternetService: { type: "categorical", encoder: "OneHotEncoder" },
  };

  // Generate real Python script
  const generatedCode = generatePythonScript({
    datasetName: dataset ? `${dataset.id}.csv` : "dataset.csv",
    targetColumn: dataset?.target_column || "churn",
    testSize: 0.2,
    randomState: 42,
    cleaningConfigs,
    transformConfigs,
    modelType: selectedModel,
    hyperparameters: hyperparams
  });

  const pipelineJson = {
    name: "chained_automl_pipeline",
    dataset: dataset?.id || datasetId,
    target: dataset?.target_column || "churn",
    chains: {
      cleaning: [
        { step: "drop", columns: ["customerID"] },
        { step: "impute", strategy: "median", columns: ["TotalCharges"] },
        { step: "outlier_clipping", threshold: 1.5, columns: ["TotalCharges"] }
      ],
      transformations: [
        { step: "log_transform", columns: ["TotalCharges"] },
        { step: "scale", method: "StandardScaler", columns: ["TotalCharges", "MonthlyCharges"] },
        { step: "scale", method: "MinMaxScaler", columns: ["tenure"] },
        { step: "encode", method: "OneHotEncoder", columns: ["Contract", "PaymentMethod", "InternetService"] }
      ]
    },
    estimator: {
      algorithm: selectedModel,
      hyperparameters: hyperparams
    }
  };

  const copyCode = () => {
    const textToCopy = activeCodeTab === "python" ? generatedCode : JSON.stringify(pipelineJson, null, 2);
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadFile = () => {
    const isPy = activeCodeTab === "python";
    const content = isPy ? generatedCode : JSON.stringify(pipelineJson, null, 2);
    const filename = isPy ? `train_${selectedModel.toLowerCase()}.py` : `pipeline_config.json`;
    const mime = isPy ? "text/x-python" : "application/json";

    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const runAsExperiment = () => {
    navigate(`/experiments/run?datasetId=${dataset?.id || datasetId}&modelType=${selectedModel}`);
  };

  return (
    <div className="page-container">
      {/* Header */}
      <div className="page-header flex justify-between items-start flex-wrap gap-4">
        <div>
          <h1 className="page-title">Visual Pipeline Studio & Python Codegen</h1>
          <p className="page-subtitle">
            Construct reproducible end-to-end ML pipelines with chained data cleaning and feature transformations. Export as standalone, executable Python scripts.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Link 
            to={`/preprocessing?datasetId=${dataset?.id || datasetId}`}
            className="btn btn-secondary flex items-center gap-1.5"
          >
            <Sliders size={15} />
            <span>Tune Chained Recipes</span>
          </Link>
          <button 
            className="btn btn-primary flex items-center gap-1.5"
            onClick={runAsExperiment}
          >
            <Play size={15} />
            <span>Run as Experiment</span>
          </button>
        </div>
      </div>

      {/* Interactive Visual DAG Canvas */}
      <div className="mb-6">
        <PipelineCanvas 
          dataset={dataset}
          cleaningCounts={{ dropped: 1, imputed: 1, outliers: 1 }}
          transformCounts={{ math: 1, scaled: 3, encoded: 3 }}
          selectedModel={selectedModel}
          onSelectModel={setSelectedModel}
          onOpenStep={(step) => navigate(`/preprocessing?datasetId=${dataset?.id || datasetId}`)}
        />
      </div>

      {/* Dual Column Layout: Pipeline Config Summary + Python Codegen Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Col: Chained Pipeline Blueprint (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          {/* Stage 1 Summary Card */}
          <div className="studio-card">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-primary">
                Stage 1 • Cleaning Chain
              </span>
              <span className="text-xs text-muted">Executed on Raw Data</span>
            </div>
            <ul className="pipeline-blueprint-list">
              <li>
                <div className="blueprint-bullet bg-rose-500"></div>
                <div>
                  <strong>Drop Features:</strong> <code>customerID</code> (Proxy ID / High Unique)
                </div>
              </li>
              <li>
                <div className="blueprint-bullet bg-blue-500"></div>
                <div>
                  <strong>Median Imputation:</strong> <code>TotalCharges</code> (Handles nulls)
                </div>
              </li>
              <li>
                <div className="blueprint-bullet bg-amber-500"></div>
                <div>
                  <strong>Outlier IQR Clipping:</strong> <code>TotalCharges</code> (Capped at 1.5× IQR)
                </div>
              </li>
            </ul>
          </div>

          {/* Stage 2 Summary Card */}
          <div className="studio-card">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-accent">
                Stage 2 • Feature Transform Chain
              </span>
              <span className="text-xs text-muted">Feature Engineering</span>
            </div>
            <ul className="pipeline-blueprint-list">
              <li>
                <div className="blueprint-bullet bg-purple-500"></div>
                <div>
                  <strong>Log1p Math Transform:</strong> <code>TotalCharges</code> (Fixes skewness)
                </div>
              </li>
              <li>
                <div className="blueprint-bullet bg-indigo-500"></div>
                <div>
                  <strong>StandardScaler:</strong> <code>TotalCharges</code>, <code>MonthlyCharges</code>
                </div>
              </li>
              <li>
                <div className="blueprint-bullet bg-cyan-500"></div>
                <div>
                  <strong>OneHotEncoder:</strong> <code>Contract</code>, <code>PaymentMethod</code>, <code>InternetService</code>
                </div>
              </li>
            </ul>
          </div>

          {/* Estimator Config Card */}
          <div className="studio-card">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-600">
                Stage 3 • Estimator Hyperparameters
              </span>
              <span className="text-xs font-semibold text-emerald-700">{selectedModel}</span>
            </div>
            <div className="grid grid-cols-2 gap-3 text-xs mt-3">
              <div className="p-2 bg-slate-50 rounded border border-slate-200">
                <span className="text-muted block">n_estimators</span>
                <strong>{hyperparams.n_estimators}</strong>
              </div>
              <div className="p-2 bg-slate-50 rounded border border-slate-200">
                <span className="text-muted block">max_depth</span>
                <strong>{hyperparams.max_depth}</strong>
              </div>
            </div>
          </div>
        </div>

        {/* Right Col: Live Standalone Python Code Generator (7 cols) */}
        <div className="lg:col-span-7 studio-card codegen-container">
          <div className="codegen-header flex justify-between items-center pb-3 mb-3 border-b border-slate-200">
            <div className="flex items-center gap-2">
              <button
                className={`codegen-tab-btn ${activeCodeTab === "python" ? "active" : ""}`}
                onClick={() => setActiveCodeTab("python")}
              >
                <FileCode size={15} />
                <span>train_pipeline.py</span>
              </button>
              <button
                className={`codegen-tab-btn ${activeCodeTab === "json" ? "active" : ""}`}
                onClick={() => setActiveCodeTab("json")}
              >
                <Code2 size={15} />
                <span>pipeline_schema.json</span>
              </button>
            </div>

            <div className="flex items-center gap-2">
              <button 
                className="btn btn-secondary btn-sm flex items-center gap-1"
                onClick={copyCode}
                title="Copy script to clipboard"
              >
                {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
                <span>{copied ? "Copied!" : "Copy"}</span>
              </button>
              <button 
                className="btn btn-primary btn-sm flex items-center gap-1"
                onClick={downloadFile}
                title="Download standalone file"
              >
                <Download size={14} />
                <span>Download</span>
              </button>
            </div>
          </div>

          <div className="codegen-editor-wrapper">
            <pre className="codegen-code-box">
              <code>
                {activeCodeTab === "python" ? generatedCode : JSON.stringify(pipelineJson, null, 2)}
              </code>
            </pre>
          </div>

          <div className="codegen-footer mt-3 pt-3 border-t border-slate-200 flex justify-between items-center text-xs text-muted">
            <span>✨ Pure standalone scikit-learn script — no framework lock-in</span>
            <span className="font-mono">{generatedCode.split("\n").length} lines generated</span>
          </div>
        </div>
      </div>
    </div>
  );
}
