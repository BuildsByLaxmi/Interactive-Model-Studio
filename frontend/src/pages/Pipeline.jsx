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
  TrendingDown,
  Split,
  Cpu,
  Sparkles
} from "lucide-react";
import api from "../api/client";
import { getMockDatasetById, mockDatasets } from "../api/mockData";
import PipelineCanvas from "../components/PipelineCanvas";
import CodePreview from "../components/CodePreview";
import { generatePythonScript } from "../services/codegen";

export default function Pipeline() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const datasetId = searchParams.get("datasetId") || mockDatasets[0].id;

  const [dataset, setDataset] = useState(null);
  const [selectedModel, setSelectedModel] = useState("RandomForestClassifier");
  const [activeCodeTab, setActiveCodeTab] = useState("python"); // "python" | "json" | "onnx"
  const [copied, setCopied] = useState(false);

  // Hyperparameters
  const [hyperparams, setHyperparams] = useState({
    n_estimators: 100,
    max_depth: 8,
    random_state: 42
  });

  const [reductionConfig, setReductionConfig] = useState({
    enabled: true,
    method: "pca",
    n_components: 2,
    k_best: 5,
    variance_threshold: 0.01,
  });

  const [splitConfig, setSplitConfig] = useState({
    train: 70,
    val: 15,
    test: 15,
    stratify: true,
    kfold: false,
    randomState: 42,
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

  // Default cleaning configuration
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
    splitConfig,
    cleaningConfigs,
    transformConfigs,
    reductionConfig,
    modelType: selectedModel,
    hyperparameters: hyperparams,
    exportOnnx: true,
  });

  const pipelineJson = {
    pipeline_name: "end_to_end_studio_pipeline",
    dataset: dataset?.id || datasetId,
    target: dataset?.target_column || "churn",
    chains: {
      stage_1_cleaning: [
        { step: "drop_duplicates" },
        { step: "drop_columns", columns: ["customerID"] },
        { step: "impute", strategy: "median", columns: ["TotalCharges"] },
        { step: "outlier_clip", method: "iqr", threshold: 1.5, columns: ["TotalCharges"] }
      ],
      stage_2_transformation: [
        { step: "math_transform", method: "log1p", columns: ["TotalCharges"] },
        { step: "scale", method: "StandardScaler", columns: ["TotalCharges", "MonthlyCharges"] },
        { step: "scale", method: "MinMaxScaler", columns: ["tenure"] },
        { step: "encode", method: "OneHotEncoder", columns: ["Contract", "PaymentMethod", "InternetService"] }
      ],
      stage_3_reduction: [
        { step: "pca", n_components: reductionConfig.n_components }
      ],
      stage_4_splitting: {
        train_ratio: 0.70,
        val_ratio: 0.15,
        test_ratio: 0.15,
        stratify: true,
        random_state: 42
      }
    },
    estimator: {
      algorithm: selectedModel,
      hyperparameters: hyperparams
    },
    export: {
      onnx_enabled: true,
      target_runtime: "onnxruntime-web"
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
            Complete end-to-end DAG: Data Cleaning ➔ Transformations ➔ Dimension Reduction (PCA) ➔ Splitting ➔ Estimator. Export as standalone Python scripts & ONNX edge artifacts.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Link 
            to={`/preprocessing?datasetId=${dataset?.id || datasetId}`}
            className="btn btn-secondary flex items-center gap-1.5"
          >
            <Sliders size={15} />
            <span>Configure All 4 Stages</span>
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
          reductionConfig={reductionConfig}
          splitConfig={splitConfig}
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
                Stage 1 • Cleaning & Outlier Handling
              </span>
              <span className="text-xs text-muted">Quality & Noise</span>
            </div>
            <ul className="pipeline-blueprint-list">
              <li>
                <div className="blueprint-bullet bg-rose-500"></div>
                <div><strong>Drop Redundant Columns:</strong> <code>customerID</code> (ID proxy)</div>
              </li>
              <li>
                <div className="blueprint-bullet bg-blue-500"></div>
                <div><strong>Median Imputation:</strong> <code>TotalCharges</code> (Null values)</div>
              </li>
              <li>
                <div className="blueprint-bullet bg-amber-500"></div>
                <div><strong>IQR 1.5× Winsorization:</strong> <code>TotalCharges</code> (Outlier clipping)</div>
              </li>
            </ul>
          </div>

          {/* Stage 2 Summary Card */}
          <div className="studio-card">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-accent">
                Stage 2 • Feature Transformations
              </span>
              <span className="text-xs text-muted">Feature Scaling</span>
            </div>
            <ul className="pipeline-blueprint-list">
              <li>
                <div className="blueprint-bullet bg-purple-500"></div>
                <div><strong>Log1p Math Transform:</strong> <code>TotalCharges</code> (Skewness fix)</div>
              </li>
              <li>
                <div className="blueprint-bullet bg-indigo-500"></div>
                <div><strong>StandardScaler:</strong> <code>TotalCharges</code>, <code>MonthlyCharges</code></div>
              </li>
              <li>
                <div className="blueprint-bullet bg-cyan-500"></div>
                <div><strong>OneHotEncoder:</strong> <code>Contract</code>, <code>PaymentMethod</code>, <code>InternetService</code></div>
              </li>
            </ul>
          </div>

          {/* Stage 3 & 4 Summary Card */}
          <div className="studio-card">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-rose-600">
                Stage 3 & 4 • Reduction & Splitting
              </span>
              <span className="text-xs text-muted">Orthogonal Projection</span>
            </div>
            <ul className="pipeline-blueprint-list">
              <li>
                <div className="blueprint-bullet bg-rose-600"></div>
                <div><strong>PCA Dimension Reduction:</strong> Compress to <code>{reductionConfig.n_components} components</code></div>
              </li>
              <li>
                <div className="blueprint-bullet bg-teal-600"></div>
                <div><strong>Stratified Split:</strong> 70% Train, 15% Validation, 15% Test</div>
              </li>
            </ul>
          </div>

          {/* Estimator Config Card */}
          <div className="studio-card">
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-bold uppercase tracking-wider text-emerald-600">
                Stage 5 • Model Estimator
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
        <div className="lg:col-span-7">
          <CodePreview
            code={generatedCode}
            language="python"
            filename="train_pipeline.py"
            jsonFallback={pipelineJson}
            title="Generated Standalone Pipeline"
            subtitle="Pure standalone scikit-learn script with PCA & ONNX export"
          />
        </div>
      </div>
    </div>
  );
}
