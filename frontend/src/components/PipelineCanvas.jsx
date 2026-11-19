import React from "react";
import { 
  Database, 
  Brush, 
  Zap, 
  Cpu, 
  ArrowRight, 
  Activity, 
  FileCode,
  TrendingDown,
  Split
} from "lucide-react";

export default function PipelineCanvas({
  dataset,
  cleaningCounts = { dropped: 1, imputed: 2, outliers: 1 },
  transformCounts = { math: 1, scaled: 3, encoded: 3 },
  reductionConfig = { enabled: true, method: "pca", n_components: 2 },
  splitConfig = { train: 70, val: 15, test: 15 },
  selectedModel = "RandomForestClassifier",
  onSelectModel,
  onOpenStep
}) {
  return (
    <div className="pipeline-canvas-container">
      <div className="canvas-header flex justify-between items-center mb-4">
        <div>
          <h3 className="text-base font-bold text-main">Complete End-to-End ML Pipeline Architecture</h3>
          <p className="text-xs text-muted">
            Directed Acyclic Graph (DAG): Ingestion ➔ Data Cleaning ➔ Transformations ➔ Dimension Reduction ➔ Splitting ➔ Estimator
          </p>
        </div>
        <span className="pipeline-status-badge">
          <Activity size={13} className="text-success animate-pulse" />
          <span>Active Pipeline Graph</span>
        </span>
      </div>

      <div className="canvas-flow">
        {/* Node 1: Ingestion & Integration */}
        <div className="canvas-node node-dataset">
          <div className="node-icon-wrapper bg-indigo-50 text-primary">
            <Database size={20} />
          </div>
          <div className="node-content">
            <span className="node-stage-tag">Source / Ingest</span>
            <h4 className="node-title">{dataset?.name || "Dataset"}</h4>
            <div className="node-stats">
              <span>{dataset?.rows || 1000} rows</span>
              <span>•</span>
              <span>{dataset?.columns || 12} cols</span>
            </div>
          </div>
        </div>

        <div className="canvas-arrow">
          <ArrowRight size={18} />
        </div>

        {/* Node 2: Stage 1 - Data Cleaning & Outlier / Noise Handling */}
        <div 
          className="canvas-node node-cleaning cursor-pointer hover:border-primary"
          onClick={() => onOpenStep && onOpenStep("cleaning")}
          title="Click to tune data cleaning and outlier clipping"
        >
          <div className="node-icon-wrapper bg-blue-50 text-blue-600">
            <Brush size={20} />
          </div>
          <div className="node-content">
            <span className="node-stage-tag stage-clean">Stage 1: Clean</span>
            <h4 className="node-title">Noise & Outliers</h4>
            <div className="node-badges flex flex-wrap gap-1 mt-1">
              <span className="node-mini-badge bg-rose-50 text-rose-700">
                {cleaningCounts.dropped} Drop
              </span>
              <span className="node-mini-badge bg-blue-50 text-blue-700">
                {cleaningCounts.imputed} Imputed
              </span>
              <span className="node-mini-badge bg-amber-50 text-amber-700">
                {cleaningCounts.outliers} IQR Clip
              </span>
            </div>
          </div>
        </div>

        <div className="canvas-arrow">
          <ArrowRight size={18} />
        </div>

        {/* Node 3: Stage 2 - Feature Transformation Chain */}
        <div 
          className="canvas-node node-transform cursor-pointer hover:border-accent"
          onClick={() => onOpenStep && onOpenStep("transform")}
          title="Click to tune transformations"
        >
          <div className="node-icon-wrapper bg-violet-50 text-accent">
            <Zap size={20} />
          </div>
          <div className="node-content">
            <span className="node-stage-tag stage-trans">Stage 2: Transform</span>
            <h4 className="node-title">Feature Scaling</h4>
            <div className="node-badges flex flex-wrap gap-1 mt-1">
              <span className="node-mini-badge bg-purple-50 text-purple-700">
                {transformCounts.math} Math Power
              </span>
              <span className="node-mini-badge bg-indigo-50 text-indigo-700">
                {transformCounts.scaled} Scaled
              </span>
              <span className="node-mini-badge bg-cyan-50 text-cyan-700">
                {transformCounts.encoded} Encoded
              </span>
            </div>
          </div>
        </div>

        <div className="canvas-arrow">
          <ArrowRight size={18} />
        </div>

        {/* Node 4: Stage 3 - Dimension Reduction */}
        <div 
          className="canvas-node node-reduction cursor-pointer hover:border-rose-400"
          onClick={() => onOpenStep && onOpenStep("reduction")}
          title="Click to tune PCA and dimensionality reduction"
        >
          <div className="node-icon-wrapper bg-rose-50 text-rose-600">
            <TrendingDown size={20} />
          </div>
          <div className="node-content">
            <span className="node-stage-tag" style={{ background: "#ffe4e6", color: "#9f1239" }}>
              Stage 3: Reduction
            </span>
            <h4 className="node-title">PCA Dimension</h4>
            <div className="node-badges flex flex-wrap gap-1 mt-1">
              <span className="node-mini-badge bg-rose-100 text-rose-800 font-bold">
                {reductionConfig.enabled ? `${reductionConfig.n_components} Components` : "Bypassed"}
              </span>
            </div>
          </div>
        </div>

        <div className="canvas-arrow">
          <ArrowRight size={18} />
        </div>

        {/* Node 5: Stage 4 - Data Splitting */}
        <div 
          className="canvas-node node-split cursor-pointer hover:border-teal-400"
          onClick={() => onOpenStep && onOpenStep("splitting")}
          title="Click to adjust train/val/test splits"
        >
          <div className="node-icon-wrapper bg-teal-50 text-teal-600">
            <Split size={20} />
          </div>
          <div className="node-content">
            <span className="node-stage-tag" style={{ background: "#ccfbf1", color: "#115e59" }}>
              Stage 4: Splitting
            </span>
            <h4 className="node-title">Train / Val / Test</h4>
            <div className="node-stats font-bold text-teal-800">
              {splitConfig.train}% / {splitConfig.val}% / {splitConfig.test}%
            </div>
          </div>
        </div>

        <div className="canvas-arrow">
          <ArrowRight size={18} />
        </div>

        {/* Node 6: Model Estimator */}
        <div className="canvas-node node-estimator">
          <div className="node-icon-wrapper bg-emerald-50 text-emerald-600">
            <Cpu size={20} />
          </div>
          <div className="node-content">
            <span className="node-stage-tag stage-model">Estimator</span>
            <select
              value={selectedModel}
              onChange={(e) => onSelectModel && onSelectModel(e.target.value)}
              className="canvas-algo-select"
            >
              <option value="RandomForestClassifier">Random Forest</option>
              <option value="XGBClassifier">XGBoost Classifier</option>
              <option value="LogisticRegression">Logistic Regression</option>
              <option value="DecisionTreeClassifier">Decision Tree</option>
            </select>
            <div className="node-stats text-emerald-700 font-semibold">
              Supervised Classifier
            </div>
          </div>
        </div>

        <div className="canvas-arrow">
          <ArrowRight size={18} />
        </div>

        {/* Node 7: Output & Edge Export */}
        <div className="canvas-node node-output">
          <div className="node-icon-wrapper bg-amber-50 text-amber-600">
            <FileCode size={20} />
          </div>
          <div className="node-content">
            <span className="node-stage-tag stage-eval">Export</span>
            <h4 className="node-title">Python & ONNX</h4>
            <div className="node-stats text-amber-800 font-semibold">
              Standalone Script
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
