import React from "react";
import { 
  Database, 
  Brush, 
  Zap, 
  Cpu, 
  CheckCircle, 
  ArrowRight, 
  Layers, 
  Activity, 
  FileCode,
  ShieldCheck
} from "lucide-react";

export default function PipelineCanvas({
  dataset,
  cleaningCounts = { dropped: 0, imputed: 0, outliers: 0 },
  transformCounts = { math: 0, scaled: 0, encoded: 0 },
  selectedModel = "RandomForestClassifier",
  onSelectModel,
  onOpenStep
}) {
  return (
    <div className="pipeline-canvas-container">
      <div className="canvas-header flex justify-between items-center mb-4">
        <div>
          <h3 className="text-base font-bold text-main">Visual ML Pipeline Architecture</h3>
          <p className="text-xs text-muted">
            End-to-end directed acyclic graph (DAG) from raw tabular ingestion to trained estimator.
          </p>
        </div>
        <span className="pipeline-status-badge">
          <Activity size={13} className="text-success animate-pulse" />
          <span>Active Graph</span>
        </span>
      </div>

      <div className="canvas-flow">
        {/* Node 1: Ingestion */}
        <div className="canvas-node node-dataset">
          <div className="node-icon-wrapper bg-indigo-50 text-primary">
            <Database size={20} />
          </div>
          <div className="node-content">
            <span className="node-stage-tag">Source</span>
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

        {/* Node 2: Stage 1 - Data Cleaning Chain */}
        <div 
          className="canvas-node node-cleaning cursor-pointer hover:border-primary"
          onClick={() => onOpenStep && onOpenStep("cleaning")}
          title="Click to tune data cleaning"
        >
          <div className="node-icon-wrapper bg-blue-50 text-blue-600">
            <Brush size={20} />
          </div>
          <div className="node-content">
            <span className="node-stage-tag stage-clean">Stage 1</span>
            <h4 className="node-title">Data Cleaning Chain</h4>
            <div className="node-badges flex flex-wrap gap-1 mt-1">
              <span className="node-mini-badge bg-rose-50 text-rose-700">
                {cleaningCounts.dropped} Dropped
              </span>
              <span className="node-mini-badge bg-blue-50 text-blue-700">
                {cleaningCounts.imputed} Imputed
              </span>
              {cleaningCounts.outliers > 0 && (
                <span className="node-mini-badge bg-amber-50 text-amber-700">
                  {cleaningCounts.outliers} IQR Clip
                </span>
              )}
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
            <span className="node-stage-tag stage-trans">Stage 2</span>
            <h4 className="node-title">Feature Transforms</h4>
            <div className="node-badges flex flex-wrap gap-1 mt-1">
              {transformCounts.math > 0 && (
                <span className="node-mini-badge bg-purple-50 text-purple-700">
                  {transformCounts.math} Math Power
                </span>
              )}
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

        {/* Node 4: Model Estimator */}
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

        {/* Node 5: Output & Evaluation */}
        <div className="canvas-node node-output">
          <div className="node-icon-wrapper bg-amber-50 text-amber-600">
            <FileCode size={20} />
          </div>
          <div className="node-content">
            <span className="node-stage-tag stage-eval">Export</span>
            <h4 className="node-title">Python Codegen</h4>
            <div className="node-stats text-amber-800 font-semibold">
              Executable .py Script
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
