import React, { useState } from "react";
import { 
  AlertTriangle, 
  AlertOctagon, 
  CheckCircle2, 
  Lightbulb, 
  ShieldCheck, 
  Activity, 
  ArrowRight,
  TrendingDown,
  Layers,
  Sparkles
} from "lucide-react";
import { Link } from "react-router-dom";

export default function DiagnosticsDetail({ 
  dataset, 
  diagnostics = [], 
  outlierReport = null,
  onApplyPreset 
}) {
  const [selectedCategory, setSelectedCategory] = useState("all");

  const categories = [
    { id: "all", label: `All Issues (${diagnostics.length})` },
    { id: "leakage", label: "Leakage & Imbalance" },
    { id: "quality", label: "Missing & Cardinality" },
    { id: "outliers", label: "Outliers & Noise" },
  ];

  const filteredDiagnostics = diagnostics.filter((diag) => {
    if (selectedCategory === "all") return true;
    if (selectedCategory === "leakage") return diag.type === "leakage" || diag.type === "class_imbalance";
    if (selectedCategory === "quality") return diag.type === "high_nulls" || diag.type === "high_cardinality";
    if (selectedCategory === "outliers") return diag.type === "outlier" || diag.type === "noise";
    return true;
  });

  return (
    <div className="diagnostics-detail-container studio-card mt-6">
      <div className="flex justify-between items-start flex-wrap gap-4 mb-4 pb-3 border-b border-slate-200">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-50 text-primary rounded-lg">
            <Sparkles size={24} />
          </div>
          <div>
            <h3 className="text-base font-bold text-main">Deep Diagnostics & Automated Quality Engine</h3>
            <p className="text-xs text-muted">
              Rule-based checks with actionable remediation advice for production-ready data preparation.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {onApplyPreset && (
            <button 
              className="btn btn-primary btn-sm flex items-center gap-1.5"
              onClick={() => onApplyPreset("recommended")}
            >
              <Lightbulb size={14} />
              <span>Apply Recommended Preprocessing</span>
            </button>
          )}
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-2 mb-4 overflow-x-auto pb-1">
        {categories.map((cat) => (
          <button
            key={cat.id}
            onClick={() => setSelectedCategory(cat.id)}
            className={`diag-filter-btn ${selectedCategory === cat.id ? "active" : ""}`}
          >
            {cat.label}
          </button>
        ))}
      </div>

      {/* Diagnostics List */}
      <div className="diagnostics-list">
        {filteredDiagnostics.length === 0 ? (
          <div className="p-6 text-center text-muted border border-dashed rounded-lg bg-slate-50">
            <ShieldCheck size={28} className="text-success mx-auto mb-2" />
            <p className="font-semibold text-sm">No issues detected in this category!</p>
            <span className="text-xs">Your dataset conforms to best practices for model training.</span>
          </div>
        ) : (
          filteredDiagnostics.map((diag, idx) => (
            <div 
              key={idx} 
              className={`diag-card ${diag.severity === "red" ? "diag-card-red" : "diag-card-yellow"}`}
            >
              <div className="diag-card-icon">
                {diag.severity === "red" ? (
                  <AlertOctagon size={20} className="text-danger" />
                ) : (
                  <AlertTriangle size={20} className="text-warning" />
                )}
              </div>
              <div className="diag-card-content">
                <div className="diag-card-top">
                  <h4 className="diag-card-title">{diag.title}</h4>
                  {diag.column && (
                    <span className="diag-column-tag">
                      <code>{diag.column}</code>
                    </span>
                  )}
                </div>
                <p className="diag-description">{diag.description}</p>
                {diag.suggestion && (
                  <div className="diag-suggestion">
                    <Lightbulb size={14} className="suggestion-icon" />
                    <span><strong>Actionable Fix:</strong> {diag.suggestion}</span>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Outlier Deep Breakdown Preview */}
      {outlierReport && outlierReport.columns_with_outliers > 0 && (
        <div className="mt-6 pt-4 border-t border-slate-200">
          <h4 className="text-xs font-bold uppercase tracking-wider text-muted mb-2">
            Automated Outlier Scan Results (IQR 1.5× Threshold)
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {Object.entries(outlierReport.column_details || {}).map(([col, details]) => (
              <div key={col} className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs">
                <div className="flex justify-between items-center mb-1">
                  <strong>{col}</strong>
                  <span className="text-rose-600 font-bold">{details.percentage}% outliers</span>
                </div>
                <span className="text-muted block">Bounds: [{details.lower_bound}, {details.upper_bound}]</span>
                <span className="text-muted block mt-1">Observed max: {details.max_observed}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
