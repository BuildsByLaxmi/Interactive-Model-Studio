import React, { useState } from "react";
import { 
  FileCode, 
  Code2, 
  Copy, 
  Check, 
  Download, 
  WrapText, 
  Sparkles,
  Terminal
} from "lucide-react";

export default function CodePreview({
  code,
  language = "python",
  filename = "train_pipeline.py",
  jsonFallback = null,
  title = "Generated Code",
  subtitle = "Zero-dependency standalone script ready for local execution"
}) {
  const [activeTab, setActiveTab] = useState("code"); // "code" | "json"
  const [copied, setCopied] = useState(false);
  const [lineWrap, setLineWrap] = useState(false);

  const displayCode = activeTab === "code" 
    ? code 
    : JSON.stringify(jsonFallback || {}, null, 2);

  const handleCopy = () => {
    navigator.clipboard.writeText(displayCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const ext = activeTab === "code" ? "py" : "json";
    const fname = activeTab === "code" ? filename : "pipeline_schema.json";
    const mime = activeTab === "code" ? "text/x-python" : "application/json";
    
    const blob = new Blob([displayCode], { type: mime });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fname;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const lineCount = displayCode ? displayCode.split("\n").length : 0;

  return (
    <div className="codegen-container studio-card">
      <div className="codegen-header flex justify-between items-center pb-3 mb-3 border-b border-slate-200 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <button
            className={`codegen-tab-btn ${activeTab === "code" ? "active" : ""}`}
            onClick={() => setActiveTab("code")}
          >
            <FileCode size={15} />
            <span>{filename}</span>
          </button>
          {jsonFallback && (
            <button
              className={`codegen-tab-btn ${activeTab === "json" ? "active" : ""}`}
              onClick={() => setActiveTab("json")}
            >
              <Code2 size={15} />
              <span>pipeline_schema.json</span>
            </button>
          )}
        </div>

        <div className="flex items-center gap-2">
          <button
            className={`btn btn-secondary btn-sm flex items-center gap-1 ${lineWrap ? "active" : ""}`}
            onClick={() => setLineWrap(!lineWrap)}
            title="Toggle word wrap"
          >
            <WrapText size={14} />
            <span className="hidden sm:inline">Wrap</span>
          </button>
          <button 
            className="btn btn-secondary btn-sm flex items-center gap-1"
            onClick={handleCopy}
            title="Copy script to clipboard"
          >
            {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
            <span>{copied ? "Copied!" : "Copy"}</span>
          </button>
          <button 
            className="btn btn-primary btn-sm flex items-center gap-1"
            onClick={handleDownload}
            title="Download standalone file"
          >
            <Download size={14} />
            <span>Download</span>
          </button>
        </div>
      </div>

      <div className="codegen-editor-wrapper">
        <pre className={`codegen-code-box ${lineWrap ? "whitespace-pre-wrap" : ""}`}>
          <code>{displayCode}</code>
        </pre>
      </div>

      <div className="codegen-footer mt-3 pt-3 border-t border-slate-200 flex justify-between items-center text-xs text-muted flex-wrap gap-2">
        <span className="flex items-center gap-1">
          <Sparkles size={13} className="text-primary" />
          <span>{subtitle}</span>
        </span>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1 text-slate-500 font-mono">
            <Terminal size={12} />
            python {filename}
          </span>
          <span className="font-mono bg-slate-100 px-2 py-0.5 rounded text-slate-700">
            {lineCount} lines
          </span>
        </div>
      </div>
    </div>
  );
}
