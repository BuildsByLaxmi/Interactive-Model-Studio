import { useState, useEffect } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { 
  Database, 
  Table, 
  Hash, 
  Layers, 
  Sliders, 
  ArrowRight, 
  Check, 
  Info,
  Calendar,
  Key
} from "lucide-react";
import api from "../api/client";
import { getMockDatasetById } from "../api/mockData";
import DiagnosticsPanel from "../components/DiagnosticsPanel";
import DiagnosticsDetail from "../components/DiagnosticsDetail";

export default function DatasetView() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [dataset, setDataset] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    setLoading(true);
    // Attempt live fetch first, fallback to mock data
    api.getDatasets()
      .then((data) => {
        const found = Array.isArray(data) ? data.find((d) => d.id === id) : null;
        if (found) {
          setDataset(found);
          setSelectedTarget(found.target_column || found.schema?.[found.schema.length - 1]?.name || "");
        } else {
          const fallback = getMockDatasetById(id);
          setDataset(fallback);
          setSelectedTarget(fallback.target_column);
        }
      })
      .catch(() => {
        const fallback = getMockDatasetById(id);
        setDataset(fallback);
        setSelectedTarget(fallback.target_column);
      })
      .finally(() => {
        setLoading(false);
      });
  }, [id]);

  if (loading) {
    return (
      <div className="page-container flex justify-center items-center py-20">
        <div className="text-center">
          <div className="animate-spin text-primary mb-3">⟳</div>
          <p className="text-gray-500 font-medium">Loading schema & running diagnostics...</p>
        </div>
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="page-container">
        <div className="studio-card text-center py-12">
          <h2>Dataset Not Found</h2>
          <p className="text-muted mt-2">Could not locate dataset details.</p>
          <Link to="/upload" className="btn btn-primary mt-4">Return to Upload</Link>
        </div>
      </div>
    );
  }

  const filteredColumns = (dataset.schema || []).filter((col) => 
    col.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    col.type.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="page-container">
      {/* Top Banner with Metadata */}
      <div className="dataset-header-card">
        <div className="flex justify-between items-start flex-wrap gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="dataset-badge">v{dataset.version || 1}</span>
              <h1 className="dataset-heading">{dataset.name}</h1>
            </div>
            <div className="dataset-meta-row">
              <span className="meta-item">
                <Hash size={14} /> SHA-256: <code>{dataset.file_hash ? `${dataset.file_hash.substring(0, 16)}...` : "e3b0c442..."}</code>
              </span>
              <span className="meta-item">
                <Layers size={14} /> {dataset.row_count?.toLocaleString()} Rows
              </span>
              <span className="meta-item">
                <Table size={14} /> {dataset.column_count || dataset.schema?.length} Columns
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Link to="/upload" className="btn btn-secondary">
              Upload New
            </Link>
            <button 
              className="btn btn-primary flex items-center gap-2"
              onClick={() => navigate(`/preprocessing?datasetId=${dataset.id}&target=${selectedTarget}`)}
            >
              <Sliders size={16} />
              <span>Configure Preprocessing</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Diagnostics Alert Engine */}
      <div className="mb-8 space-y-4">
        <DiagnosticsPanel diagnostics={dataset.diagnostics || []} />
        <DiagnosticsDetail 
          dataset={dataset} 
          diagnostics={dataset.diagnostics || []} 
        />
      </div>

      {/* Schema Inspection Section */}
      <div className="studio-card">
        <div className="flex justify-between items-center mb-4 flex-wrap gap-3">
          <div>
            <h2 className="card-heading">Inferred Column Schema & Data Types</h2>
            <p className="card-description">
              Verify column distributions, null ratios, and pick the target prediction variable.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <input 
              type="text" 
              placeholder="Filter columns..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>
        </div>

        <div className="table-responsive">
          <table className="studio-table">
            <thead>
              <tr>
                <th>Target</th>
                <th>Column Name</th>
                <th>Inferred Type</th>
                <th>Null Count</th>
                <th>Missing %</th>
                <th>Cardinality</th>
                <th>Sample Value</th>
              </tr>
            </thead>
            <tbody>
              {filteredColumns.map((col) => {
                const isTarget = selectedTarget === col.name;
                const isHighNull = col.null_pct > 30;
                const isHighCard = col.cardinality > 50 && col.type === "categorical";

                return (
                  <tr key={col.name} className={isTarget ? "target-row" : ""}>
                    <td className="text-center">
                      <button 
                        className={`target-select-btn ${isTarget ? "active" : ""}`}
                        onClick={() => setSelectedTarget(col.name)}
                        title={isTarget ? "Current prediction target" : "Set as prediction target"}
                      >
                        {isTarget ? <Key size={14} className="text-primary" /> : <div className="radio-dot" />}
                      </button>
                    </td>
                    <td>
                      <strong className="column-name">{col.name}</strong>
                      {isTarget && <span className="target-badge">Target (y)</span>}
                    </td>
                    <td>
                      <span className={`type-tag type-${col.type}`}>
                        {col.type}
                      </span>
                    </td>
                    <td>{col.null_count?.toLocaleString()}</td>
                    <td>
                      <div className="flex items-center gap-2">
                        <span className={isHighNull ? "text-danger font-bold" : ""}>
                          {col.null_pct?.toFixed(1)}%
                        </span>
                        {isHighNull && <span className="pill-warn">High</span>}
                      </div>
                    </td>
                    <td>
                      <div className="flex items-center gap-2">
                        <span>{col.cardinality?.toLocaleString()}</span>
                        {isHighCard && <span className="pill-warn">High</span>}
                      </div>
                    </td>
                    <td>
                      <code className="sample-val">{col.sample || "-"}</code>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
