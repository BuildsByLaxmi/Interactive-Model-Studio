import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { 
  Database, 
  Sliders,
  Cpu,
  History,
  Workflow, 
  BarChart3, 
  CheckCircle2, 
  AlertCircle, 
  Loader2,
  Sparkles,
  Zap
} from "lucide-react";
import api from "../api/client";

export default function Navbar() {
  const [status, setStatus] = useState("checking"); // 'checking' | 'connected' | 'error'
  const [backendMessage, setBackendMessage] = useState("");

  const checkConnection = () => {
    setStatus("checking");
    api.healthCheck()
      .then((data) => {
        setStatus("connected");
        setBackendMessage(data?.message || "Connected");
      })
      .catch(() => {
        setStatus("error");
        setBackendMessage("Offline");
      });
  };

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="navbar">
      <div className="nav-container">
        <div className="brand-section">
          <div className="brand-icon">
            <Sparkles className="icon-pulse" size={20} />
          </div>
          <div className="brand-text">
            <span className="brand-title">Interactive Model Studio</span>
            <span className="brand-tagline">Visual ML & Codegen Platform</span>
          </div>
        </div>

        <nav className="nav-links">
          <NavLink 
            to="/upload" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <Database size={15} />
            <span>Datasets</span>
          </NavLink>
          <NavLink 
            to="/preprocessing" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <Sliders size={15} />
            <span>Recipe</span>
          </NavLink>
          <NavLink 
            to="/models" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <Cpu size={15} />
            <span>Model Config</span>
          </NavLink>
          <NavLink 
            to="/experiments" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <History size={15} />
            <span>Experiments</span>
          </NavLink>
          <NavLink 
            to="/pipeline" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <Workflow size={15} />
            <span>Pipeline DAG</span>
          </NavLink>
          <NavLink 
            to="/dashboard" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <BarChart3 size={15} />
            <span>Dashboard</span>
          </NavLink>
          <NavLink 
            to="/edge-export" 
            className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}
          >
            <Zap size={15} />
            <span>Edge Deploy</span>
          </NavLink>
        </nav>

        <div className="status-section">
          <div 
            className={`status-pill ${status}`} 
            title={`Backend: ${backendMessage} (http://127.0.0.1:8000)`}
            onClick={checkConnection}
          >
            {status === "connected" && <CheckCircle2 size={14} className="text-success" />}
            {status === "checking" && <Loader2 size={14} className="animate-spin text-warning" />}
            {status === "error" && <AlertCircle size={14} className="text-danger" />}
            <span>
              {status === "connected" ? "Backend Live" : status === "checking" ? "Checking API..." : "Backend Offline"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
