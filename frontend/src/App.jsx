import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Navbar from "./components/Navbar";
import Upload from "./pages/Upload";
import DatasetView from "./pages/DatasetView";
import Preprocessing from "./pages/Preprocessing";
import ModelConfig from "./pages/ModelConfig";
import RunExperiment from "./pages/RunExperiment";
import ExperimentList from "./pages/ExperimentList";
import RestoreExperiment from "./pages/RestoreExperiment";
import Pipeline from "./pages/Pipeline";
import EdgeExport from "./pages/EdgeExport";
import Dashboard from "./pages/Dashboard";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Navigate to="/upload" replace />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/dataset/:id" element={<DatasetView />} />
            <Route path="/preprocessing" element={<Preprocessing />} />
            <Route path="/models" element={<ModelConfig />} />
            <Route path="/experiments/run" element={<RunExperiment />} />
            <Route path="/experiments" element={<ExperimentList />} />
            <Route path="/restore/:id" element={<RestoreExperiment />} />
            <Route path="/pipeline" element={<Pipeline />} />
            <Route path="/edge-export" element={<EdgeExport />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="*" element={<Navigate to="/upload" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}