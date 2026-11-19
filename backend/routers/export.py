"""
Export Router (Person B)
Provides endpoints to export fitted models to ONNX and Edge JSON formats
for client-side in-browser inference.
"""

import os
import joblib
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import RunDB
from ml_engine.export_onnx import export_model_to_onnx

router = APIRouter(prefix="/api/export", tags=["Export & Edge Deployment"])

STORAGE_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "models")


@router.post("/onnx/{run_id}")
def generate_onnx_export(run_id: int, db: Session = Depends(get_db)):
    """Exports a completed experiment run into an ONNX edge artifact."""
    run = db.query(RunDB).filter(RunDB.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")

    if not run.model_artifact_path or not os.path.exists(run.model_artifact_path):
        raise HTTPException(status_code=400, detail="Fitted model artifact not available on disk.")

    artifact = joblib.load(run.model_artifact_path)
    model = artifact.get("fitted_model")
    feature_names = artifact.get("config", {}).get("recipe", {}).get("columns") or [f"feature_{i}" for i in range(10)]

    output_filename = f"model_run_{run_id}.onnx"
    output_path = os.path.join(STORAGE_MODELS_DIR, output_filename)

    export_result = export_model_to_onnx(
        fitted_model=model,
        feature_names=feature_names,
        output_path=output_path
    )

    return {
        "status": "success",
        "run_id": run_id,
        "algorithm": run.algorithm,
        "export_details": export_result,
        "download_url": f"/api/export/download/{output_filename}"
    }


@router.get("/download/{filename}")
def download_exported_artifact(filename: str):
    """Downloads an exported ONNX or Edge JSON model file."""
    file_path = os.path.join(STORAGE_MODELS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    media_type = "application/octet-stream" if filename.endswith(".onnx") else "application/json"
    return FileResponse(file_path, media_type=media_type, filename=filename)
