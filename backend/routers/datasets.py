from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Body
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import DatasetDB
from ..schemas import DatasetResponse
from ..services.dataset_service import process_and_save_dataset, load_dataset_dataframe
from ..services.data_processing_service import (
    merge_datasets,
    detect_outliers,
    detect_noise_and_quality,
    clean_dataset,
)

router = APIRouter(prefix="/api/datasets", tags=["Datasets"])


class MergeDatasetsRequest(BaseModel):
    primary_dataset_id: int
    secondary_dataset_id: int
    on: Optional[str] = None
    how: str = "inner"
    output_filename: Optional[str] = None


class CleanDatasetRequest(BaseModel):
    drop_columns: Optional[List[str]] = None
    drop_duplicates: bool = True
    impute_strategy: Optional[Dict[str, str]] = None
    outlier_clipping: Optional[Dict[str, Any]] = None
    save_as_new: bool = False



@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a CSV or Parquet dataset file and compute schema summary metadata."""
    if not (file.filename.endswith(".csv") or file.filename.endswith(".parquet")):
        raise HTTPException(status_code=400, detail="Only CSV and Parquet files are supported.")

    content = await file.read()
    try:
        record = process_and_save_dataset(file.filename, content, db)
        return record
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process dataset file: {str(e)}")


@router.get("", response_model=List[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    """List all uploaded datasets."""
    return db.query(DatasetDB).order_by(DatasetDB.created_at.desc()).all()


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Retrieve details for a specific dataset."""
    record = db.query(DatasetDB).filter(DatasetDB.id == dataset_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return record


@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: int, limit: int = 10, db: Session = Depends(get_db)):
    """Retrieve sample head rows for frontend data tables."""
    try:
        df = load_dataset_dataframe(dataset_id, db)
        return {
            "columns": list(df.columns),
            "total_rows": len(df),
            "sample_rows": df.head(limit).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{dataset_id}/diagnostics/outliers")
def get_outlier_and_noise_diagnostics(
    dataset_id: int,
    method: str = "iqr",
    threshold: float = 1.5,
    db: Session = Depends(get_db)
):
    """Deep scan for outliers, noise, zero variance features, and duplicate rows (Person A)."""
    try:
        df = load_dataset_dataframe(dataset_id, db)
        outlier_data = detect_outliers(df, method=method, threshold=threshold)
        noise_data = detect_noise_and_quality(df)
        return {
            "dataset_id": dataset_id,
            "outliers": outlier_data,
            "quality_and_noise": noise_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/merge")
def merge_and_integrate_datasets(
    request: MergeDatasetsRequest,
    db: Session = Depends(get_db)
):
    """Integrates two datasets via SQL-like joins into a unified dataset (Person A)."""
    try:
        df_prim = load_dataset_dataframe(request.primary_dataset_id, db)
        df_sec = load_dataset_dataframe(request.secondary_dataset_id, db)

        merged_df, summary = merge_datasets(
            df_prim,
            df_sec,
            on=request.on,
            how=request.how
        )

        filename = request.output_filename or f"integrated_{request.primary_dataset_id}_{request.secondary_dataset_id}.csv"
        csv_bytes = merged_df.to_csv(index=False).encode("utf-8")
        saved_record = process_and_save_dataset(filename, csv_bytes, db)

        return {
            "status": "success",
            "integration_summary": summary,
            "dataset": {
                "id": saved_record.id,
                "filename": saved_record.filename,
                "row_count": saved_record.row_count,
                "col_count": saved_record.col_count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Integration failed: {str(e)}")


@router.post("/{dataset_id}/clean")
def run_dataset_cleaning(
    dataset_id: int,
    request: CleanDatasetRequest,
    db: Session = Depends(get_db)
):
    """Executes data cleaning (dropping duplicates, imputation, outlier clipping) on dataset (Person A)."""
    try:
        df = load_dataset_dataframe(dataset_id, db)
        cleaned_df, summary = clean_dataset(
            df,
            drop_columns=request.drop_columns,
            drop_duplicates=request.drop_duplicates,
            impute_strategy=request.impute_strategy,
            outlier_clipping=request.outlier_clipping
        )

        if request.save_as_new:
            new_filename = f"cleaned_dataset_{dataset_id}.csv"
            csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
            saved_record = process_and_save_dataset(new_filename, csv_bytes, db)
            summary["saved_dataset_id"] = saved_record.id

        summary["sample_preview"] = cleaned_df.head(5).to_dict(orient="records")
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cleaning failed: {str(e)}")

