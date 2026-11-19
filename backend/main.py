import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .routers import datasets, pipelines, runs, predict, export

# Initialize DB tables
init_db()

app = FastAPI(
    title="Interactive Model Studio API",
    description="Backend API for visual machine learning platform and model orchestration",
    version="1.0.0"
)

# Configure CORS for Frontend (Person C)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(datasets.router)
app.include_router(pipelines.router)
app.include_router(runs.router)
app.include_router(predict.router)
app.include_router(export.router)



@app.get("/")
def home():
    return {
        "message": "Interactive Model Studio backend is running!",
        "version": "1.0.0"
    }


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy"
    }