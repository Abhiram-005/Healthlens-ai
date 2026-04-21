"""
HealthLens AI — FastAPI Server
Main entry point. Handles file uploads, pipeline orchestration, and chat.
"""

import os
import sys
import uuid
import shutil
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("healthlens")

# Validate API key early
if not os.getenv("GROQ_API_KEY"):
    logger.error("=" * 60)
    logger.error("GROQ_API_KEY is not set!")
    logger.error("Get a FREE key at: https://console.groq.com")
    logger.error("Then copy .env.example to .env and add your key.")
    logger.error("=" * 60)

# Import after env loaded
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import HealthLensPipeline
from database import Database

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
MAX_FILE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

db = Database()
pipeline = HealthLensPipeline()


# ─── App Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("HealthLens AI starting...")
    logger.info(f"Model (accurate): {os.getenv('GROQ_MODEL_ACCURATE', 'llama-3.3-70b-versatile')}")
    logger.info(f"Model (fast):     {os.getenv('GROQ_MODEL_FAST', 'llama-3.1-8b-instant')}")
    yield
    logger.info("HealthLens AI shutting down.")


app = FastAPI(
    title="HealthLens AI",
    description="AI-powered medical document analysis — Cognizant Technoverse 2026",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ─── Request / Response Models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    file_id: str
    question: str

class DeleteResponse(BaseModel):
    success: bool
    message: str


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>HealthLens AI</h1><p>Frontend not found. Make sure frontend/index.html exists.</p>")


@app.get("/api/health")
async def health_check():
    groq_key = os.getenv("GROQ_API_KEY", "")
    return {
        "status": "running",
        "groq_configured": bool(groq_key and groq_key != "your_groq_api_key_here"),
        "model_accurate": os.getenv("GROQ_MODEL_ACCURATE", "llama-3.3-70b-versatile"),
        "model_fast": os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant"),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Upload and analyze a medical document.
    Supports: PDF, JPG, PNG, TIFF, BMP, WEBP
    """
    # Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_MB} MB."
        )

    # Save file
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}{ext}"
    file_path.write_bytes(content)

    logger.info(f"Received file: {file.filename} ({size_mb:.2f} MB) → {file_id}")

    # Find previous report for the same patient (for comparison)
    # We'll attempt this after extraction via the DB
    try:
        result = await pipeline.run(
            file_path=str(file_path),
            filename=file.filename or f"document{ext}",
            file_id=file_id,
        )

        # Try to find previous report for comparison
        extraction = result.get("extraction") or {}
        patient = extraction.get("patient") or {}
        patient_name = patient.get("name")
        doc_type = extraction.get("document_type", "other")

        if patient_name:
            previous = db.get_previous_report(patient_name, doc_type, file_id)
            if previous:
                logger.info(f"Found previous report for {patient_name} — running comparison")
                # Re-run pipeline with previous report for comparison
                result = await pipeline.run(
                    file_path=str(file_path),
                    filename=file.filename or f"document{ext}",
                    file_id=file_id,
                    previous_report=previous,
                )

        # Save to database
        db.save_report(file_id, result)

        return JSONResponse(content={
            "success": True,
            "file_id": file_id,
            "result": result,
        })

    except Exception as e:
        logger.error(f"Pipeline error for {file_id}: {e}", exc_info=True)
        # Clean up file
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/chat")
async def chat_with_report(request: ChatRequest):
    """Ask a follow-up question about an already-analyzed report."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    report = db.get_report(request.file_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{request.file_id}' not found.")

    try:
        response = await pipeline.chat(report, request.question.strip())
        return {"success": True, "response": response}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/api/history")
async def get_history():
    """Get list of all analyzed reports."""
    reports = db.get_all_reports()
    return {"reports": reports, "count": len(reports)}


@app.get("/api/report/{file_id}")
async def get_report(file_id: str):
    """Get full report data for a specific file_id."""
    report = db.get_report(file_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.delete("/api/report/{file_id}")
async def delete_report(file_id: str):
    """Delete a report and its uploaded file."""
    # Delete file
    for ext in ALLOWED_EXTENSIONS:
        file_path = UPLOAD_DIR / f"{file_id}{ext}"
        if file_path.exists():
            file_path.unlink()
            break

    db.delete_report(file_id)
    return DeleteResponse(success=True, message=f"Report {file_id} deleted.")


# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
