from __future__ import annotations

import asyncio
import atexit
import io
import json
import multiprocessing
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

from app.agent.workflow import run_analysis
from app.services.profiler import profile_dataset

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR.parent / ".env")
SAMPLE_DIR = APP_DIR.parent / "samples"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "autonomous-data-analyst"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_CELLS = 2_000_000
ANALYSIS_TIMEOUT_SECONDS = 90
DB_PATH = APP_DIR.parent / "analyst.sqlite3"
datasets: dict[str, dict[str, Any]] = {}
analyses: dict[str, dict[str, Any]] = {}
lock = threading.RLock()

app = FastAPI(title="Autonomous Data Analyst Agent", version="1.0.0",
              description="Evidence-backed dataset analysis with registered tools and a LangGraph workflow.")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class StartAnalysis(BaseModel):
    dataset_id: str
    objective: str = Field(default="", max_length=1000)
    provider: Literal["local", "openai", "gemini", "auto"] = "auto"
    api_key: SecretStr | None = None


def _init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, objective TEXT NOT NULL,
            status TEXT NOT NULL, created_at REAL NOT NULL
        )""")
_init_db()


def _read_frame(data: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".csv":
            try:
                df = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
        elif suffix == ".xlsx":
            df = pd.read_excel(io.BytesIO(data), engine="openpyxl")
        else:
            raise HTTPException(415, "Upload a CSV or XLSX file.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Could not read this dataset: {exc}") from exc
    if df.empty:
        raise HTTPException(422, "The dataset has no rows.")
    if df.shape[1] == 0:
        raise HTTPException(422, "The dataset has no columns.")
    if df.shape[0] * df.shape[1] > MAX_CELLS:
        raise HTTPException(413, f"Dataset exceeds the {MAX_CELLS:,}-cell processing limit.")
    df.columns = [str(c).strip() or f"column_{i+1}" for i, c in enumerate(df.columns)]
    if len(set(df.columns)) != len(df.columns):
        raise HTTPException(422, "Column names must be unique.")
    return df


def _add_dataset(df: pd.DataFrame, name: str) -> dict[str, Any]:
    dataset_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{dataset_id}.csv"
    df.to_csv(path, index=False)
    record = {"id": dataset_id, "name": Path(name).name[:150], "rows": len(df), "columns": len(df.columns),
              "profile": profile_dataset(df).as_dict(), "path": str(path), "created_at": time.time()}
    with lock:
        datasets[dataset_id] = {**record, "dataframe": df}
    return {k: v for k, v in record.items() if k != "path"}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "langgraph", "llm_mode": "openai" if os.getenv("OPENAI_API_KEY") else "local"}


@app.get("/api/datasets/examples")
def example_datasets() -> list[dict[str, Any]]:
    return [
        {"id": "sample-sales", "name": "Retail sales", "description": "24 months of monthly product and regional sales, with a late-period decline."},
        {"id": "sample-churn", "name": "Customer churn", "description": "Subscription tenure, plan, usage and churn outcomes."},
        {"id": "sample-ecommerce", "name": "E-commerce orders", "description": "Order-level basket, channel, region and fulfillment details."},
    ]


@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    name = file.filename or "dataset.csv"
    if Path(name).suffix.lower() not in {".csv", ".xlsx"}:
        raise HTTPException(415, "Upload a CSV or XLSX file.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File exceeds the 20 MB upload limit.")
    df = _read_frame(data, name)
    return _add_dataset(df, name)


@app.post("/api/datasets/examples/{example_id}")
def load_example(example_id: str) -> dict[str, Any]:
    allowed = {"sample-sales": "sales.csv", "sample-churn": "customer_churn.csv", "sample-ecommerce": "ecommerce.csv"}
    if example_id not in allowed:
        raise HTTPException(404, "Example dataset not found.")
    path = SAMPLE_DIR / allowed[example_id]
    df = pd.read_csv(path)
    return _add_dataset(df, path.name)


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict[str, Any]:
    item = datasets.get(dataset_id)
    if not item:
        raise HTTPException(404, "Dataset not found. Upload it again to start a new session.")
    return {k: item[k] for k in ("id", "name", "rows", "columns", "created_at")}


@app.get("/api/datasets/{dataset_id}/profile")
def get_profile(dataset_id: str) -> dict[str, Any]:
    item = datasets.get(dataset_id)
    if not item:
        raise HTTPException(404, "Dataset not found.")
    return item["profile"]


@app.post("/api/analysis/start", status_code=202)
def start_analysis(request: StartAnalysis, background_tasks: BackgroundTasks) -> dict[str, str]:
    item = datasets.get(request.dataset_id)
    if not item:
        raise HTTPException(404, "Dataset not found.")
    api_key = request.api_key.get_secret_value() if request.api_key else None
    provider = request.provider
    if provider == "openai" and not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if provider == "auto":
        provider = "openai" if os.getenv("OPENAI_API_KEY") else "local"
        api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else None
    if provider in {"openai", "gemini"} and not api_key:
        raise HTTPException(422, "Enter an API key for the selected provider.")
    analysis_id = str(uuid.uuid4())
    record = {"id": analysis_id, "dataset_id": request.dataset_id, "objective": request.objective,
              "status": "queued", "trace": [{"node": "system", "status": "complete", "message": "Analysis queued.", "details": {}}],
              "created_at": time.time(), "report": None, "error": None}
    with lock:
        analyses[analysis_id] = record
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO analyses VALUES (?, ?, ?, ?, ?)",
                         (analysis_id, request.dataset_id, request.objective, "queued", record["created_at"]))
    background_tasks.add_task(_execute_analysis, analysis_id, provider, api_key)
    return {"analysis_id": analysis_id, "status": "queued"}


def _execute_analysis(analysis_id: str, provider: str = "local", api_key: str | None = None) -> None:
    with lock:
        record = analyses[analysis_id]
        record["status"] = "running"
        record["trace"].append({"node": "system", "status": "complete", "message": "Profiling the dataset and preparing the agent plan.", "details": {}})
        dataset = datasets[record["dataset_id"]]
    try:
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        process = context.Process(
            target=_analysis_process,
            args=(dataset["path"], record["objective"], provider, api_key, result_queue),
            daemon=True,
        )
        process.start()
        process.join(ANALYSIS_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise TimeoutError(f"Analysis exceeded the {ANALYSIS_TIMEOUT_SECONDS}-second limit.")
        try:
            result = result_queue.get(timeout=2)
        except Exception as exc:
            raise RuntimeError("Analysis worker exited without returning a result.") from exc
        finally:
            result_queue.close()
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Analysis worker failed."))
        with lock:
            record["report"] = result["report"]
            record["trace"] = result["trace"]
            record["status"] = "complete"
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("UPDATE analyses SET status='complete' WHERE id=?", (analysis_id,))
    except Exception as exc:
        with lock:
            record["status"] = "failed"
            message = str(exc) if provider == "local" else "The selected model provider could not create a plan. Check the API key, model access, and provider availability."
            record["error"] = message
            record["trace"].append({"node": "system", "status": "error", "message": f"Analysis failed: {message}", "details": {}})
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("UPDATE analyses SET status='failed' WHERE id=?", (analysis_id,))


def _analysis_process(dataset_path: str, objective: str, provider: str, api_key: str | None, result_queue: Any) -> None:
    """Child process boundary lets the API terminate an analysis at its deadline."""
    try:
        dataframe = pd.read_csv(dataset_path)
        result = run_analysis(dataframe, objective, provider, api_key)
        result_queue.put({"ok": True, "report": result["report"], "trace": result["trace"]})
    except Exception:
        # Provider SDK exceptions can include request headers; never return them to clients.
        error = "Analysis worker failed." if provider == "local" else "The selected model provider could not create a plan. Check the API key, model access, and provider availability."
        result_queue.put({"ok": False, "error": error})


@app.get("/api/analysis/{analysis_id}")
def get_analysis(analysis_id: str) -> dict[str, Any]:
    record = analyses.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found.")
    return {"id": record["id"], "dataset_id": record["dataset_id"], "status": record["status"],
            "created_at": record["created_at"], "error": record["error"], "report": record["report"]}


@app.get("/api/analysis/{analysis_id}/trace")
def get_trace(analysis_id: str) -> dict[str, Any]:
    record = analyses.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found.")
    return {"analysis_id": analysis_id, "status": record["status"], "events": record["trace"]}


@app.get("/api/analysis/{analysis_id}/events")
async def analysis_events(analysis_id: str):
    if analysis_id not in analyses:
        raise HTTPException(404, "Analysis not found.")

    async def stream():
        cursor = 0
        while True:
            record = analyses.get(analysis_id)
            if not record:
                break
            events = record["trace"][cursor:]
            for event in events:
                cursor += 1
                yield f"data: {json.dumps(event)}\n\n"
            if record["status"] in {"complete", "failed"} and cursor >= len(record["trace"]):
                yield f"event: done\ndata: {json.dumps({'status': record['status']})}\n\n"
                break
            await asyncio.sleep(0.4)
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/analysis/{analysis_id}/report")
def get_report(analysis_id: str, format: str = "json"):
    record = analyses.get(analysis_id)
    if not record:
        raise HTTPException(404, "Analysis not found.")
    if record["status"] != "complete":
        raise HTTPException(409, "The report is not ready yet.")
    report = record["report"]
    if format == "json":
        return report
    if format == "csv":
        rows = [{"title": f["title"], "detail": f["detail"], "kind": f["kind"],
                 "evidence": json.dumps(f.get("evidence", {}), ensure_ascii=False)} for f in report["findings"]]
        return StreamingResponse(iter([pd.DataFrame(rows, columns=["title", "detail", "kind", "evidence"]).to_csv(index=False)]),
                                 media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="analysis-{analysis_id[:8]}-findings.csv"'})
    if format == "pdf":
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except ImportError as exc:
            raise HTTPException(501, "PDF export requires reportlab. Install backend requirements.") from exc
        output = io.BytesIO()
        pdf = canvas.Canvas(output, pagesize=letter)
        width, height = letter
        y = height - 48
        pdf.setTitle("Autonomous Data Analyst Report")
        for heading, body in [("Autonomous Data Analyst", report["executive_summary"]),
                              ("Dataset overview", f'{report["dataset_overview"]["rows"]:,} rows · {report["dataset_overview"]["columns"]} columns'),
                              ("Key findings", "\n".join(f'• {f["title"]}: {f["detail"]}' for f in report["findings"]) or "No standout findings."),
                              ("Recommendations", "\n".join("• " + x for x in report["recommendations"])),
                              ("Limitations", "\n".join("• " + x for x in report["limitations"]))]:
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(48, y, heading[:100]); y -= 22
            pdf.setFont("Helvetica", 10)
            for paragraph in body.splitlines():
                for line in _wrap(paragraph, 92):
                    if y < 48:
                        pdf.showPage(); y = height - 48
                    pdf.drawString(48, y, line[:130]); y -= 14
            y -= 12
        pdf.save(); output.seek(0)
        return StreamingResponse(output, media_type="application/pdf",
                                 headers={"Content-Disposition": f'attachment; filename="analysis-{analysis_id[:8]}-report.pdf"'})
    raise HTTPException(400, "format must be json, csv, or pdf.")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > width and current:
            lines.append(current); current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _cleanup_uploads() -> None:
    for item in datasets.values():
        try:
            Path(item["path"]).unlink(missing_ok=True)
        except OSError:
            pass


atexit.register(_cleanup_uploads)
