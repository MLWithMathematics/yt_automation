"""
dashboard/server.py — FastAPI backend with SSE for real-time pipeline updates.

FIXES:
  - start_run now uses JSON body (Pydantic model) — avoids double-run
  - run_pipeline(run_id=...) is called with the pre-created run_id so the
    pipeline does NOT create a second run record
  - resume works on FAILED, PAUSED, and RUNNING (crash recovery)
  - /api/runs/start accepts video_length + voice config per-run
  - PATCH /api/config to update niche/length/voice at runtime
"""
import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import config
import database.db as db
from database.models import RunStatus
from graph.pipeline import (
    run_pipeline,
    resume_pipeline,
    retry_single_agent,
    set_sse_broadcaster,
    AGENT_SEQUENCE,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Agent Pipeline Dashboard")

# ─── SSE Event Queues ─────────────────────────────────────────────────────────
_event_queues: Dict[str, asyncio.Queue] = {}
_global_queue: asyncio.Queue = asyncio.Queue()


def _get_or_create_queue(run_id: str) -> asyncio.Queue:
    if run_id not in _event_queues:
        _event_queues[run_id] = asyncio.Queue()
    return _event_queues[run_id]


def broadcast_event(event_type: str, data: dict):
    """Thread-safe SSE broadcast from pipeline thread → async SSE clients."""
    try:
        payload = json.dumps({"type": event_type, "data": data, "ts": time.time()})
        run_id = data.get("run_id", "global")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(
                    lambda p=payload, r=run_id: asyncio.ensure_future(_put_event(r, p))
                )
        except RuntimeError:
            pass
    except Exception as e:
        logger.warning(f"Broadcast failed: {e}")


async def _put_event(run_id: str, payload: str):
    q = _get_or_create_queue(run_id)
    await q.put(payload)
    await _global_queue.put(payload)


set_sse_broadcaster(broadcast_event)

# ─── Static / Output files ────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
OUTPUT_DIR = Path(config.OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard loading…</h1>")


# ─── SSE Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/sse/{run_id}")
async def sse_run_events(run_id: str):
    queue = _get_or_create_queue(run_id)

    async def event_generator():
        yield {"event": "connected", "data": json.dumps({"run_id": run_id})}
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=30)
                yield {"event": "message", "data": payload}
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "{}"}

    return EventSourceResponse(event_generator())


@app.get("/api/sse/global/stream")
async def sse_global():
    async def generator():
        yield {"event": "connected", "data": '{"stream":"global"}'}
        while True:
            try:
                payload = await asyncio.wait_for(_global_queue.get(), timeout=30)
                yield {"event": "message", "data": payload}
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "{}"}

    return EventSourceResponse(generator())


# ─── Pipeline Control ─────────────────────────────────────────────────────────

class StartRunRequest(BaseModel):
    niche: Optional[str] = None
    video_length: Optional[str] = None   # "short" | "medium" | "long"
    voice: Optional[str] = None          # edge-tts voice name


@app.post("/api/runs/start")
async def start_run(request: StartRunRequest, background_tasks: BackgroundTasks):
    """
    Start a new pipeline run.

    FIX: The run record is created HERE (once), then the run_id is passed to
    run_pipeline() so it does NOT create a second record. The previous bug was
    that run_pipeline() always called db.create_run() internally, resulting in
    two DB rows per button click.
    """
    target_niche = request.niche or config.NICHE

    # Build per-run config overrides
    run_config = {}
    if request.video_length:
        run_config["video_length"] = request.video_length
    if request.voice:
        run_config["voice"] = request.voice

    # ── Create the single DB record here ──────────────────────────────────
    run_data = db.create_run(target_niche)
    run_id = run_data["id"]
    db.update_run_status(run_id, RunStatus.RUNNING)

    def _run():
        try:
            # Pass run_id so pipeline skips create_run()
            run_pipeline(niche=target_niche, run_id=run_id, run_config=run_config or None)
        except Exception as e:
            logger.error(f"Pipeline {run_id} background thread error: {e}")

    background_tasks.add_task(_run)
    return {"run_id": run_id, "niche": target_niche, "status": "RUNNING"}


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str, background_tasks: BackgroundTasks):
    """
    Resume a failed/paused/crashed pipeline.
    Works on FAILED, PAUSED, and (after crash recovery) RUNNING status.
    """
    run_data = db.get_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")

    def _resume():
        try:
            resume_pipeline(run_id)
        except Exception as e:
            logger.error(f"Resume {run_id} failed: {e}")

    background_tasks.add_task(_resume)
    return {"run_id": run_id, "status": "RESUMING"}


@app.post("/api/runs/{run_id}/agents/{agent_name}/retry")
async def retry_agent(run_id: str, agent_name: str, background_tasks: BackgroundTasks):
    """Retry a single agent and continue the pipeline from that point."""
    run_data = db.get_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")

    def _retry():
        try:
            retry_single_agent(run_id, agent_name)
        except Exception as e:
            logger.error(f"Agent retry {agent_name} in {run_id} failed: {e}")

    background_tasks.add_task(_retry)
    return {"run_id": run_id, "agent_name": agent_name, "status": "RETRYING"}


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    db.delete_run(run_id)
    return {"deleted": run_id}


# ─── Config Update ────────────────────────────────────────────────────────────

class ConfigUpdate(BaseModel):
    niche: Optional[str] = None
    video_length: Optional[str] = None
    voice: Optional[str] = None
    scheduler_interval_hours: Optional[int] = None


@app.patch("/api/config")
async def update_config(req: ConfigUpdate):
    """
    Update runtime config values in memory.
    These persist until server restart; for permanent changes edit config.py.
    """
    if req.niche:
        config.NICHE = req.niche
    if req.video_length and req.video_length in ("short", "medium", "long"):
        config.VIDEO_LENGTH = req.video_length
    if req.voice:
        config.VOICE = req.voice
    if req.scheduler_interval_hours and req.scheduler_interval_hours > 0:
        config.SCHEDULER_INTERVAL_HOURS = req.scheduler_interval_hours
    return {
        "niche": config.NICHE,
        "video_length": config.VIDEO_LENGTH,
        "voice": config.VOICE,
        "scheduler_interval_hours": config.SCHEDULER_INTERVAL_HOURS,
    }


# ─── Query Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/runs")
async def list_runs(limit: int = 10, offset: int = 0):
    runs = db.list_runs(limit=limit, offset=offset)
    total = db.count_runs()
    return {"runs": runs, "total": total, "limit": limit, "offset": offset}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    run_data = db.get_run(run_id)
    if not run_data:
        raise HTTPException(status_code=404, detail="Run not found")
    tasks = db.get_all_tasks_for_run(run_id)
    logs = db.get_logs_for_run(run_id, limit=100)
    return {"run": run_data, "tasks": tasks, "recent_logs": logs}


@app.get("/api/runs/{run_id}/logs")
async def get_run_logs(run_id: str, agent_name: Optional[str] = None, limit: int = 500):
    logs = db.get_logs_for_run(run_id, agent_name=agent_name, limit=limit)
    return {"logs": logs}


@app.get("/api/runs/{run_id}/output")
async def get_run_output(run_id: str):
    run_dir = OUTPUT_DIR / f"run_{run_id}"
    result: Dict[str, Any] = {
        "video": None, "thumbnail": None, "script": None, "metadata": None
    }
    if run_dir.exists():
        if (run_dir / "final_video.mp4").exists():
            result["video"] = f"/output/run_{run_id}/final_video.mp4"
        if (run_dir / "thumbnail.jpg").exists():
            result["thumbnail"] = f"/output/run_{run_id}/thumbnail.jpg"

    meta = db.get_task_output(run_id, "MetadataAgent")
    if meta:
        result["metadata"] = meta.get("metadata", meta)

    script = db.get_task_output(run_id, "ScriptAgent")
    if script:
        result["script"] = script.get("script_data", script)

    trend = db.get_task_output(run_id, "TrendAgent")
    if trend:
        td = trend.get("trend_data", trend)
        result["topic"] = td.get("topic", "")

    return result


@app.get("/api/config")
async def get_config():
    return {
        "niche": config.NICHE,
        "video_length": config.VIDEO_LENGTH,
        "voice": config.VOICE,
        "groq_model": config.GROQ_MODEL,
        "scheduler_interval_hours": config.SCHEDULER_INTERVAL_HOURS,
        "agent_sequence": AGENT_SEQUENCE,
        "video_lengths": ["short", "medium", "long"],
        "voices": [
            "en-US-GuyNeural",
            "en-US-JennyNeural",
            "en-US-AriaNeural",
            "en-US-DavisNeural",
            "en-GB-RyanNeural",
            "en-AU-WilliamNeural",
        ],
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
