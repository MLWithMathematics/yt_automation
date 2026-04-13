"""
graph/pipeline.py — LangGraph StateGraph pipeline with retry edges and resume support.

FIXES:
  - run_pipeline() no longer ignores passed run_id (was creating duplicate runs)
  - resume_pipeline() works even on RUNNING state (handles crash recovery)
  - per-run config (video_length, voice) passed via state
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

import config
import database.db as db
from database.models import RunStatus
from agents.trend_agent import TrendAgent
from agents.keyword_agent import KeywordAgent
from agents.script_agent import ScriptAgent
from agents.voiceover_agent import VoiceoverAgent
from agents.visual_agent import VisualAgent
from agents.video_agent import VideoAgent
from agents.thumbnail_agent import ThumbnailAgent
from agents.metadata_agent import MetadataAgent

logger = logging.getLogger(__name__)

# ─── SSE Event Broadcaster ───────────────────────────────────────────────────
_sse_broadcaster = None

def set_sse_broadcaster(broadcaster):
    global _sse_broadcaster
    _sse_broadcaster = broadcaster

def emit_event(event_type: str, data: dict):
    if _sse_broadcaster:
        try:
            _sse_broadcaster(event_type, data)
        except Exception as e:
            logger.warning(f"SSE emit failed: {e}")

# ─── Pipeline State ───────────────────────────────────────────────────────────
class PipelineState(TypedDict):
    run_id: str
    niche: str
    config: dict          # per-run config overrides
    trend_data: dict
    keyword_data: dict
    script_data: dict
    voiceover_data: dict
    visual_data: dict
    video_data: dict
    thumbnail_data: dict
    metadata: dict
    current_agent: str
    errors: List[str]
    completed_agents: List[str]


AGENT_SEQUENCE = [
    "TrendAgent",
    "KeywordAgent",
    "ScriptAgent",
    "VoiceoverAgent",
    "VisualAgent",
    "VideoAgent",
    "ThumbnailAgent",
    "MetadataAgent",
]

AGENT_MAP = {
    "TrendAgent": TrendAgent,
    "KeywordAgent": KeywordAgent,
    "ScriptAgent": ScriptAgent,
    "VoiceoverAgent": VoiceoverAgent,
    "VisualAgent": VisualAgent,
    "VideoAgent": VideoAgent,
    "ThumbnailAgent": ThumbnailAgent,
    "MetadataAgent": MetadataAgent,
}


def _run_agent_with_retry(agent_class, state: PipelineState) -> dict:
    """Run agent with exponential backoff retry on failure."""
    run_id = state["run_id"]
    agent_name = agent_class.__name__
    max_retries = config.MAX_AGENT_RETRIES
    backoffs = config.RETRY_BACKOFF_SECONDS

    agent = agent_class()
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            emit_event("agent_started", {
                "run_id": run_id,
                "agent_name": agent_name,
                "timestamp": time.time(),
                "attempt": attempt,
            })
            t0 = time.time()
            result = agent.run(dict(state))
            elapsed = round(time.time() - t0, 2)

            emit_event("agent_completed", {
                "run_id": run_id,
                "agent_name": agent_name,
                "duration": elapsed,
                "output_summary": str(list(result.keys())[:3]),
            })
            return result

        except Exception as e:
            last_error = str(e)
            retry_count = db.get_task_retry_count(run_id, agent_name)

            emit_event("agent_failed", {
                "run_id": run_id,
                "agent_name": agent_name,
                "error": last_error[:300],
                "retry_count": retry_count,
            })

            if attempt < max_retries:
                wait = backoffs[attempt] if attempt < len(backoffs) else backoffs[-1]
                logger.warning(
                    f"[{agent_name}] Attempt {attempt+1} failed: {e}. Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"[{agent_name}] All {max_retries} retries exhausted. Error: {e}"
                )
                raise RuntimeError(
                    f"{agent_name} failed after {max_retries} retries: {last_error}"
                )

    raise RuntimeError(f"{agent_name} failed: {last_error}")


def make_agent_node(agent_name: str):
    """Factory that creates a LangGraph node function for an agent."""
    agent_class = AGENT_MAP[agent_name]

    def node_fn(state: PipelineState) -> PipelineState:
        run_id = state["run_id"]
        try:
            db.update_run_status(run_id, RunStatus.RUNNING)
            result = _run_agent_with_retry(agent_class, state)

            new_state = dict(state)
            new_state.update(result)
            completed = list(new_state.get("completed_agents", []))
            if agent_name not in completed:
                completed.append(agent_name)
            new_state["completed_agents"] = completed
            new_state["current_agent"] = agent_name
            return new_state

        except Exception as e:
            new_state = dict(state)
            errors = list(new_state.get("errors", []))
            errors.append(f"{agent_name}: {str(e)}")
            new_state["errors"] = errors
            new_state["current_agent"] = agent_name
            db.update_run_status(run_id, RunStatus.FAILED, str(e))
            raise

    node_fn.__name__ = f"node_{agent_name}"
    return node_fn


def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)
    for agent_name in AGENT_SEQUENCE:
        graph.add_node(agent_name, make_agent_node(agent_name))
    for i, agent_name in enumerate(AGENT_SEQUENCE):
        if i == 0:
            graph.set_entry_point(agent_name)
        if i < len(AGENT_SEQUENCE) - 1:
            graph.add_edge(agent_name, AGENT_SEQUENCE[i + 1])
        else:
            graph.add_edge(agent_name, END)
    return graph.compile()


# ─── Pipeline Runner ──────────────────────────────────────────────────────────

def _build_initial_state(run_id: str, niche: str, run_config: dict = None) -> PipelineState:
    """Build a fresh pipeline state."""
    cfg = {
        "video_length": config.VIDEO_LENGTH,
        "voice": config.VOICE,
        "groq_model": config.GROQ_MODEL,
    }
    if run_config:
        cfg.update(run_config)

    return {
        "run_id": run_id,
        "niche": niche,
        "config": cfg,
        "trend_data": {},
        "keyword_data": {},
        "script_data": {},
        "voiceover_data": {},
        "visual_data": {},
        "video_data": {},
        "thumbnail_data": {},
        "metadata": {},
        "current_agent": "",
        "errors": [],
        "completed_agents": [],
    }


def run_pipeline(niche: str = None, run_id: str = None, run_config: dict = None) -> dict:
    """
    Start a fresh pipeline run.

    FIX: If run_id is provided (created by the API endpoint), use it directly.
    Do NOT create a second run record. This was the source of the double-run bug.
    """
    if niche is None:
        niche = config.NICHE

    # ── Only create DB record if caller didn't already create one ──────────
    if run_id is None:
        run_data = db.create_run(niche)
        run_id = run_data["id"]
    # else: caller already created the run (server.py → start_run endpoint)

    logger.info(f"Starting pipeline run {run_id} for niche: {niche}")
    db.update_run_status(run_id, RunStatus.RUNNING)

    for agent_name in AGENT_SEQUENCE:
        db.upsert_agent_task(run_id, agent_name)

    initial_state = _build_initial_state(run_id, niche, run_config)
    pipeline = build_pipeline()

    try:
        final_state = pipeline.invoke(initial_state)
        if not final_state.get("errors"):
            db.update_run_status(run_id, RunStatus.COMPLETED)
            emit_event("pipeline_done", {
                "run_id": run_id,
                "video_path": final_state.get("video_data", {}).get("video_path", ""),
                "thumbnail_path": final_state.get("thumbnail_data", {}).get("thumbnail_path", ""),
                "metadata": final_state.get("metadata", {}),
            })
            logger.info(f"Pipeline {run_id} completed successfully")
        else:
            db.update_run_status(run_id, RunStatus.FAILED, str(final_state["errors"]))
        return dict(final_state)
    except Exception as e:
        logger.error(f"Pipeline {run_id} crashed: {e}")
        db.update_run_status(run_id, RunStatus.FAILED, str(e))
        raise


OUTPUT_KEY_MAP = {
    "TrendAgent": "trend_data",
    "KeywordAgent": "keyword_data",
    "ScriptAgent": "script_data",
    "VoiceoverAgent": "voiceover_data",
    "VisualAgent": "visual_data",
    "VideoAgent": "video_data",
    "ThumbnailAgent": "thumbnail_data",
    "MetadataAgent": "metadata",
}


def _rebuild_state_from_db(run_id: str, niche: str, completed_agents: list) -> PipelineState:
    """Reconstruct PipelineState from saved DB outputs."""
    state = _build_initial_state(run_id, niche)
    state["completed_agents"] = list(completed_agents)

    for agent_name in completed_agents:
        output = db.get_task_output(run_id, agent_name)
        if output:
            key = OUTPUT_KEY_MAP.get(agent_name)
            if key:
                state[key] = output.get(key, output)
            logger.info(f"  Restored {agent_name} output from DB")

    return state


def resume_pipeline(run_id: str) -> dict:
    """
    Resume a failed/paused/crashed pipeline run from the last completed agent.

    FIX: Also works when run is stuck in RUNNING state (process crash scenario).
    """
    run_data = db.get_run(run_id)
    if not run_data:
        raise ValueError(f"Run {run_id} not found")

    niche = run_data["niche"]
    completed_agents = db.get_completed_agents(run_id)
    logger.info(f"Resuming run {run_id}. Completed agents: {completed_agents}")

    state = _rebuild_state_from_db(run_id, niche, completed_agents)

    # Find first pending agent
    first_pending = None
    for agent_name in AGENT_SEQUENCE:
        if agent_name not in completed_agents:
            first_pending = agent_name
            break

    if first_pending is None:
        logger.info(f"All agents already completed for run {run_id}")
        db.update_run_status(run_id, RunStatus.COMPLETED)
        return dict(state)

    logger.info(f"Resuming from agent: {first_pending}")
    db.update_run_status(run_id, RunStatus.RUNNING)

    # Reset the first pending agent's task status so it re-runs cleanly
    db.reset_agent_task(run_id, first_pending)

    # Build partial pipeline
    remaining = AGENT_SEQUENCE[AGENT_SEQUENCE.index(first_pending):]
    graph = StateGraph(PipelineState)
    for agent_name in remaining:
        graph.add_node(agent_name, make_agent_node(agent_name))
    graph.set_entry_point(remaining[0])
    for i in range(len(remaining) - 1):
        graph.add_edge(remaining[i], remaining[i + 1])
    graph.add_edge(remaining[-1], END)

    partial_pipeline = graph.compile()

    try:
        final_state = partial_pipeline.invoke(state)
        if not final_state.get("errors"):
            db.update_run_status(run_id, RunStatus.COMPLETED)
            emit_event("pipeline_done", {
                "run_id": run_id,
                "video_path": final_state.get("video_data", {}).get("video_path", ""),
                "thumbnail_path": final_state.get("thumbnail_data", {}).get("thumbnail_path", ""),
                "metadata": final_state.get("metadata", {}),
            })
        else:
            db.update_run_status(run_id, RunStatus.FAILED, str(final_state["errors"]))
        return dict(final_state)
    except Exception as e:
        logger.error(f"Resume pipeline {run_id} failed: {e}")
        db.update_run_status(run_id, RunStatus.FAILED, str(e))
        raise


def retry_single_agent(run_id: str, agent_name: str) -> dict:
    """Retry a specific agent without re-running the whole pipeline."""
    if agent_name not in AGENT_MAP:
        raise ValueError(f"Unknown agent: {agent_name}")

    run_data = db.get_run(run_id)
    if not run_data:
        raise ValueError(f"Run {run_id} not found")

    agent_idx = AGENT_SEQUENCE.index(agent_name)
    completed_agents = db.get_completed_agents(run_id)
    completed_before = [a for a in AGENT_SEQUENCE[:agent_idx] if a in completed_agents]

    state = _rebuild_state_from_db(run_id, run_data["niche"], completed_before)

    # Reset the target agent task
    db.reset_agent_task(run_id, agent_name)
    db.update_run_status(run_id, RunStatus.RUNNING)

    agent_class = AGENT_MAP[agent_name]
    result = _run_agent_with_retry(agent_class, state)

    # If successful, continue the pipeline from the next agent
    next_idx = agent_idx + 1
    if next_idx < len(AGENT_SEQUENCE):
        # Mark this agent completed in state
        completed_with_current = completed_before + [agent_name]
        new_state = _rebuild_state_from_db(run_id, run_data["niche"], completed_before)
        new_state.update(result)
        new_state["completed_agents"] = completed_with_current

        remaining = AGENT_SEQUENCE[next_idx:]
        graph = StateGraph(PipelineState)
        for aname in remaining:
            graph.add_node(aname, make_agent_node(aname))
        graph.set_entry_point(remaining[0])
        for i in range(len(remaining) - 1):
            graph.add_edge(remaining[i], remaining[i + 1])
        graph.add_edge(remaining[-1], END)
        partial = graph.compile()

        try:
            final_state = partial.invoke(new_state)
            if not final_state.get("errors"):
                db.update_run_status(run_id, RunStatus.COMPLETED)
                emit_event("pipeline_done", {
                    "run_id": run_id,
                    "video_path": final_state.get("video_data", {}).get("video_path", ""),
                    "thumbnail_path": final_state.get("thumbnail_data", {}).get("thumbnail_path", ""),
                    "metadata": final_state.get("metadata", {}),
                })
            else:
                db.update_run_status(run_id, RunStatus.FAILED, str(final_state["errors"]))
            return dict(final_state)
        except Exception as e:
            db.update_run_status(run_id, RunStatus.FAILED, str(e))
            raise
    else:
        db.update_run_status(run_id, RunStatus.COMPLETED)
        return result
