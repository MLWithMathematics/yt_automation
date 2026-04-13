"""database/db.py — Session management and CRUD helpers."""
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config
from database.models import (
    Base, PipelineRun, AgentTask, AgentLog,
    RunStatus, TaskStatus, LogLevel,
)

logger = logging.getLogger(__name__)
engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized.")


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─── Crash Recovery ───────────────────────────────────────────────────────────

def recover_stuck_runs():
    """
    On server startup, mark any RUNNING runs as FAILED.

    If the process was killed (OOM, SIGKILL, crash) while a run was active,
    the run stays in RUNNING state permanently. This function fixes that so
    the dashboard shows Resume buttons instead of frozen RUNNING badges.
    Also marks their RUNNING tasks as FAILED.
    """
    with get_session() as s:
        stuck_runs = s.query(PipelineRun).filter_by(status=RunStatus.RUNNING).all()
        for run in stuck_runs:
            run.status = RunStatus.FAILED
            run.error_message = "Server restarted while pipeline was running. Use Resume to continue."
            run.updated_at = datetime.utcnow()
            logger.warning(f"Recovery: marked run {run.id} as FAILED (was RUNNING on restart)")

        # Also fix RUNNING tasks
        stuck_tasks = s.query(AgentTask).filter_by(status=TaskStatus.RUNNING).all()
        for task in stuck_tasks:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.error_message = "Server restarted during execution."
            logger.warning(
                f"Recovery: marked task {task.agent_name} in run {task.run_id} as FAILED"
            )

    if stuck_runs:
        logger.info(
            f"Crash recovery: fixed {len(stuck_runs)} stuck run(s). "
            "Use the dashboard Resume button to continue."
        )


# ─── Run CRUD ─────────────────────────────────────────────────────────────────

def create_run(niche: str) -> dict:
    with get_session() as s:
        run = PipelineRun(niche=niche, status=RunStatus.PENDING)
        s.add(run)
        s.flush()
        d = run.to_dict()
    return d


def get_run(run_id: str) -> Optional[dict]:
    with get_session() as s:
        run = s.query(PipelineRun).filter_by(id=run_id).first()
        return run.to_dict() if run else None


def list_runs(limit: int = 50, offset: int = 0) -> List[dict]:
    with get_session() as s:
        runs = (
            s.query(PipelineRun)
            .order_by(PipelineRun.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [r.to_dict() for r in runs]


def count_runs() -> int:
    with get_session() as s:
        return s.query(PipelineRun).count()


def update_run_status(run_id: str, status: RunStatus, error_message: str = None):
    with get_session() as s:
        run = s.query(PipelineRun).filter_by(id=run_id).first()
        if run:
            run.status = status
            run.updated_at = datetime.utcnow()
            if error_message:
                run.error_message = error_message


def delete_run(run_id: str):
    with get_session() as s:
        run = s.query(PipelineRun).filter_by(id=run_id).first()
        if run:
            s.delete(run)


# ─── Task CRUD ────────────────────────────────────────────────────────────────

def upsert_agent_task(run_id: str, agent_name: str, input_data: dict = None) -> str:
    with get_session() as s:
        task = (
            s.query(AgentTask)
            .filter_by(run_id=run_id, agent_name=agent_name)
            .first()
        )
        if task:
            if task.status in (TaskStatus.FAILED,):
                task.status = TaskStatus.PENDING
                task.error_message = None
                task.started_at = None
                task.completed_at = None
            return task.id
        task = AgentTask(
            run_id=run_id, agent_name=agent_name, status=TaskStatus.PENDING
        )
        if input_data:
            task.set_input(input_data)
        s.add(task)
        s.flush()
        return task.id


def reset_agent_task(run_id: str, agent_name: str):
    """Reset a task to PENDING so it can be re-run (used by resume/retry)."""
    with get_session() as s:
        task = (
            s.query(AgentTask)
            .filter_by(run_id=run_id, agent_name=agent_name)
            .first()
        )
        if task:
            task.status = TaskStatus.PENDING
            task.error_message = None
            task.started_at = None
            task.completed_at = None
            # Don't reset retry_count so we track total attempts
        else:
            # Create fresh task record
            s.add(AgentTask(
                run_id=run_id, agent_name=agent_name, status=TaskStatus.PENDING
            ))


def mark_task_running(run_id: str, agent_name: str):
    with get_session() as s:
        task = (
            s.query(AgentTask)
            .filter_by(run_id=run_id, agent_name=agent_name)
            .first()
        )
        if task:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()


def mark_task_completed(run_id: str, agent_name: str, output_data: dict):
    with get_session() as s:
        task = (
            s.query(AgentTask)
            .filter_by(run_id=run_id, agent_name=agent_name)
            .first()
        )
        if task:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.set_output(output_data)


def mark_task_failed(run_id: str, agent_name: str, error_message: str):
    with get_session() as s:
        task = (
            s.query(AgentTask)
            .filter_by(run_id=run_id, agent_name=agent_name)
            .first()
        )
        if task:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.error_message = error_message
            task.retry_count = (task.retry_count or 0) + 1


def get_completed_agents(run_id: str) -> List[str]:
    with get_session() as s:
        tasks = (
            s.query(AgentTask)
            .filter_by(run_id=run_id, status=TaskStatus.COMPLETED)
            .all()
        )
        # Return in AGENT_SEQUENCE order
        from graph.pipeline import AGENT_SEQUENCE
        order = {name: i for i, name in enumerate(AGENT_SEQUENCE)}
        return sorted(
            [t.agent_name for t in tasks],
            key=lambda n: order.get(n, 999),
        )


def get_task_output(run_id: str, agent_name: str) -> dict:
    with get_session() as s:
        task = (
            s.query(AgentTask)
            .filter_by(run_id=run_id, agent_name=agent_name, status=TaskStatus.COMPLETED)
            .first()
        )
        return task.get_output() if task else {}


def get_task_retry_count(run_id: str, agent_name: str) -> int:
    with get_session() as s:
        task = (
            s.query(AgentTask)
            .filter_by(run_id=run_id, agent_name=agent_name)
            .first()
        )
        return task.retry_count if task else 0


def get_all_tasks_for_run(run_id: str) -> List[dict]:
    with get_session() as s:
        tasks = (
            s.query(AgentTask)
            .filter_by(run_id=run_id)
            .order_by(AgentTask.started_at)
            .all()
        )
        return [t.to_dict() for t in tasks]


# ─── Logs ─────────────────────────────────────────────────────────────────────

def add_log(run_id: str, agent_name: str, message: str, level: str = "INFO"):
    lm = {
        "INFO": LogLevel.INFO,
        "WARNING": LogLevel.WARNING,
        "ERROR": LogLevel.ERROR,
    }
    try:
        with get_session() as s:
            log = AgentLog(
                run_id=run_id,
                agent_name=agent_name,
                log_level=lm.get(level.upper(), LogLevel.INFO),
                message=message,
                timestamp=datetime.utcnow(),
            )
            s.add(log)
    except Exception as e:
        logger.error(f"Failed to write log to DB: {e}")


def get_logs_for_run(
    run_id: str, agent_name: str = None, limit: int = 500
) -> List[dict]:
    with get_session() as s:
        q = s.query(AgentLog).filter_by(run_id=run_id)
        if agent_name:
            q = q.filter_by(agent_name=agent_name)
        logs = q.order_by(AgentLog.timestamp.desc()).limit(limit).all()
        return [l.to_dict() for l in reversed(logs)]
