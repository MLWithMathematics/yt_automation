"""database/models.py — SQLAlchemy ORM models."""
import uuid, json
from datetime import datetime
from enum import Enum as PyEnum
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

def gen_uuid(): return str(uuid.uuid4())

class RunStatus(str, PyEnum):
    PENDING="PENDING"; RUNNING="RUNNING"; PAUSED="PAUSED"
    COMPLETED="COMPLETED"; FAILED="FAILED"

class TaskStatus(str, PyEnum):
    PENDING="PENDING"; RUNNING="RUNNING"; COMPLETED="COMPLETED"
    FAILED="FAILED"; SKIPPED="SKIPPED"

class LogLevel(str, PyEnum):
    INFO="INFO"; WARNING="WARNING"; ERROR="ERROR"

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    niche = Column(String(128), nullable=False)
    status = Column(Enum(RunStatus), nullable=False, default=RunStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    retry_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    tasks = relationship("AgentTask", back_populates="run", cascade="all, delete-orphan")
    logs = relationship("AgentLog", back_populates="run", cascade="all, delete-orphan")
    def to_dict(self):
        return {"id":self.id,"niche":self.niche,
                "status":self.status.value if self.status else None,
                "created_at":self.created_at.isoformat() if self.created_at else None,
                "updated_at":self.updated_at.isoformat() if self.updated_at else None,
                "retry_count":self.retry_count,"error_message":self.error_message}

class AgentTask(Base):
    __tablename__ = "agent_tasks"
    id = Column(String(36), primary_key=True, default=gen_uuid)
    run_id = Column(String(36), ForeignKey("pipeline_runs.id"), nullable=False)
    agent_name = Column(String(64), nullable=False)
    status = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.PENDING)
    input_data = Column(Text, nullable=True)
    output_data = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    run = relationship("PipelineRun", back_populates="tasks")
    def get_input(self): return json.loads(self.input_data) if self.input_data else {}
    def set_input(self, d): self.input_data = json.dumps(d)
    def get_output(self): return json.loads(self.output_data) if self.output_data else {}
    def set_output(self, d): self.output_data = json.dumps(d)
    def duration_seconds(self):
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    def to_dict(self):
        return {"id":self.id,"run_id":self.run_id,"agent_name":self.agent_name,
                "status":self.status.value if self.status else None,
                "input_data":self.get_input(),"output_data":self.get_output(),
                "started_at":self.started_at.isoformat() if self.started_at else None,
                "completed_at":self.completed_at.isoformat() if self.completed_at else None,
                "error_message":self.error_message,"retry_count":self.retry_count,
                "duration_seconds":self.duration_seconds()}

class AgentLog(Base):
    __tablename__ = "agent_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), ForeignKey("pipeline_runs.id"), nullable=False)
    agent_name = Column(String(64), nullable=False)
    log_level = Column(Enum(LogLevel), nullable=False, default=LogLevel.INFO)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    run = relationship("PipelineRun", back_populates="logs")
    def to_dict(self):
        return {"id":self.id,"run_id":self.run_id,"agent_name":self.agent_name,
                "log_level":self.log_level.value if self.log_level else "INFO",
                "message":self.message,
                "timestamp":self.timestamp.isoformat() if self.timestamp else None}
