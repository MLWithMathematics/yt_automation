"""
agents/base_agent.py — BaseAgent with self-understanding, validation, logging.
"""
import logging
import time
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

import config
import database.db as db

logger = logging.getLogger(__name__)


class AgentInputError(Exception):
    pass


class BaseAgent(ABC):
    """
    BaseAgent enforces the understand → validate → execute → log pattern.
    Every agent must inherit this class and implement execute().
    """

    def __init__(
        self,
        name: str,
        role_description: str,
        inputs_expected: List[str],
        outputs_produced: List[str],
    ):
        self.name = name
        self.role = role_description
        self.inputs_expected = inputs_expected
        self.outputs_produced = outputs_produced
        self._llm = None

    @property
    def llm(self) -> ChatGroq:
        if self._llm is None:
            self._llm = ChatGroq(
                api_key=config.GROQ_API_KEY,
                model=config.GROQ_MODEL,
                temperature=0.7,
                max_tokens=4096,
            )
        return self._llm

    def _log(self, run_id: str, message: str, level: str = "INFO"):
        """Write to both Python logger and DB agent_logs table."""
        log_fn = {"INFO": logger.info, "WARNING": logger.warning, "ERROR": logger.error}
        log_fn.get(level, logger.info)(f"[{self.name}] {message}")
        try:
            db.add_log(run_id, self.name, message, level)
        except Exception as e:
            logger.error(f"Failed to write log to DB: {e}")

    def understand_task(self, state: Dict[str, Any]) -> str:
        """
        Use Groq LLM to reason about the current task before executing.
        Logs the plan to agent_logs before execution starts.
        """
        run_id = state.get("run_id", "unknown")
        context_summary = {
            "niche": state.get("niche", ""),
            "completed_agents": state.get("completed_agents", []),
            "current_agent": self.name,
            "available_data_keys": [k for k, v in state.items() if v and k not in ("config",)],
        }

        prompt = f"""You are {self.name}.
Your role: {self.role}
Expected inputs: {self.inputs_expected}
Outputs you will produce: {self.outputs_produced}
Current pipeline context: {json.dumps(context_summary, indent=2)}

In 2-3 sentences, describe your specific execution plan for this task right now.
Be concrete about what API calls, transformations, or generations you will perform."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            plan = response.content.strip()
        except Exception as e:
            plan = f"Plan generation failed ({e}). Proceeding with standard execution."

        self._log(run_id, f"PLAN: {plan}", "INFO")
        return plan

    def validate_inputs(self, state: Dict[str, Any]):
        """Check all required inputs exist in state."""
        run_id = state.get("run_id", "unknown")
        missing = []
        for key in self.inputs_expected:
            if not state.get(key):
                missing.append(key)
        if missing:
            msg = f"Missing required inputs: {missing}"
            self._log(run_id, msg, "ERROR")
            raise AgentInputError(msg)
        self._log(run_id, f"Input validation passed. Keys present: {self.inputs_expected}", "INFO")

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Main agent logic — must be implemented by subclasses."""
        raise NotImplementedError

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrated execution: understand → validate → execute → log result.
        Updates DB task status throughout.
        """
        run_id = state.get("run_id", "unknown")
        self._log(run_id, f"Starting agent: {self.name}", "INFO")

        # Mark running in DB
        db.upsert_agent_task(run_id, self.name)
        db.mark_task_running(run_id, self.name)

        start_time = time.time()
        try:
            # Step 1: Understand
            self.understand_task(state)

            # Step 2: Validate
            self.validate_inputs(state)

            # Step 3: Execute
            result = self.execute(state)

            # Step 4: Log success
            elapsed = round(time.time() - start_time, 2)
            self._log(run_id, f"Completed in {elapsed}s. Outputs: {list(result.keys())}", "INFO")
            db.mark_task_completed(run_id, self.name, result)

            return result

        except AgentInputError as e:
            elapsed = round(time.time() - start_time, 2)
            self._log(run_id, f"Input validation failed after {elapsed}s: {e}", "ERROR")
            db.mark_task_failed(run_id, self.name, str(e))
            raise

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            self._log(run_id, f"Execution failed after {elapsed}s: {e}", "ERROR")
            db.mark_task_failed(run_id, self.name, str(e))
            raise

    def call_llm_with_retry(self, messages: list, run_id: str = "unknown") -> str:
        """Call Groq LLM with rate-limit-aware retry logic."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                return response.content.strip()
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str:
                    wait = 60
                    self._log(run_id, f"Rate limit hit, waiting {wait}s (attempt {attempt+1})", "WARNING")
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("LLM call failed after max retries")

    def parse_llm_json(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code blocks."""
        import re
        # Remove ```json ... ``` wrappers
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        return json.loads(text)
