"""BaseAgent: one role, one prompt, retry + budget-aware LLM access.

All orchestration policy (what to run next, what context to inject) lives
in the orchestrator; agents stay deliberately dumb so each one is testable
with a scripted MockLLMClient and a hand-built blackboard.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from src.llm import BudgetExceeded, LLMClient
from src.prompts import ROLE_SYSTEMS, task_prompt
from src.trace import RunTrace

logger = logging.getLogger("mas.agent")


class AgentError(RuntimeError):
    """Raised when an agent exhausts its retries for one step."""


class BaseAgent:
    role: str = ""                    # key into ROLE_SYSTEMS
    name: str = ""

    def __init__(self, client: LLMClient, max_retries: int = 2) -> None:
        self.client = client
        self.max_retries = max_retries
        self.system_prompt = ROLE_SYSTEMS[self.role]

    def ask(self, goal: str, context: str = "", trace: Optional[RunTrace] = None,
            task_id: str = "") -> str:
        """One logical step: system + context + goal -> text, with retries."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_prompt(goal, context)},
        ]
        last_error: Optional[Exception] = None
        for attempt in range(1 + self.max_retries):
            started = time.monotonic()
            try:
                before = self.client.total_usage.total
                response = self.client.complete(messages)
                latency = (time.monotonic() - started) * 1000
                if trace:
                    trace.record(
                        agent=self.name, action="llm_call",
                        prompt_chars=sum(len(m["content"]) for m in messages),
                        response_chars=len(response.content),
                        tokens_used=self.client.total_usage.total - before,
                        latency_ms=round(latency, 1), ok=True,
                    )
                return response.content.strip()
            except BudgetExceeded:
                raise                      # guards are not retryable
            except Exception as exc:       # noqa: BLE001 -- provider errors vary widely
                last_error = exc
                logger.warning("%s attempt %d failed: %s", self.name, attempt + 1, exc)
                if trace:
                    trace.record(
                        agent=self.name, action="retry" if attempt < self.max_retries else "fallback",
                        prompt_chars=0, response_chars=0, tokens_used=0,
                        latency_ms=0, ok=False, error=str(exc),
                    )
        raise AgentError(f"{self.name} failed after {1 + self.max_retries} attempts: {last_error}")
