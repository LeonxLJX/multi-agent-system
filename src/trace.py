"""Structured tracing: every agent step becomes a replayable record.

A trace is what you debug from when a multi-agent run goes sideways --
"which agent said what, with which context, at what token cost". The
orchestrator writes one TraceStep per LLM call; ``export_json`` dumps the
whole run for post-mortems or eval pipelines.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TraceStep:
    seq: int
    task_id: str
    agent: str
    action: str                    # llm_call | retry | fallback | guard_abort
    prompt_chars: int
    response_chars: int
    tokens_used: int
    latency_ms: float
    ok: bool
    error: Optional[str] = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RunTrace:
    """Append-only log of TraceSteps for one orchestrated run."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.steps: List[TraceStep] = []
        self.started_at = datetime.now(timezone.utc)

    def record(self, **kwargs) -> TraceStep:
        step = TraceStep(seq=len(self.steps), task_id=self.task_id, **kwargs)
        self.steps.append(step)
        return step

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_used for s in self.steps)

    def agent_summary(self) -> Dict[str, int]:
        """Per-agent share of total token spend."""
        summary: Dict[str, int] = {}
        for s in self.steps:
            summary[s.agent] = summary.get(s.agent, 0) + s.tokens_used
        return summary

    def export_json(self, path: Optional[Path] = None) -> str:
        blob = {
            "task_id": self.task_id,
            "started_at": self.started_at.isoformat(),
            "total_tokens": self.total_tokens,
            "steps": [_step_dict(s) for s in self.steps],
        }
        text = json.dumps(blob, ensure_ascii=False, indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text


def _step_dict(step: TraceStep) -> Dict:
    d = asdict(step)
    d["ts"] = step.ts.isoformat()
    return d
