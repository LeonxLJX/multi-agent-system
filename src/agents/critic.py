"""Critic agent: parses the draft review into a structured verdict.

The convergence contract is deliberately tiny: `SCORE: n` + bullet feedback.
Anything unparseable scores 0 (worst case) so the loop keeps iterating
rather than silently accepting garbage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.agents.base import BaseAgent
from src.blackboard import Blackboard
from src.trace import RunTrace

_SCORE_RE = re.compile(r"SCORE:\s*(\d+)", re.IGNORECASE)


@dataclass
class CriticVerdict:
    score: int                    # 1..10; 0 means "unparseable -> treat as failing"
    feedback: str
    passed: bool = False
    bullets: List[str] = field(default_factory=list)


class CriticAgent(BaseAgent):
    role = "critic"
    name = "Critic"

    def execute(self, goal: str, board: Blackboard, context: str = "",
                trace: Optional[RunTrace] = None, task_id: str = "") -> str:
        return self.review(board, pass_score=0, trace=trace, task_id=task_id).feedback

    def review(self, board: Blackboard, pass_score: int,
               trace: Optional[RunTrace] = None, task_id: str = "") -> CriticVerdict:
        draft = board.latest("draft")
        if draft is None:
            raise ValueError("cannot review: no draft on the blackboard yet")
        raw = self.ask(
            "Review the draft below.",
            context=f"DRAFT (v{draft.version}):\n{draft.content}",
            trace=trace, task_id=task_id,
        )
        verdict = parse_verdict(raw, pass_score=pass_score)
        board.put("review", raw, author=self.name, score=verdict.score)
        return verdict


def parse_verdict(raw: str, pass_score: int) -> CriticVerdict:
    match = _SCORE_RE.search(raw)
    if not match:
        return CriticVerdict(score=0, feedback=raw, passed=False,
                             bullets=["[unparseable review -- treating as failing]"])
    score = max(0, min(10, int(match.group(1))))
    bullets = [line.lstrip("- ").strip() for line in raw.splitlines()
               if line.strip().startswith("-")]
    return CriticVerdict(score=score, feedback=raw, passed=score >= pass_score, bullets=bullets)
