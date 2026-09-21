"""Planner agent: goal -> validated, dependency-ordered task graph."""
from __future__ import annotations

from typing import Optional

from src.agents.base import BaseAgent
from src.task_graph import PlanError, TaskGraph, parse_plan
from src.trace import RunTrace


class PlannerAgent(BaseAgent):
    role = "planner"
    name = "Planner"

    def plan(self, goal: str, trace: Optional[RunTrace] = None, task_id: str = "") -> TaskGraph:
        raw = self.ask(goal, trace=trace, task_id=task_id)
        try:
            return TaskGraph.from_plan(parse_plan(raw))
        except PlanError as exc:
            # One corrective round-trip: feed the error back to the planner.
            raw = self.ask(
                f"{goal}\n\nYour previous plan was invalid ({exc}). Emit valid JSON only.",
                trace=trace, task_id=task_id,
            )
            return TaskGraph.from_plan(parse_plan(raw))
