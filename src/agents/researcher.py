"""Researcher agent: gathers facts and writes them to the blackboard."""
from __future__ import annotations

from typing import Optional

from src.agents.base import BaseAgent
from src.blackboard import Blackboard
from src.trace import RunTrace


class ResearcherAgent(BaseAgent):
    role = "researcher"
    name = "Researcher"

    def execute(self, goal: str, board: Blackboard, context: str = "",
                trace: Optional[RunTrace] = None, task_id: str = "") -> str:
        notes = self.ask(goal, context=context, trace=trace, task_id=task_id)
        artifact = board.put("research", notes, author=self.name, step_goal=goal)
        return artifact.content
