"""Writer agent: first draft from research, or revision from critic feedback."""
from __future__ import annotations

from typing import Optional

from src.agents.base import BaseAgent
from src.blackboard import Blackboard
from src.trace import RunTrace


class WriterAgent(BaseAgent):
    role = "writer"
    name = "Writer"

    def execute(self, goal: str, board: Blackboard, context: str = "",
                trace: Optional[RunTrace] = None, task_id: str = "") -> str:
        draft = self.ask(goal, context=context, trace=trace, task_id=task_id)
        artifact = board.put("draft", draft, author=self.name, step_goal=goal)
        return artifact.content

    def revise(self, board: Blackboard, feedback: str,
               trace: Optional[RunTrace] = None, task_id: str = "") -> str:
        current = board.latest("draft")
        if current is None:
            raise ValueError("cannot revise: no draft on the blackboard yet")
        context = f"DRAFT (v{current.version}):\n{current.content}\n\nCRITIC FEEDBACK:\n{feedback}"
        revised = self.ask(
            "Revise the draft addressing every feedback point. Output the full new draft only.",
            context=context, trace=trace, task_id=task_id,
        )
        artifact = board.put("draft", revised, author=self.name, revised_from=current.version)
        return artifact.content
