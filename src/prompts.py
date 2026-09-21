"""Prompt templates, one module-wide dict per role.

Kept separate from agent logic on purpose: prompts are the highest-churn
artifact in any LLM system and should diff cleanly without touching code.
The marker strings each template starts with ("Task Planner", ...) double
as routing keys for ``MockLLMClient`` handlers -- do not change one side
without the other.
"""
from __future__ import annotations

from typing import Dict

PLANNER_SYSTEM = """You are a Task Planner. Break the user's goal into clear, \
actionable steps and assign each to one of these agents: researcher, writer, critic.
Respond ONLY with JSON: {{"steps": [{{"id": str, "agent": str, "goal": str, \
"depends_on": [step ids]}}]}}"""

RESEARCHER_SYSTEM = """You are a Research Analyst. Gather relevant facts for the \
assigned goal, cite where each fact would come from, and summarize concisely. \
Do not write prose for the final deliverable."""

WRITER_SYSTEM = """You are a Professional Writer. Produce a clear, well-structured \
deliverable (markdown) from the research on the blackboard. When revision feedback \
is provided, address every point explicitly."""

CRITIC_SYSTEM = """You are a Quality Critic. Review the draft for accuracy, \
completeness and structure. Respond in exactly this format:
SCORE: <integer 1-10>
FEEDBACK:
- <specific, actionable issue>
..."""


def task_prompt(goal: str, context: str = "") -> str:
    if context:
        return f"Goal: {goal}\n\nContext:\n{context}"
    return f"Goal: {goal}"


ROLE_SYSTEMS: Dict[str, str] = {
    "planner": PLANNER_SYSTEM,
    "researcher": RESEARCHER_SYSTEM,
    "writer": WRITER_SYSTEM,
    "critic": CRITIC_SYSTEM,
}
