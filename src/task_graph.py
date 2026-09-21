"""Task graph: planner output -> dependency-resolved execution order.

The planner agent emits JSON steps; this module validates that plan,
detects cycles and computes a topological order. Keeping scheduling as a
pure data structure (no LLM in the loop) is what makes orchestration
deterministic and unit-testable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


class PlanError(ValueError):
    """Raised when a planner-produced plan is unusable."""


@dataclass
class TaskNode:
    id: str
    agent: str
    goal: str
    depends_on: List[str] = field(default_factory=list)
    state: str = "pending"        # pending | running | done | failed | skipped


VALID_AGENTS = {"planner", "researcher", "writer", "critic"}


def parse_plan(raw: str) -> List[Dict]:
    """Extract the steps list from an LLM response (tolerant of code fences)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanError(f"planner output is not valid JSON: {exc}") from exc
    steps = data.get("steps") if isinstance(data, dict) else data
    if not isinstance(steps, list) or not steps:
        raise PlanError("plan must contain a non-empty 'steps' list")
    return steps


class TaskGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, TaskNode] = {}

    def add(self, node: TaskNode) -> None:
        if node.id in self.nodes:
            raise PlanError(f"duplicate step id: {node.id}")
        unknown_agents = {node.agent} - VALID_AGENTS
        if unknown_agents:
            raise PlanError(f"step {node.id!r} assigns unknown agent {node.agent!r}")
        self.nodes[node.id] = node

    @classmethod
    def from_plan(cls, steps: Sequence[Dict]) -> "TaskGraph":
        graph = cls()
        for i, s in enumerate(steps):
            try:
                node = TaskNode(
                    id=str(s["id"]),
                    agent=str(s["agent"]).lower(),
                    goal=str(s.get("goal", "")),
                    depends_on=[str(d) for d in s.get("depends_on", [])],
                )
            except KeyError as exc:
                raise PlanError(f"step #{i} missing field {exc}") from exc
            graph.add(node)
        graph._validate_deps()
        return graph

    def _validate_deps(self) -> None:
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise PlanError(f"step {node.id!r} depends on unknown step {dep!r}")

    def topo_order(self) -> List[str]:
        """Kahn's algorithm; raises on cycles instead of silently proceeding."""
        indegree = {nid: len(n.depends_on) for nid, n in self.nodes.items()}
        children: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dep in node.depends_on:
                children[dep].append(nid)
        queue = sorted(nid for nid, deg in indegree.items() if deg == 0)
        order: List[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for child in children[nid]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
            queue.sort()
        if len(order) != len(self.nodes):
            raise PlanError("plan contains a dependency cycle")
        return order

    def ready_nodes(self) -> List[str]:
        """Pending nodes whose dependencies are all done."""
        return [
            nid for nid, n in self.nodes.items()
            if n.state == "pending"
            and all(self.nodes[d].state == "done" for d in n.depends_on)
        ]

    def mark(self, node_id: str, state: str) -> None:
        self.nodes[node_id].state = state

    def is_finished(self) -> bool:
        return all(n.state in {"done", "failed", "skipped"} for n in self.nodes.values())

    def downstream_of(self, node_id: str) -> List[str]:
        """All transitively dependent nodes -- used to skip after failure."""
        seen: List[str] = []
        frontier = [n for n in self.nodes.values() if node_id in n.depends_on]
        while frontier:
            node = frontier.pop()
            if node.id in seen:
                continue
            seen.append(node.id)
            frontier.extend(n for n in self.nodes.values() if node.id in n.depends_on)
        return seen
