"""Shared working memory ("blackboard") for a task.

Agents never pass giant strings through call chains; they write named
artifacts (research notes, draft v1, review v2...) to the blackboard and
reference them by key. The orchestrator decides what context each agent
sees by selecting which keys to render into its prompt.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class Artifact:
    key: str                    # e.g. "draft", "research", "review"
    content: str
    author: str                 # producing agent name
    version: int = 1
    meta: Dict = field(default_factory=dict)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Blackboard:
    """Versioned key/value store scoped to one orchestrated task."""

    def __init__(self) -> None:
        self._store: Dict[str, List[Artifact]] = {}
        self._lock = threading.Lock()

    def put(self, key: str, content: str, author: str, **meta) -> Artifact:
        with self._lock:
            versions = self._store.setdefault(key, [])
            artifact = Artifact(key=key, content=content, author=author,
                                version=len(versions) + 1, meta=meta)
            versions.append(artifact)
            return artifact

    def latest(self, key: str) -> Optional[Artifact]:
        versions = self._store.get(key) or []
        return versions[-1] if versions else None

    def get(self, key: str, version: Optional[int] = None) -> Optional[Artifact]:
        versions = self._store.get(key) or []
        if not versions:
            return None
        if version is None:
            return versions[-1]
        return next((a for a in versions if a.version == version), None)

    def history(self, key: str) -> List[Artifact]:
        return list(self._store.get(key) or [])

    def keys(self) -> List[str]:
        return sorted(self._store)

    def render_context(self, keys: List[str]) -> str:
        """Concatenate latest versions of `keys` for prompt injection."""
        parts: List[str] = []
        for k in keys:
            art = self.latest(k)
            if art:
                parts.append(f"[{k} v{art.version} by {art.author}]\n{art.content}")
        return "\n\n---\n\n".join(parts)
