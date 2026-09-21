"""Agent message format and in-process message bus.

The bus is the only channel agents use to talk to the orchestrator (and,
indirectly, to each other). Keeping it explicit -- instead of hidden Python
calls between agents -- buys three things:

1. every interaction is loggable / replayable (see ``trace.py``),
2. routing rules live in one place (who may publish which kind),
3. tests can assert on message flow without inspecting agent internals.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional


@dataclass
class AgentMessage:
    """One unit of inter-agent communication."""

    sender: str                 # agent name, or "orchestrator"
    recipient: str              # agent name, "blackboard", or "*" (broadcast)
    kind: str                   # task.assign | artifact | feedback | status | error
    payload: str                # message body (markdown / json text)
    task_id: str = ""
    metadata: Dict = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        return {
            "msg_id": self.msg_id,
            "ts": self.ts.isoformat(),
            "task_id": self.task_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "kind": self.kind,
            "payload": self.payload,
            "metadata": self.metadata,
        }


Handler = Callable[[AgentMessage], None]


class MessageBus:
    """Tiny pub/sub router with a retained log (for traces and tests)."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Handler]] = {}
        self.log: List[AgentMessage] = []

    def subscribe(self, recipient: str, handler: Handler) -> None:
        self._subscribers.setdefault(recipient, []).append(handler)

    def publish(self, message: AgentMessage) -> None:
        self.log.append(message)
        targets = list(self._subscribers.get(message.recipient, []))
        if message.recipient != "*":
            targets.extend(self._subscribers.get("*", []))
        for handler in targets:
            handler(message)

    def messages_for(self, recipient: str, kind: Optional[str] = None) -> List[AgentMessage]:
        return [
            m for m in self.log
            if m.recipient == recipient and (kind is None or m.kind == kind)
        ]
