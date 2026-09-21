"""LLM client abstraction: provider-agnostic interface + two implementations.

Every agent talks to ``LLMClient``, never to a vendor SDK directly. That is
what makes the whole system testable offline (``MockLLMClient``) and keeps
the OpenAI dependency behind a single seam.

* ``OpenAIClient`` -- thin wrapper over the chat-completions API with
  timeout, retry-with-backoff and token accounting.
* ``MockLLMClient`` -- deterministic, scripted responses keyed off the
  system prompt; no network, used by tests / CI / demo mode.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class LLMUsage:
    """Token accounting for one completion call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    content: str
    usage: LLMUsage = field(default_factory=LLMUsage)


class BudgetExceeded(RuntimeError):
    """Raised when a run crosses its configured token budget."""


class LLMClient(ABC):
    """Minimal chat interface every provider must implement."""

    @abstractmethod
    def complete(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        ...

    @property
    @abstractmethod
    def total_usage(self) -> LLMUsage:
        """Cumulative usage across all calls on this client."""


def _estimate_tokens(text: str) -> int:
    """Rough 4-chars-per-token heuristic -- good enough for budget guards."""
    return max(1, len(text) // 4)


class OpenAIClient(LLMClient):
    """Real provider. Constructed lazily so `openai` is only needed in prod."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: int = 30,
        max_retries: int = 2,
    ) -> None:
        from openai import OpenAI  # imported here on purpose: mock mode needs nothing

        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model
        self._usage = LLMUsage()

    def complete(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.4),
        )
        choice = resp.choices[0].message.content or ""
        usage = LLMUsage(
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else _estimate_tokens(str(messages)),
            completion_tokens=resp.usage.completion_tokens if resp.usage else _estimate_tokens(choice),
        )
        self._usage.prompt_tokens += usage.prompt_tokens
        self._usage.completion_tokens += usage.completion_tokens
        return LLMResponse(content=choice, usage=usage)

    @property
    def total_usage(self) -> LLMUsage:
        return self._usage


# A handler maps (system_prompt, user_prompt) -> response text.
MockHandler = Callable[[str, str], str]


class MockLLMClient(LLMClient):
    """Deterministic offline client.

    Responses are produced by a *handler registry*: each registered handler
    claims requests whose system prompt contains a marker string. This lets
    tests script per-role behaviour (planner vs critic) without touching
    agent code, and lets the demo run end-to-end with zero secrets.
    """

    def __init__(self, handlers: Optional[Dict[str, MockHandler]] = None) -> None:
        self._handlers: Dict[str, MockHandler] = dict(handlers or {})
        self._usage = LLMUsage()
        self.calls: List[List[Dict[str, str]]] = []  # full request log for assertions

    def register(self, marker: str, handler: MockHandler) -> "MockLLMClient":
        self._handlers[marker] = handler
        return self

    def complete(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        self.calls.append(messages)
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        for marker, handler in self._handlers.items():
            if marker in system:
                text = handler(system, user)
                usage = LLMUsage(_estimate_tokens(system + user), _estimate_tokens(text))
                self._usage.prompt_tokens += usage.prompt_tokens
                self._usage.completion_tokens += usage.completion_tokens
                return LLMResponse(content=text, usage=usage)
        raise AssertionError(f"MockLLMClient has no handler matching system prompt: {system[:80]!r}")

    @property
    def total_usage(self) -> LLMUsage:
        return self._usage


def build_client(settings) -> LLMClient:
    """Factory honoring ``settings.llm_provider`` ('mock' | 'openai')."""
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("llm_provider=openai requires OPENAI_API_KEY")
        return OpenAIClient(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
    return default_mock_client()


def default_mock_client() -> MockLLMClient:
    """Scripted behaviours mirroring the content-pipeline demo roles."""
    client = MockLLMClient()

    client.register(
        "Task Planner",
        lambda s, u: json.dumps(
            {
                "steps": [
                    {"id": "research", "agent": "researcher", "goal": "Gather market facts"},
                    {"id": "draft", "agent": "writer", "goal": "Write the report", "depends_on": ["research"]},
                    {"id": "review", "agent": "critic", "goal": "Score the draft", "depends_on": ["draft"]},
                ]
            }
        ),
    )
    client.register(
        "Research Analyst",
        lambda s, u: "FINDINGS:\n- Market size ~$2.3B (2026)\n- Leaders: GitHub Copilot, Cursor, Windsurf\n- Growth driver: agentic workflows",
    )
    client.register(
        "Professional Writer",
        lambda s, u: "# AI Coding Tools Market Analysis\n\nThe AI coding-tools market reached roughly $2.3B in 2026, led by GitHub Copilot and Cursor...\n\n## Outlook\nAgentic workflows are the primary growth driver.",
    )
    client.register(
        "Quality Critic",
        lambda s, u: "SCORE: 7\nFEEDBACK:\n- Add pricing comparison\n- Cite sources for the $2.3B figure",
    )

    return client
