"""Application configuration.

Environment-driven settings with safe defaults: the system boots fully
offline (``llm_provider="mock"``) without an API key, which is what CI
and the test-suite rely on. Real calls require opting in explicitly.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "multi-agent-system"
    api_prefix: str = "/api"

    # ---- LLM client -----------------------------------------------------
    # "mock" -> deterministic offline client (default); "openai" -> real API.
    llm_provider: str = "mock"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_timeout: int = 30
    llm_max_retries: int = 2

    # ---- Cost / runaway guards ------------------------------------------
    token_budget: int = 50_000        # hard cap per task; exceeding aborts
    max_critic_rounds: int = 3        # writer<->critic revision loop cap
    critic_pass_score: int = 8        # score at/above which draft is accepted

    # ---- Service ---------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8002


@lru_cache
def get_settings() -> Settings:
    return Settings()
