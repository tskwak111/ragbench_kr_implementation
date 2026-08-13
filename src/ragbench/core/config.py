"""Typed runtime configuration for RAGBench-KR."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Self

from pydantic import PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables and an optional local `.env` file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://ragbench:ragbench@localhost:5432/ragbench"
    upstage_api_key: str | None = None
    upstage_base_url: str = "https://api.upstage.ai/v1"

    max_project_budget_usd: Decimal = Decimal("135.00")
    max_concurrency: int = 5
    max_lock_connections: PositiveInt = 2
    max_retries: int = 5
    run_live_upstage_tests: bool = False
    allow_gold_access: bool = False

    cache_dir: Path = Path(".ragbench/cache")
    data_dir: Path = Path("data")

    upstage_solar_pro3_model_id: str = "solar-pro3"
    upstage_solar_pro4_model_id: str = "solar-pro4"
    upstage_embedding_model_id: str = "embedding-query"
    upstage_document_parse_model_id: str = "document-parse"
    upstage_embed_2_promotion_ends_at: datetime = datetime(2026, 8, 23, tzinfo=UTC)

    @model_validator(mode="after")
    def require_api_key_for_live_tests(self) -> Self:
        """Prevent accidental live runs without an explicitly supplied credential."""
        if self.run_live_upstage_tests and not self.upstage_api_key:
            msg = "UPSTAGE_API_KEY is required when RUN_LIVE_UPSTAGE_TESTS=1"
            raise ValueError(msg)
        return self
