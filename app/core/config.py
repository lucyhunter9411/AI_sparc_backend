# app/core/config.py

import os
from dotenv import load_dotenv
from pathlib import Path
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

# ─── 1) Load your .env into os.environ ─────────────────────────────────────────
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path, override=True)


class Settings(BaseSettings):
    # ──────────────────────────────────────────────────────────
    testing: bool = Field(False, env="TESTING")

    # ─────────────────────── Mongo / Cosmos ──────────────────
    mongo_user:     str = Field(..., env="MONGO_USER")
    mongo_password: str = Field(..., env="MONGO_PASSWORD")
    mongo_db:       str = Field(..., env="DB_NAME")
    mongo_host:     str = Field("cosmon-sparc-dev-wus-001.global.mongocluster.cosmos.azure.com", env="MONGO_HOST")

    # ───────────────────── Azure / OpenAI ────────────────────
    azure_openai_key:     str | None = Field(None, env="AZURE_OPENAI_KEY")
    azure_openai_base:    str | None = Field(None, env="AZURE_OPENAI_BASE")   # ← now a str, not AnyUrl
    azure_openai_version: str         = Field("2023-12-01-preview", env="AZURE_OPENAI_VERSION")
    azure_deploy_41mini:  str | None = Field(None, env="AZURE_OPENAI_DEPLOYMENT_41_Mini")
    azure_deploy_4:       str | None = Field(None, env="AZURE_OPENAI_DEPLOYMENT_4")
    azure_deploy_35:      str | None = Field(None, env="AZURE_OPENAI_DEPLOYMENT_35_TURBO")

    # ───────────────────────── Speech ─────────────────────────
    tts_subscription_key: str | None = Field(None, env="SUBSCRIPTION_KEY")
    tts_region:           str | None = Field(None, env="REGION")

    # ───────────────────────── Frontend ────────────────────────
    frontend_url:         str | None = Field(None, env="FRONTEND_URL")

    # ───────────────────────── Misc ───────────────────────────
    project_root:    Path = Path(__file__).resolve().parents[2]
    model_cache_size: int  = 4

    @model_validator(mode="after")
    def _require_mongo_creds(self) -> "Settings":
        if not self.testing:
            missing = [
                name for name in ("mongo_db", "mongo_user", "mongo_password")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(f"Missing required env-vars: {', '.join(missing)}")
        return self


@lru_cache
def get_settings() -> Settings:
    """
    Build Settings from os.environ, extracting only the keys we need.
    """
    data = {
        "mongo_user":     os.getenv("MONGO_USER"),
        "mongo_password": os.getenv("MONGO_PASSWORD"),
        "mongo_db":       os.getenv("DB_NAME"),
        # 'mongo_host' intentionally omitted to allow default fallback
        "azure_openai_key":     os.getenv("AZURE_OPENAI_KEY"),
        "azure_openai_base":    os.getenv("AZURE_OPENAI_BASE"),
        "azure_openai_version": os.getenv("AZURE_OPENAI_VERSION"),
        "azure_deploy_41mini":  os.getenv("AZURE_OPENAI_DEPLOYMENT_41_Mini"),
        "azure_deploy_4":       os.getenv("AZURE_OPENAI_DEPLOYMENT_4"),
        "azure_deploy_35":      os.getenv("AZURE_OPENAI_DEPLOYMENT_35_Turbo"),
        "tts_subscription_key": os.getenv("SUBSCRIPTION_KEY"),
        "tts_region":           os.getenv("REGION"),
        # 'frontend_url' intentionally omitted to allow environment variable handling
        "testing":              os.getenv("TESTING", "0") == "1",
    }
    return Settings.model_validate(data)
