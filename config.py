"""Configuration loaded from environment / `.env` and the OpenAI client factory."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent

# override=False: variables injected by the shell (e.g. `op run --env-file=.env.op`)
# take precedence over values in `.env`.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").split("#", 1)[0].strip()
    return int(raw) if raw else default


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name, "").split("#", 1)[0].strip()
    return float(raw) if raw else default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name, "").split("#", 1)[0].strip()
    return raw or default


def _resolve_python(path: str) -> str:
    """Relative paths such as `.venv-bcb/bin/python` are resolved against the project root."""
    if os.sep in path and not Path(path).is_absolute():
        return str(PROJECT_ROOT / path)
    return path


@dataclass(frozen=True)
class Config:
    model: str
    max_samples: int
    dataset: str
    max_correction_rounds: int
    temperature: float | None
    exec_timeout_s: float
    exec_python: str
    exec_memory_mb: int
    concurrency: int
    results_dir: Path
    data_dir: Path

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            model=_env_str("OPENAI_MODEL", "gpt-4o"),
            max_samples=_env_int("MAX_SAMPLES", 50),
            dataset=_env_str("DATASET", "mbpp"),
            max_correction_rounds=_env_int("MAX_CORRECTION_ROUNDS", 1),
            temperature=_env_float("TEMPERATURE", 0.0),
            exec_timeout_s=_env_float("EXEC_TIMEOUT_S", 10.0),
            exec_python=_resolve_python(_env_str("EXEC_PYTHON", sys.executable)),
            exec_memory_mb=_env_int("EXEC_MEMORY_MB", 4096),
            concurrency=_env_int("CONCURRENCY", 4),
            results_dir=PROJECT_ROOT / _env_str("RESULTS_DIR", "results"),
            data_dir=PROJECT_ROOT / _env_str("DATA_DIR", "data"),
        )


def make_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in .env or inject it with "
            "`op run --env-file=.env.op -- python main.py`."
        )
    return OpenAI(api_key=api_key, max_retries=5, timeout=180.0)
