"""Common utilities: paths, logging, env loading, LLM client factory."""
from __future__ import annotations

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_DISTILLED = ROOT / "data" / "distilled"
DATA_FILTERED = ROOT / "data" / "filtered"
REPORTS = ROOT / "reports"

for p in (DATA_RAW, DATA_PROCESSED, DATA_DISTILLED, DATA_FILTERED, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")
# Fallback: also load Hermes global env (where OPENAI_API_KEY etc. live by default).
# Project .env takes precedence (loaded first, override=False on the second call).
_hermes_env = Path.home() / ".hermes" / ".env"
if _hermes_env.exists():
    load_dotenv(_hermes_env, override=False)


def get_logger(name: str = "wh_bench") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stderr)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s | %(message)s",
                            datefmt="%H:%M:%S")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger


def jsonl_write(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def get_llm_client() -> tuple[Any, str]:
    """Return (openai.OpenAI client, model_name). Auto-detect from env.

    Priority:
      1) OPENAI_API_KEY  (any OpenAI-compatible endpoint via OPENAI_BASE_URL)
      2) FOXNIO_API_KEY

    Note: we override User-Agent because some OpenAI-compatible providers
    (e.g. Foxnio's WAF) block the default ``OpenAI/Python`` UA.
    """
    from openai import OpenAI  # local import to keep cold-start light

    safe_headers = {"User-Agent": "curl/8.4.0"}

    if os.getenv("OPENAI_API_KEY"):
        return (
            OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                default_headers=safe_headers,
            ),
            os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
    if os.getenv("FOXNIO_API_KEY"):
        return (
            OpenAI(
                api_key=os.environ["FOXNIO_API_KEY"],
                base_url=os.getenv("FOXNIO_BASE_URL", "https://api.foxnio.com/v1"),
                default_headers=safe_headers,
            ),
            os.getenv("FOXNIO_MODEL", "gpt-5.5"),
        )
    raise RuntimeError(
        "未找到 LLM API key。请在 .env 设置 OPENAI_API_KEY 或 FOXNIO_API_KEY"
    )


def stable_id(prefix: str, n: int) -> str:
    return f"{prefix}-{n:04d}"
