"""All tunable settings in one place. Override any field with an environment variable or a .env file."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.5-flash")
    router_model: str = os.getenv("ROUTER_MODEL", "gemini-3.5-flash-lite")
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # local via fastembed; no API quota
    embedding_dim: int = 384

    pdf_path: Path = ROOT / "data" / "raw" / "iMac_PER_Oct2024.pdf"
    index_dir: Path = ROOT / "data" / "index"
    cache_dir: Path = ROOT / "data" / "cache"
    vision_cache_dir: Path = ROOT / "data" / "cache" / "vision"

    # Retrieval. Chunks are one per element (paragraph, table, chart), so no parent/child split here.
    dense_k: int = 25
    bm25_k: int = 25
    rrf_k: int = 60
    candidates: int = 10  # chunks passed to the reranker
    final_k: int = 5  # chunks passed to the answer model
    use_reranker: bool = True  # measured on eval: 9/9 with it on vs 8/9 off (see README)


def get_settings() -> Settings:
    return Settings()
