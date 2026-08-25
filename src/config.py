"""환경변수 기반 애플리케이션 설정."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    api_key: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-5-mini"
    data_dir: Path = PROJECT_ROOT / "data"
    db_dir: Path = PROJECT_ROOT / "vector_db"
    collection_name: str = "hwaseong_civil_documents"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5
    min_similarity: float = 0.35

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ).strip(),
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini").strip(),
        )

    def validate(self) -> None:
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요."
            )
        if not self.data_dir.exists():
            raise FileNotFoundError(f"문서 폴더를 찾을 수 없습니다: {self.data_dir}")

