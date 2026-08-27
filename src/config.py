"""Cloudflare Workers AI 기반 애플리케이션 설정."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHAT_MODEL = "@cf/qwen/qwen3-30b-a3b-fp8"
DEFAULT_EMBEDDING_MODEL = "@cf/baai/bge-m3"


@dataclass(frozen=True)
class Settings:
    account_id: str
    api_token: str
    base_url: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chat_model: str = DEFAULT_CHAT_MODEL
    data_dir: Path = PROJECT_ROOT / "data"
    db_dir: Path = PROJECT_ROOT / "vector_db"
    collection_name: str = "hwaseong_collection_manuals"
    # Larger, overlapping chunks preserve a complete statutory article or a
    # short procedure section.  This reduces the chance of answering from a
    # single item in a numbered list.
    chunk_size: int = 1800
    chunk_overlap: int = 300
    top_k: int = 6
    min_similarity: float = 0.35
    max_answer_tokens: int = 1000
    embedding_batch_size: int = 32

    @classmethod
    def from_env(cls, secrets: Mapping[str, object] | None = None) -> "Settings":
        """로컬 .env와 Streamlit Secrets를 함께 읽는다.

        Streamlit Secrets 값이 있으면 로컬 환경변수보다 우선한다.
        """

        load_dotenv(PROJECT_ROOT / ".env")
        secret_values = secrets or {}

        def value(name: str, default: str = "") -> str:
            raw = secret_values.get(name, os.getenv(name, default))
            return str(raw).strip()

        account_id = value("CLOUDFLARE_ACCOUNT_ID")
        base_url = ""
        if account_id:
            base_url = (
                "https://api.cloudflare.com/client/v4/accounts/"
                f"{account_id}/ai/v1"
            )

        return cls(
            account_id=account_id,
            api_token=value("CLOUDFLARE_API_TOKEN"),
            base_url=base_url.rstrip("/"),
            embedding_model=value("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            chat_model=value("CHAT_MODEL", DEFAULT_CHAT_MODEL),
        )

    def validate(self) -> None:
        if not re.fullmatch(r"[0-9a-fA-F]{32}", self.account_id):
            raise ValueError(
                "CLOUDFLARE_ACCOUNT_ID는 이메일이 아니라 Cloudflare에서 복사한 "
                "32자리 Account ID여야 합니다."
            )
        if not self.api_token or self.api_token.startswith("your_"):
            raise ValueError(
                "CLOUDFLARE_API_TOKEN이 설정되지 않았습니다. "
                ".env 또는 Streamlit Secrets를 확인하세요."
            )
        parsed_url = urlparse(self.base_url)
        if parsed_url.scheme != "https" or parsed_url.netloc != "api.cloudflare.com":
            raise ValueError("CLOUDFLARE_BASE_URL이 올바른 HTTPS 주소가 아닙니다.")
        expected_path = f"/client/v4/accounts/{self.account_id}/ai/v1"
        if parsed_url.path.rstrip("/") != expected_path:
            raise ValueError(
                "CLOUDFLARE_BASE_URL의 Account ID가 설정값과 일치하지 않습니다."
            )
        if not self.chat_model.startswith("@cf/"):
            raise ValueError("CHAT_MODEL은 @cf/로 시작하는 Workers AI 모델이어야 합니다.")
        if not self.embedding_model.startswith("@cf/"):
            raise ValueError(
                "EMBEDDING_MODEL은 @cf/로 시작하는 Workers AI 모델이어야 합니다."
            )
        if not self.data_dir.exists():
            raise FileNotFoundError(f"문서 폴더를 찾을 수 없습니다: {self.data_dir}")
