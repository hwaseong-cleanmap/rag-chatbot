"""Build only the local Ollama backup index.

Run when Ollama is installed or when a separate local backup refresh is needed:
    python -m scripts.build_ollama_index
"""

from __future__ import annotations

from src.config import Settings
from src.rag import LocalKeywordBackup


def main() -> None:
    settings = Settings.from_env()
    settings.validate()
    stats = LocalKeywordBackup(settings).build(
        progress_callback=lambda message, current, total: print(f"[{current}/{total}] {message}", flush=True)
    )
    print(
        f"Ollama 백업 색인 완료: 문서 {stats['documents']}건 · 청크 {stats['chunks']:,}개",
        flush=True,
    )


if __name__ == "__main__":
    main()
