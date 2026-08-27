"""Build the local RAG search index before starting the Streamlit service.

Run from the project root:
    python -m scripts.build_index
"""

from __future__ import annotations

from src.config import Settings
from src.rag import build_index_atomically


def main() -> None:
    settings = Settings.from_env()
    try:
        settings.validate()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"설정 오류: {error}") from error

    last_document_reported = 0

    def report(message: str, current: int, total: int) -> None:
        nonlocal last_document_reported
        # Avoid flooding the terminal while still proving that a large corpus is moving.
        if message.startswith("문서 읽는 중"):
            if current != total and current - last_document_reported < 5:
                return
            last_document_reported = current
        print(f"[{current:,}/{total:,}] {message}", flush=True)

    print("업무자료 검색 색인을 준비합니다.", flush=True)
    # The live DB is replaced only after the staging DB finishes validation.
    # Close Streamlit before running this command on Windows.
    stats = build_index_atomically(settings, report=report)
    print(
        "색인 완료: "
        f"문서 {stats['documents']}건, 청크 {stats['chunks']:,}개, "
        f"처리 실패 {stats['failures']}건",
        flush=True,
    )


if __name__ == "__main__":
    main()
