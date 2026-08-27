"""Build Cloudflare and Ollama RAG indexes before starting Streamlit.

Run from the project root:
    python -m scripts.build_index
"""

from __future__ import annotations

from src.config import Settings
from src.rag import build_all_indexes


def main() -> None:
    settings = Settings.from_env()
    try:
        settings.validate()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"설정 오류: {error}") from error

    last_document_reported = 0

    def report(message: str, current: int, total: int) -> None:
        nonlocal last_document_reported
        if "문서" in message and current != total and current - last_document_reported < 5:
            return
        last_document_reported = current
        print(f"[{current:,}/{total:,}] {message}", flush=True)

    print("Ollama 로컬 백업 → Cloudflare 순서로 색인을 준비합니다.", flush=True)
    results = build_all_indexes(settings, report=report)
    for provider in ("ollama", "cloudflare"):
        stats = results.get(provider)
        if isinstance(stats, dict):
            print(
                f"{provider} 색인 완료: 문서 {stats['documents']}건 · "
                f"청크 {stats['chunks']:,}개 · 처리 실패 {stats['failures']}건",
                flush=True,
            )
        else:
            print(f"{provider} 색인 실패: {results.get(f'{provider}_error', '알 수 없는 오류')}", flush=True)


if __name__ == "__main__":
    main()
