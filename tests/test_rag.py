from types import SimpleNamespace

from src.config import Settings
from src.rag import NO_ANSWER, RagService, SearchResult, filter_relevant


def test_filter_relevant_applies_similarity_threshold() -> None:
    results = filter_relevant(
        documents=["관련 근거", "무관한 근거"],
        metadatas=[
            {"source": "관련.txt", "chunk_index": 0},
            {"source": "무관.txt", "chunk_index": 1},
        ],
        distances=[0.2, 0.9],
        min_similarity=0.35,
    )
    assert len(results) == 1
    assert results[0].source == "관련.txt"
    assert results[0].similarity == 0.8


def test_answer_returns_fixed_message_without_evidence() -> None:
    service = object.__new__(RagService)
    service.search = lambda _question: []
    result = service.answer("자료에 없는 질문")
    assert result.answer == NO_ANSWER
    assert result.sources == []


def test_answer_uses_chat_completions_and_returns_sources() -> None:
    service = object.__new__(RagService)
    service.settings = Settings(
        account_id="a" * 32,
        api_token="secret-token",
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{'a' * 32}/ai/v1",
    )
    service.search = lambda _question: [
        SearchResult("포상금 지급 근거", "조례.doc", 0.8, 0)
    ]
    create_calls = []

    def create(**kwargs):
        create_calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="문서상 지급 대상입니다.")
                )
            ]
        )

    service.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = service.answer("지급 대상은 누구인가요?")

    assert result.answer == "문서상 지급 대상입니다."
    assert result.sources == ["조례.doc"]
    assert create_calls[0]["model"] == service.settings.chat_model
    assert create_calls[0]["temperature"] == 0
    assert create_calls[0]["max_tokens"] == 2000


def test_rag_service_exposes_search_methods() -> None:
    """Regression test for methods accidentally nested in the build function."""
    assert hasattr(RagService, "_stats")
    assert hasattr(RagService, "document_stats")
    assert hasattr(RagService, "search")
    assert hasattr(RagService, "answer")
