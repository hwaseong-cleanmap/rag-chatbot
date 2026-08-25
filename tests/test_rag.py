from src.rag import NO_ANSWER, RagService, filter_relevant


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
