from pathlib import Path

import pytest

from src.documents import load_documents, read_document, split_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_all_project_documents_are_readable() -> None:
    documents, duplicates = load_documents(PROJECT_ROOT / "data")
    assert len(documents) >= 1
    assert all(document.text.strip() for document in documents)
    assert all(document.path.suffix.lower() in {".txt", ".doc"} for document in documents)
    assert len(duplicates) >= 1
    assert any("지방세" in document.text for document in documents if document.path.suffix == ".doc")


def test_rtf_doc_is_converted_to_plain_text() -> None:
    doc_path = next((PROJECT_ROOT / "data").glob("*.doc"))
    text = read_document(doc_path)
    assert len(text) > 100
    assert "{\\rtf" not in text


def test_split_text_respects_size_and_overlap() -> None:
    text = " ".join(f"문장-{index}" for index in range(500))
    chunks = split_text(text, chunk_size=200, overlap=30)
    assert len(chunks) > 2
    assert all(0 < len(chunk) <= 200 for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (100, -1), (100, 100), (100, 101)],
)
def test_split_text_rejects_invalid_settings(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("테스트 문서", chunk_size=chunk_size, overlap=overlap)
