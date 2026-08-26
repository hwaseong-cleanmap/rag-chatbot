import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import documents as document_module
from src.documents import load_documents, read_document, split_text


def test_rtf_doc_is_converted_to_plain_text(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.doc"
    doc_path.write_bytes(b"{\\rtf1\\ansi This is a sample RTF document.}")
    text = read_document(doc_path)
    assert "sample RTF document" in text
    assert "{\\rtf" not in text


def test_hwpx_is_read_without_external_application(tmp_path: Path) -> None:
    path = tmp_path / "압류" / "업무매뉴얼.hwpx"
    path.parent.mkdir()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Contents/section0.xml", '<root xmlns:h="urn:test"><h:t>압류 업무 절차</h:t></root>')
    outcome = load_documents(tmp_path)
    assert outcome.documents[0].text == "압류 업무 절차"
    assert outcome.documents[0].category == "압류"
    assert outcome.documents[0].document_type == "업무매뉴얼"


def test_pptx_is_read_without_external_application(tmp_path: Path) -> None:
    path = tmp_path / "매뉴얼" / "징수 업무.pptx"
    path.parent.mkdir()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", '<root xmlns:a="urn:test"><a:t>징수 업무 안내</a:t></root>')
    outcome = load_documents(tmp_path)
    assert "슬라이드 1" in outcome.documents[0].text
    assert "징수 업무 안내" in outcome.documents[0].text


def test_pdf_pages_keep_page_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "법령" / "지방세징수법.pdf"
    path.parent.mkdir()
    path.write_bytes(b"pdf")

    class FakeReader:
        pages = [SimpleNamespace(extract_text=lambda: "제1페이지"), SimpleNamespace(extract_text=lambda: "제2페이지")]

        def __init__(self, _path: str) -> None:
            pass

    monkeypatch.setattr(document_module, "PdfReader", FakeReader)
    outcome = load_documents(tmp_path)
    assert [document.page for document in outcome.documents] == [1, 2]
    assert all(document.category == "법령" for document in outcome.documents)
    assert all(document.document_type == "법령" for document in outcome.documents)


def test_failed_document_does_not_stop_other_documents(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("정상 문서", encoding="utf-8")
    (tmp_path / "bad.hwpx").write_bytes(b"not a zip")
    outcome = load_documents(tmp_path)
    assert outcome.processed_files == 1
    assert outcome.failed_files == ["bad.hwpx"]


def test_document_personal_information_is_masked(tmp_path: Path) -> None:
    (tmp_path / "개인정보.txt").write_text("연락처 010-1234-5678, 이메일 test@example.com", encoding="utf-8")
    outcome = load_documents(tmp_path)
    assert "010-****-5678" in outcome.documents[0].text
    assert "t******@example.com" in outcome.documents[0].text
    assert outcome.pii_counts == {"전화번호": 1, "이메일 주소": 1}


def test_split_text_respects_size_and_overlap() -> None:
    text = " ".join(f"문장-{index}" for index in range(500))
    chunks = split_text(text, chunk_size=200, overlap=30)
    assert len(chunks) > 2
    assert all(0 < len(chunk) <= 200 for chunk in chunks)


@pytest.mark.parametrize(("chunk_size", "overlap"), [(0, 0), (100, -1), (100, 100), (100, 101)])
def test_split_text_rejects_invalid_settings(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split_text("테스트 문서", chunk_size=chunk_size, overlap=overlap)
