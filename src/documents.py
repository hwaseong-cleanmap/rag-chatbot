"""Document loading, masking, chunking, and metadata creation."""

from __future__ import annotations

import hashlib
import re
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text

from src.privacy import mask_personal_information


SUPPORTED_SUFFIXES = {".txt", ".doc", ".pdf", ".hwpx", ".pptx"}


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    relative_path: str
    text: str
    file_hash: str
    category: str
    document_type: str
    page: int | None = None


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    relative_path: str
    file_hash: str
    chunk_index: int
    category: str
    document_type: str
    page: int | None


@dataclass(frozen=True)
class LoadOutcome:
    documents: list[SourceDocument]
    skipped_duplicates: list[str]
    failed_files: list[str]
    pii_counts: dict[str, int]
    category_counts: dict[str, int]
    total_files: int
    processed_files: int


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _read_txt(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949", "utf-16"):
        try:
            return _normalize_text(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    return _normalize_text(raw.decode("utf-8", errors="replace"))


def _read_doc(path: Path) -> str:
    raw = path.read_bytes()
    if not raw.lstrip().startswith(b"{\\rtf"):
        raise ValueError("RTF 형식이 아닌 구형 DOC 파일입니다")
    # Some public RTF files contain embedded binary data (\\binN). Remove
    # exactly that byte range before handing the remaining RTF to striprtf.
    cleaned = bytearray()
    position = 0
    while True:
        marker_start = raw.find(b"\\bin", position)
        if marker_start < 0:
            cleaned.extend(raw[position:])
            break
        length_end = raw.find(b" ", marker_start)
        if length_end < 0:
            cleaned.extend(raw[position:])
            break
        try:
            binary_length = int(raw[marker_start + 4 : length_end])
        except ValueError:
            cleaned.extend(raw[position : marker_start + 4])
            position = marker_start + 4
            continue
        binary_end = length_end + 1 + binary_length
        if binary_end > len(raw):
            raise ValueError("손상된 RTF 바이너리 블록입니다")
        cleaned.extend(raw[position:marker_start])
        cleaned.extend(b" ")
        position = binary_end
    return _normalize_text(rtf_to_text(bytes(cleaned).decode("latin-1"), errors="ignore"))


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = _normalize_text(page.extract_text() or "")
        if text:
            pages.append((number, text))
    if not pages:
        raise ValueError("텍스트를 추출할 수 없는 PDF입니다. 스캔 문서 또는 이미지 PDF인지 확인하세요.")
    return pages


def _read_hwpx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            section_files = sorted(
                name for name in archive.namelist()
                if name.lower().startswith("contents/") and name.lower().endswith(".xml")
            )
            if not section_files:
                raise ValueError("HWPX 본문 XML을 찾을 수 없습니다")
            parts: list[str] = []
            for name in section_files:
                root = ElementTree.fromstring(archive.read(name))
                parts.extend(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
    except (zipfile.BadZipFile, ElementTree.ParseError) as error:
        raise ValueError("읽을 수 없는 HWPX 파일입니다") from error
    text = _normalize_text("\n".join(parts))
    if not text:
        raise ValueError("HWPX에서 텍스트를 추출할 수 없습니다")
    return text


def _read_pptx(path: Path) -> str:
    """Extract slide text directly from the PPTX ZIP/XML structure."""
    try:
        with zipfile.ZipFile(path) as archive:
            slide_files = sorted(
                name for name in archive.namelist()
                if name.lower().startswith("ppt/slides/slide") and name.lower().endswith(".xml")
            )
            if not slide_files:
                raise ValueError("PPTX 슬라이드를 찾을 수 없습니다")
            slides: list[str] = []
            for index, name in enumerate(slide_files, start=1):
                root = ElementTree.fromstring(archive.read(name))
                parts = [node.text or "" for node in root.iter() if node.tag.endswith("}t")]
                text = _normalize_text("\n".join(parts))
                if text:
                    slides.append(f"[슬라이드 {index}]\n{text}")
    except (zipfile.BadZipFile, ElementTree.ParseError) as error:
        raise ValueError("읽을 수 없는 PPTX 파일입니다") from error
    text = _normalize_text("\n\n".join(slides))
    if not text:
        raise ValueError("PPTX에서 텍스트를 추출할 수 없습니다")
    return text


def read_document(path: Path) -> str:
    """Read a document; PDF text is joined for backward-compatible callers."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _read_txt(path)
    if suffix == ".doc":
        return _read_doc(path)
    if suffix == ".pdf":
        return _normalize_text("\n\n".join(text for _, text in _read_pdf(path)))
    if suffix == ".hwpx":
        return _read_hwpx(path)
    if suffix == ".pptx":
        return _read_pptx(path)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")


def _category(data_dir: Path, path: Path) -> str:
    relative = path.relative_to(data_dir)
    return relative.parts[0] if len(relative.parts) > 1 else "기타"


def _document_type(path: Path, category: str) -> str:
    name = f"{category} {path.stem}"
    if "법" in name or "조례" in name:
        return "법령"
    if "매뉴얼" in name or "업무" in name:
        return "업무매뉴얼"
    if "지침" in name or "예규" in name:
        return "지침"
    return "기타"


def load_documents(
    data_dir: Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
    paths: list[Path] | None = None,
) -> LoadOutcome:
    """Read supported files recursively without stopping on individual failures.

    ``paths`` enables incremental indexing.  Callers pass only files that were
    added or changed, so unchanged PDF/HWPX files are never opened again.
    """
    documents: list[SourceDocument] = []
    duplicates: list[str] = []
    failures: list[str] = []
    pii_counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    seen_text_hashes: set[str] = set()
    processed_files: set[str] = set()
    source_paths = paths if paths is not None else list(data_dir.rglob("*"))
    paths = sorted(path for path in source_paths if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)

    for file_number, path in enumerate(paths, start=1):
        relative_path = path.relative_to(data_dir).as_posix()
        if progress_callback:
            progress_callback(file_number, len(paths), relative_path)
        category = _category(data_dir, path)
        document_type = _document_type(path, category)
        try:
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            page_texts = _read_pdf(path) if path.suffix.lower() == ".pdf" else [(None, read_document(path))]
            file_added = False
            for page, raw_text in page_texts:
                masked_text, found = mask_personal_information(raw_text)
                pii_counts.update(found)
                if not masked_text:
                    continue
                text_hash = hashlib.sha256(masked_text.encode("utf-8")).hexdigest()
                if text_hash in seen_text_hashes:
                    duplicates.append(relative_path)
                    continue
                seen_text_hashes.add(text_hash)
                documents.append(SourceDocument(path, relative_path, masked_text, file_hash, category, document_type, page))
                file_added = True
            if file_added:
                processed_files.add(relative_path)
                categories[category] += 1
        except (OSError, ValueError):
            failures.append(relative_path)

    return LoadOutcome(documents, duplicates, failures, dict(pii_counts), dict(categories), len(paths), len(processed_files))


def source_manifest(data_dir: Path) -> list[dict[str, int | str]]:
    """Return inexpensive file metadata used to detect source changes.

    This deliberately does not open PDF/HWPX/PPTX contents.  The resulting
    manifest lets a completed index be reused immediately after an app restart.
    """
    paths = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return [
        {
            "path": path.relative_to(data_dir).as_posix(),
            "size": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]


def _split_by_size(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split prose without breaking near the start of a line where possible."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        target_end = min(start + chunk_size, len(text))
        end = target_end
        if target_end < len(text):
            boundary = max(
                text.rfind("\n", start + int(chunk_size * 0.6), target_end),
                text.rfind(" ", start + int(chunk_size * 0.6), target_end),
            )
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _heading_sections(text: str) -> list[str]:
    """Split manuals at common Korean procedure headings without losing text."""
    starts = [match.start() for match in re.finditer(
        r"(?m)^\s*(?:\d+[.)]|[가-하][.)]|[Ⅰ-Ⅹ]+[.)]|\([0-9가-하]\)|[□■◆])\s+",
        text,
    )]
    if len(starts) < 2:
        return [text]
    boundaries = [0, *starts[1:], len(text)]
    return [text[start:end].strip() for start, end in zip(boundaries, boundaries[1:]) if text[start:end].strip()]


def _pack_manual_sections(sections: list[str], target_size: int, maximum_size: int, overlap: int) -> list[str]:
    """Keep procedure headings together, packing short adjacent sections only."""
    chunks: list[str] = []
    pending = ""
    for section in sections:
        if len(section) > maximum_size:
            if pending:
                chunks.append(pending)
                pending = ""
            chunks.extend(_split_by_size(section, maximum_size, overlap))
            continue
        candidate = f"{pending}\n\n{section}".strip() if pending else section
        if pending and len(candidate) > maximum_size:
            chunks.append(pending)
            pending = section
        else:
            pending = candidate
        if len(pending) >= target_size:
            chunks.append(pending)
            pending = ""
    if pending:
        chunks.append(pending)
    return chunks


def split_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
    document_type: str = "기타",
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상이어야 합니다")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap은 0 이상이고 chunk_size보다 작아야 합니다")
    # A law article must be retrieved as one unit whenever possible.  Generic
    # length-only chunking previously split a single article's numbered items.
    article_starts = [match.start() for match in re.finditer(r"(?m)^\s*제\s*\d+\s*조(?:의\s*\d+)?", text)]
    if document_type == "법령" or len(article_starts) >= 2:
        law_maximum = min(chunk_size, 1200)
        law_overlap = min(overlap, 200)
        if len(article_starts) < 2:
            return _split_by_size(text, law_maximum, law_overlap)
        boundaries = [0, *article_starts[1:], len(text)]
        chunks: list[str] = []
        for start, end in zip(boundaries, boundaries[1:]):
            section = text[start:end].strip()
            if not section:
                continue
            if len(section) <= law_maximum:
                chunks.append(section)
            else:
                chunks.extend(_split_by_size(section, law_maximum, law_overlap))
        return chunks

    if document_type == "업무매뉴얼":
        manual_maximum = min(chunk_size, 1500)
        return _pack_manual_sections(_heading_sections(text), 800, manual_maximum, min(overlap, 250))
    return _split_by_size(text, chunk_size, overlap)


def make_chunks(documents: list[SourceDocument], chunk_size: int = 1200, overlap: int = 200) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        for index, text in enumerate(split_text(document.text, chunk_size, overlap, document.document_type)):
            page_part = f"-p{document.page:04d}" if document.page else ""
            chunks.append(DocumentChunk(f"{document.file_hash[:24]}{page_part}-{index:05d}", text, document.path.name, document.relative_path, document.file_hash, index, document.category, document.document_type, document.page))
    return chunks


def corpus_fingerprint(documents: list[SourceDocument], embedding_model: str, chunk_size: int, overlap: int) -> str:
    values = [embedding_model, str(chunk_size), str(overlap), *(sorted(f"{document.file_hash}:{document.page}" for document in documents))]
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()
