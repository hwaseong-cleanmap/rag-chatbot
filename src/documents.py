"""TXT 및 RTF 형식의 DOC 문서 로딩과 청킹."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from striprtf.striprtf import rtf_to_text


SUPPORTED_SUFFIXES = {".txt", ".doc"}


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    text: str
    file_hash: str


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    file_hash: str
    chunk_index: int


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
        raise ValueError(
            f"지원하지 않는 구형 DOC 형식입니다: {path.name}. "
            "현재 로더는 RTF 형식의 .doc 파일을 지원합니다."
        )
    # 일부 행정문서는 RTF 내부에 ``\binN`` 형식의 이미지 원본 바이트를
    # 포함한다. 바이너리 데이터의 임의 중괄호가 RTF 파서를 방해하므로
    # 선언된 길이만큼 정확히 건너뛴 뒤 텍스트를 변환한다.
    marker = b"\\bin"
    cleaned = bytearray()
    position = 0
    while True:
        marker_start = raw.find(marker, position)
        if marker_start < 0:
            cleaned.extend(raw[position:])
            break
        length_end = raw.find(b" ", marker_start)
        if length_end < 0:
            cleaned.extend(raw[position:])
            break
        try:
            binary_length = int(raw[marker_start + len(marker) : length_end])
        except ValueError:
            cleaned.extend(raw[position : marker_start + len(marker)])
            position = marker_start + len(marker)
            continue
        binary_end = length_end + 1 + binary_length
        if binary_end > len(raw):
            raise ValueError(f"손상된 RTF 바이너리 블록입니다: {path.name}")
        cleaned.extend(raw[position:marker_start])
        cleaned.extend(b" ")
        position = binary_end

    rtf = bytes(cleaned).decode("latin-1")
    return _normalize_text(rtf_to_text(rtf, errors="ignore"))


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _read_txt(path)
    if suffix == ".doc":
        return _read_doc(path)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")


def load_documents(data_dir: Path) -> tuple[list[SourceDocument], list[str]]:
    """문서를 읽고, 내용이 완전히 같은 파일은 한 번만 반환한다."""
    documents: list[SourceDocument] = []
    skipped_duplicates: list[str] = []
    seen_text_hashes: set[str] = set()

    paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    for path in paths:
        raw = path.read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
        text = read_document(path)
        if not text:
            continue
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in seen_text_hashes:
            skipped_duplicates.append(path.name)
            continue
        seen_text_hashes.add(text_hash)
        documents.append(SourceDocument(path=path, text=text, file_hash=file_hash))

    return documents, skipped_duplicates


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap은 0 이상이고 chunk_size보다 작아야 합니다.")

    chunks: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        end = target_end
        if target_end < text_length:
            search_start = start + int(chunk_size * 0.6)
            candidates = [
                text.rfind("\n", search_start, target_end),
                text.rfind(" ", search_start, target_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        next_start = max(0, end - overlap)
        start = next_start if next_start > start else end

    return chunks


def make_chunks(
    documents: list[SourceDocument], chunk_size: int = 1200, overlap: int = 200
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        for index, text in enumerate(split_text(document.text, chunk_size, overlap)):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.file_hash[:24]}-{index:05d}",
                    text=text,
                    source=document.path.name,
                    file_hash=document.file_hash,
                    chunk_index=index,
                )
            )
    return chunks


def corpus_fingerprint(
    documents: list[SourceDocument], embedding_model: str, chunk_size: int, overlap: int
) -> str:
    values = [
        embedding_model,
        str(chunk_size),
        str(overlap),
        *(sorted(document.file_hash for document in documents)),
    ]
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()
