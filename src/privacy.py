"""Personal-information detection and masking before external API calls."""

from __future__ import annotations

import re
from collections import Counter


_RESIDENT_NUMBER = re.compile(r"(?<!\d)(\d{6})\s*[- ]?\s*([1-4]\d{6})(?!\d)")
_PHONE = re.compile(r"(?<!\d)((?:01[016789]|0\d{1,2})[-.\s]?(\d{3,4})[-.\s]?(\d{4}))(?!\d)")
_EMAIL = re.compile(r"(?<![A-Z0-9._%+-])([A-Z0-9._%+-]+)@([A-Z0-9.-]+\.[A-Z]{2,})(?![A-Z0-9._%+-])", re.IGNORECASE)
_CARD = re.compile(r"(?<!\d)(\d{4})[- ](\d{4})[- ](\d{4})[- ](\d{4})(?!\d)")


def detect_personal_information(text: str) -> list[str]:
    """Return labels for clearly identifiable personal-information formats."""
    labels: list[str] = []
    if _RESIDENT_NUMBER.search(text):
        labels.append("주민등록번호")
    if _PHONE.search(text):
        labels.append("전화번호")
    if _EMAIL.search(text):
        labels.append("이메일 주소")
    if _CARD.search(text):
        labels.append("카드번호 형식")
    return labels


def mask_personal_information(text: str) -> tuple[str, dict[str, int]]:
    """Mask document text and return aggregate counts by type.

    Names and addresses are intentionally not guessed in this MVP because false
    positives could remove legally relevant content.
    """
    counts: Counter[str] = Counter()

    def resident(match: re.Match[str]) -> str:
        counts["주민등록번호"] += 1
        return f"{match.group(1)}-*******"

    def phone(match: re.Match[str]) -> str:
        counts["전화번호"] += 1
        digits = re.sub(r"\D", "", match.group(1))
        return f"{digits[:-8]}-****-{digits[-4:]}"

    def email(match: re.Match[str]) -> str:
        counts["이메일 주소"] += 1
        return f"{match.group(1)[0]}******@{match.group(2)}"

    def card(match: re.Match[str]) -> str:
        counts["카드번호 형식"] += 1
        return f"{match.group(1)}-****-****-{match.group(4)}"

    masked = _RESIDENT_NUMBER.sub(resident, text)
    masked = _EMAIL.sub(email, masked)
    masked = _CARD.sub(card, masked)
    masked = _PHONE.sub(phone, masked)
    return masked, dict(counts)
