"""공개 FAQ 서비스의 고위험 개인정보 형식 감지."""

from __future__ import annotations

import re


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "주민등록번호",
        re.compile(r"(?<!\d)\d{6}\s*[- ]?\s*[1-4]\d{6}(?!\d)"),
    ),
    (
        "전화번호",
        re.compile(r"(?<!\d)(?:01[016789]|0\d{1,2})[-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"),
    ),
    (
        "이메일 주소",
        re.compile(
            r"(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"
            r"(?![A-Z0-9._%+-])",
            re.IGNORECASE,
        ),
    ),
)


def detect_personal_information(text: str) -> list[str]:
    """입력에서 확실하게 판별 가능한 개인정보 형식 이름을 반환한다."""

    return [name for name, pattern in _PATTERNS if pattern.search(text)]
