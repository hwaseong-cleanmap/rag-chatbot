import pytest

from src.privacy import detect_personal_information


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("연락처는 010-1234-5678입니다", "전화번호"),
        ("이메일 test@example.com으로 답해주세요", "이메일 주소"),
        ("주민번호 900101-1234567", "주민등록번호"),
    ],
)
def test_detect_personal_information(text: str, expected: str) -> None:
    assert expected in detect_personal_information(text)


def test_public_faq_question_is_allowed() -> None:
    assert detect_personal_information("지방세 징수포상금 지급 기준은 무엇인가요?") == []
