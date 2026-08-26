import pytest

from src.privacy import detect_personal_information, mask_personal_information


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("연락처는 010-1234-5678입니다.", "전화번호"),
        ("이메일 test@example.com으로 보내주세요.", "이메일 주소"),
        ("주민번호 900101-1234567", "주민등록번호"),
        ("카드 1234-5678-9012-3456", "카드번호 형식"),
    ],
)
def test_detect_personal_information(text: str, expected: str) -> None:
    assert expected in detect_personal_information(text)


def test_general_work_question_is_allowed() -> None:
    assert detect_personal_information("압류 업무처리 절차를 알려주세요.") == []


def test_mask_personal_information_preserves_non_sensitive_context() -> None:
    masked, counts = mask_personal_information(
        "연락처 010-1234-5678, 주민번호 900101-1234567, 메일 test@example.com"
    )
    assert "010-****-5678" in masked
    assert "900101-*******" in masked
    assert "t******@example.com" in masked
    assert counts == {"전화번호": 1, "주민등록번호": 1, "이메일 주소": 1}
