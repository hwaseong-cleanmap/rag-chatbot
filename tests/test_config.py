from src.config import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, Settings


def test_settings_builds_cloudflare_url_from_account_id(monkeypatch) -> None:
    for name in (
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_BASE_URL",
        "CHAT_MODEL",
        "EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env(
        {
            "CLOUDFLARE_ACCOUNT_ID": "a" * 32,
            "CLOUDFLARE_API_TOKEN": "secret-token",
        }
    )

    assert settings.base_url.endswith(f"/accounts/{'a' * 32}/ai/v1")
    assert settings.chat_model == DEFAULT_CHAT_MODEL
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    settings.validate()


def test_streamlit_secrets_override_environment(monkeypatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "environment-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "environment-token")

    settings = Settings.from_env(
        {
            "CLOUDFLARE_ACCOUNT_ID": "b" * 32,
            "CLOUDFLARE_API_TOKEN": "secret-token",
        }
    )

    assert settings.account_id == "b" * 32
    assert settings.api_token == "secret-token"


def test_settings_rejects_mismatched_account_url() -> None:
    settings = Settings(
        account_id="a" * 32,
        api_token="secret-token",
        base_url=(
            f"https://api.cloudflare.com/client/v4/accounts/{'b' * 32}/ai/v1"
        ),
    )

    try:
        settings.validate()
    except ValueError as error:
        assert "Account ID" in str(error)
    else:
        raise AssertionError("Account ID 불일치가 검증되지 않았습니다.")


def test_settings_rejects_email_as_account_id() -> None:
    settings = Settings(
        account_id="person@example.com",
        api_token="secret-token",
        base_url=(
            "https://api.cloudflare.com/client/v4/accounts/person@example.com/ai/v1"
        ),
    )

    try:
        settings.validate()
    except ValueError as error:
        assert "이메일이 아니라" in str(error)
    else:
        raise AssertionError("이메일 형식 Account ID가 거부되지 않았습니다.")
