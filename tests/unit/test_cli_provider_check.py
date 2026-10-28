"""The CLI's startup check uses the selected AI provider, not OpenAI only.

``uorca identify`` and ``uorca run`` used to require ``OPENAI_API_KEY`` whatever the
provider was. For the ``openai`` provider the check and its message are unchanged.
"""

from __future__ import annotations

import pytest

import uorca.ai_provider as ai_provider
import uorca.cli as cli

pytestmark = pytest.mark.unit

_ENV = (
    "UORCA_AI_PROVIDER", "UORCA_OPENAI_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "AWS_PROFILE", "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "UORCA_BEDROCK_PROFILE",
)


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ENTREZ_EMAIL", "test@example.org")
    monkeypatch.setenv("ENTREZ_API_KEY", "fake")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws_config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws_credentials"))
    monkeypatch.setattr(ai_provider, "CONFIG_PATH", tmp_path / "config.yaml")
    ai_provider._load_ai_config.cache_clear()
    yield
    ai_provider._load_ai_config.cache_clear()


def _exits(capsys, **kwargs) -> str:
    with pytest.raises(SystemExit) as excinfo:
        cli._check_environment_requirements(**kwargs)
    assert excinfo.value.code == 1
    return capsys.readouterr().out


@pytest.mark.parametrize("step", ["identify", "run"])
def test_openai_missing_key_message_unchanged(capsys, step):
    out = _exits(capsys, require_ai=step)
    assert "OPENAI_API_KEY - Required for AI-powered dataset identification" in out
    assert "OPENAI_API_KEY=sk-proj-your-key-here" in out
    assert "https://platform.openai.com/api-keys" in out


@pytest.mark.parametrize("step", ["identify", "run"])
def test_openai_with_key_passes(monkeypatch, step):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    cli._check_environment_requirements(require_ai=step)


def test_identify_with_anthropic_needs_anthropic_key_not_openai(monkeypatch, capsys):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "anthropic:claude-x")
    out = _exits(capsys, require_ai="identify")
    assert "ANTHROPIC_API_KEY" in out
    assert "OPENAI_API_KEY=sk-proj" not in out

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    cli._check_environment_requirements(require_ai="identify")  # no OPENAI_API_KEY needed
    assert "AI-powered analysis features will be disabled" not in capsys.readouterr().out


def test_identify_with_bedrock_checks_aws_credentials(monkeypatch, capsys):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "bedrock")
    assert "AWS_PROFILE" in _exits(capsys, require_ai="identify")

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    cli._check_environment_requirements(require_ai="identify")


@pytest.mark.parametrize("setting", ["anthropic:claude-x", "bedrock"])
def test_run_refuses_providers_the_containers_cannot_use(monkeypatch, capsys, setting):
    monkeypatch.setenv("UORCA_AI_PROVIDER", setting)
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    out = _exits(capsys, require_ai="run")
    assert "passes only OPENAI_API_KEY into the pipeline containers" in out
    assert "UORCA_AI_PROVIDER=openai" in out


def test_invalid_provider_setting_is_reported(monkeypatch, capsys):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "typo")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    assert "Unknown AI provider 'typo'" in _exits(capsys, require_ai="identify")


def test_no_ai_step_needs_no_ai_key(capsys):
    cli._check_environment_requirements()
    assert "OPENAI_API_KEY not set - AI-powered analysis features will be disabled" in (
        capsys.readouterr().out
    )
