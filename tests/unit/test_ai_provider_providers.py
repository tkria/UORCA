"""Unit tests for the general provider setting and the credential check in uorca.ai_provider.

The backward-compatibility tests for the original openai/bedrock behaviour are in
``test_ai_provider.py`` and are unchanged. Request/response behaviour against each
provider's wire format is in ``tests/contract/``.
"""

import pytest
import yaml

import uorca.ai_provider as ai_provider
from uorca.ai_provider import MissingCredentialsError, ProviderConfigError

_ENV = (
    "UORCA_AI_PROVIDER", "UORCA_OPENAI_MODEL", "UORCA_BEDROCK_MODEL", "UORCA_BEDROCK_REGION",
    "UORCA_BEDROCK_PROFILE", "UORCA_BEDROCK_MODEL_PREFIX", "UORCA_AZURE_DEPLOYMENT",
    "UORCA_OPENAI_COMPATIBLE_BASE_URL", "UORCA_OPENAI_COMPATIBLE_MODEL",
    "UORCA_OPENAI_COMPATIBLE_API_KEY_ENV", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "CO_API_KEY",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION",
    "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "AWS_PROFILE", "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "OLLAMA_KEY",
)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws_config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws_credentials"))
    monkeypatch.setattr(ai_provider, "CONFIG_PATH", tmp_path / "config.yaml")
    ai_provider._load_ai_config.cache_clear()
    ai_provider.get_model.cache_clear()
    yield
    ai_provider._load_ai_config.cache_clear()
    ai_provider.get_model.cache_clear()


def _write_config(tmp_path, section: dict) -> None:
    (tmp_path / "config.yaml").write_text(yaml.dump({"ai_provider": section}))
    ai_provider._load_ai_config.cache_clear()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("value", "provider", "model"),
    [
        ("anthropic:claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"),
        ("google-gla:gemini-2.5-flash", "google-gla", "gemini-2.5-flash"),
        ("google-vertex:gemini-2.5-pro", "google-vertex", "gemini-2.5-pro"),
        ("groq:llama-3.3-70b-versatile", "groq", "llama-3.3-70b-versatile"),
        ("mistral:mistral-small-latest", "mistral", "mistral-small-latest"),
        ("cohere:command-r-plus", "cohere", "command-r-plus"),
        ("openai:gpt-5", "openai", "gpt-5"),
        ("bedrock:us.anthropic.claude-x-v1:0", "bedrock", "us.anthropic.claude-x-v1:0"),
        (" Anthropic:claude-x ", "anthropic", "claude-x"),
    ],
)
def test_model_string_from_env(monkeypatch, value, provider, model):
    monkeypatch.setenv("UORCA_AI_PROVIDER", value)
    spec = ai_provider.resolve_model_spec()
    assert (spec.provider, spec.model) == (provider, model)
    assert ai_provider.get_model_name() == f"{provider}:{model}"


def test_bare_provider_name_reads_its_model_from_config(tmp_path):
    _write_config(tmp_path, {"provider": "anthropic", "anthropic": {"model": "claude-x"}})
    assert ai_provider.resolve_model_spec().display_name == "anthropic:claude-x"


def test_bare_provider_without_model_names_the_setting(tmp_path):
    _write_config(tmp_path, {"provider": "groq"})
    with pytest.raises(ProviderConfigError, match=r"ai_provider\.groq\.model"):
        ai_provider.resolve_model_spec()


def test_model_string_with_empty_model_raises(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "anthropic:")
    with pytest.raises(ProviderConfigError, match="no model name"):
        ai_provider.resolve_model_spec()


def test_unknown_provider_raises_instead_of_falling_back_to_openai(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "openia")
    with pytest.raises(ProviderConfigError, match="Unknown AI provider 'openia'"):
        ai_provider.get_model()


def test_provider_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "OpenAI")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert ai_provider.get_model() == "openai:gpt-5-mini"


def test_bedrock_config_resolution_unchanged(tmp_path):
    _write_config(tmp_path, {
        "provider": "bedrock",
        "bedrock": {"model": "anthropic.claude-3", "model_prefix": "us.",
                    "region": "us-east-1", "profile": "p"},
    })
    spec = ai_provider.resolve_model_spec()
    assert (spec.model, spec.region, spec.profile) == ("us.anthropic.claude-3", "us-east-1", "p")
    assert ai_provider.get_model_name() == "bedrock:us.anthropic.claude-3"


def test_azure_from_config(tmp_path):
    _write_config(tmp_path, {"provider": "azure", "azure": {
        "endpoint": "https://r.openai.azure.com", "deployment": "dep", "api_version": "2024-10-21"}})
    spec = ai_provider.resolve_model_spec()
    assert (spec.model, spec.endpoint, spec.api_version) == (
        "dep", "https://r.openai.azure.com", "2024-10-21")


def test_azure_missing_endpoint_names_the_variable(tmp_path):
    _write_config(tmp_path, {"provider": "azure", "azure": {"deployment": "d", "api_version": "v"}})
    with pytest.raises(ProviderConfigError, match="AZURE_OPENAI_ENDPOINT"):
        ai_provider.resolve_model_spec()


def test_openai_compatible_requires_base_url(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "openai_compatible")
    monkeypatch.setenv("UORCA_OPENAI_COMPATIBLE_MODEL", "qwen3:8b")
    with pytest.raises(ProviderConfigError, match="UORCA_OPENAI_COMPATIBLE_BASE_URL"):
        ai_provider.resolve_model_spec()


# ---------------------------------------------------------------------------
# Credential check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("value", "key_var"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("anthropic:claude-x", "ANTHROPIC_API_KEY"),
        ("google-gla:gemini-x", "GOOGLE_API_KEY"),
        ("groq:llama-x", "GROQ_API_KEY"),
        ("mistral:mistral-x", "MISTRAL_API_KEY"),
        ("cohere:command-x", "CO_API_KEY"),
    ],
)
def test_missing_key_names_the_variable(monkeypatch, value, key_var):
    monkeypatch.setenv("UORCA_AI_PROVIDER", value)
    with pytest.raises(MissingCredentialsError, match=key_var):
        ai_provider.check_provider_credentials()
    monkeypatch.setenv(key_var, "fake")
    assert ai_provider.check_provider_credentials().provider == value.split(":")[0]


def test_gemini_accepts_gemini_api_key_too(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "google-gla:gemini-x")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    ai_provider.check_provider_credentials()


def test_gemini_key_order_matches_installed_sdk():
    # pydantic-ai 0.4.2 and google-genai 1.25.0 both read GOOGLE_API_KEY first.
    assert ai_provider.API_KEY_ENV_VARS["google-gla"] == ("GOOGLE_API_KEY", "GEMINI_API_KEY")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://user:s3cret@llm.example.org/v1", "https://***@llm.example.org/v1"),
        ("https://token@llm.example.org:8443/v1", "https://***@llm.example.org:8443/v1"),
        ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1"),
    ],
)
def test_redact_url(url, expected):
    assert ai_provider.redact_url(url) == expected


def test_key_error_does_not_leak_url_credentials(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "openai_compatible")
    monkeypatch.setenv("UORCA_OPENAI_COMPATIBLE_BASE_URL", "https://bob:s3cret@llm.example.org/v1")
    monkeypatch.setenv("UORCA_OPENAI_COMPATIBLE_MODEL", "m")
    monkeypatch.setenv("UORCA_OPENAI_COMPATIBLE_API_KEY_ENV", "OLLAMA_KEY")
    with pytest.raises(MissingCredentialsError) as excinfo:
        ai_provider.check_provider_credentials()
    assert "s3cret" not in str(excinfo.value)
    assert "https://***@llm.example.org/v1" in str(excinfo.value)


def test_get_model_checks_credentials_for_new_providers(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "anthropic:claude-x")
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY"):
        ai_provider.get_model()


def test_get_model_openai_still_returns_string_without_key_check():
    # Unchanged behaviour: the OpenAI string is returned even with no key; pydantic-ai
    # raises when the agent is built. Callers that want the early check call it.
    assert ai_provider.get_model() == "openai:gpt-5-mini"


def test_azure_key_missing(tmp_path):
    _write_config(tmp_path, {"provider": "azure", "azure": {
        "endpoint": "https://r.openai.azure.com", "deployment": "d", "api_version": "v"}})
    with pytest.raises(MissingCredentialsError, match="AZURE_OPENAI_API_KEY"):
        ai_provider.check_provider_credentials()


def test_openai_compatible_without_key_env_needs_no_key(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "openai_compatible")
    monkeypatch.setenv("UORCA_OPENAI_COMPATIBLE_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("UORCA_OPENAI_COMPATIBLE_MODEL", "qwen3:8b")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-be-sent")
    from pydantic_ai.models.openai import OpenAIModel

    model = ai_provider.get_model()
    assert isinstance(model, OpenAIModel)
    # The placeholder is passed explicitly so the real OpenAI key is never sent.
    assert model.client.api_key == ai_provider.NO_KEY_PLACEHOLDER
    assert model.base_url == "http://127.0.0.1:11434/v1/"


def test_openai_compatible_key_env_must_be_set(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "openai_compatible")
    monkeypatch.setenv("UORCA_OPENAI_COMPATIBLE_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("UORCA_OPENAI_COMPATIBLE_MODEL", "qwen/qwen3-8b")
    monkeypatch.setenv("UORCA_OPENAI_COMPATIBLE_API_KEY_ENV", "OLLAMA_KEY")
    with pytest.raises(MissingCredentialsError, match="OLLAMA_KEY"):
        ai_provider.check_provider_credentials()


def test_bedrock_without_credentials_is_named(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "bedrock")
    with pytest.raises(MissingCredentialsError, match="AWS_PROFILE"):
        ai_provider.check_provider_credentials()


def test_bedrock_unknown_profile_is_named(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "bedrock")
    monkeypatch.setenv("UORCA_BEDROCK_PROFILE", "no-such-profile")
    with pytest.raises(MissingCredentialsError, match="no-such-profile"):
        ai_provider.check_provider_credentials()


def test_bedrock_with_static_keys_passes(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "bedrock")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    assert ai_provider.check_provider_credentials().provider == "bedrock"


def test_vertex_without_adc_is_named(monkeypatch):
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    def no_adc(*_a, **_k):
        raise DefaultCredentialsError("none")

    monkeypatch.setattr(google.auth, "default", no_adc)
    monkeypatch.setenv("UORCA_AI_PROVIDER", "google-vertex:gemini-x")
    with pytest.raises(MissingCredentialsError, match="GOOGLE_APPLICATION_CREDENTIALS"):
        ai_provider.check_provider_credentials()


def test_vertex_without_project_is_named(monkeypatch):
    import google.auth

    monkeypatch.setattr(google.auth, "default", lambda *a, **k: (object(), None))
    monkeypatch.setenv("UORCA_AI_PROVIDER", "google-vertex:gemini-x")
    with pytest.raises(MissingCredentialsError, match="GOOGLE_CLOUD_PROJECT"):
        ai_provider.check_provider_credentials()


# ---------------------------------------------------------------------------
# save_ai_config
# ---------------------------------------------------------------------------

def test_save_ai_config_provider_config_round_trips(tmp_path):
    ai_provider.save_ai_config(
        provider="openai_compatible",
        provider_config={"base_url": "http://localhost:11434/v1", "model": "qwen3:8b"},
    )
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert saved["ai_provider"]["openai_compatible"] == {
        "base_url": "http://localhost:11434/v1", "model": "qwen3:8b"}
    assert ai_provider.get_model_name() == "openai_compatible:qwen3:8b"
