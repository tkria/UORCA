"""The Project Setup AI-provider section renders whatever the provider setting is.

``get_model_name()`` raises ``ProviderConfigError`` for an unknown or incomplete
provider. The page used to call it unguarded, so the page that fixes the setting crashed.
The section edits only the openai and bedrock settings and must not overwrite another
provider's setting unless the user asks to replace it.
"""

from __future__ import annotations

import pytest
import yaml

import uorca.ai_provider as ai_provider

pytestmark = pytest.mark.unit


def _section() -> None:
    from uorca.gui.pages.project_setup import _render_ai_provider_section

    _render_ai_provider_section()


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path):
    for name in ("UORCA_AI_PROVIDER", "UORCA_OPENAI_MODEL", "UORCA_AZURE_DEPLOYMENT",
                 "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ai_provider, "CONFIG_PATH", tmp_path / "config.yaml")
    ai_provider._load_ai_config.cache_clear()
    ai_provider.get_model.cache_clear()
    yield
    ai_provider._load_ai_config.cache_clear()
    ai_provider.get_model.cache_clear()


def _run():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_section, default_timeout=10).run()
    assert at.exception == [], f"page raised: {at.exception}"
    return at


def _save_buttons(at):
    return [b for b in at.button if b.label == "Save AI Config"]


def test_invalid_env_provider_shows_error_and_still_renders(monkeypatch):
    monkeypatch.setenv("UORCA_AI_PROVIDER", "typo")
    at = _run()
    assert any("Unknown AI provider 'typo'" in e.value for e in at.error)
    assert any("overrides the setting saved here" in w.value for w in at.warning)
    assert len(at.radio) == 1  # the user can still pick and save a provider
    assert not any("Current model" in c.value for c in at.caption)


def test_incomplete_saved_provider_shows_error(tmp_path):
    (tmp_path / "config.yaml").write_text(yaml.dump({"ai_provider": {"provider": "azure"}}))
    at = _run()
    assert any("Azure OpenAI deployment name" in e.value for e in at.error)


def test_other_provider_is_not_overwritten_without_consent(tmp_path):
    config = {"ai_provider": {"provider": "anthropic:claude-x"}}
    (tmp_path / "config.yaml").write_text(yaml.dump(config))
    at = _run()

    assert any("'anthropic:claude-x'" in i.value and "only the OpenAI and Bedrock" in i.value
               for i in at.info)
    assert all(b.disabled for b in _save_buttons(at))
    assert any("Current model: anthropic:claude-x" in c.value for c in at.caption)

    at.checkbox(key="ai_replace_other_provider").check().run()
    buttons = _save_buttons(at)
    assert buttons and not any(b.disabled for b in buttons)
    buttons[0].click().run()
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert saved["ai_provider"]["provider"] == "openai"


def test_openai_and_bedrock_save_as_before(tmp_path):
    at = _run()
    assert not at.info and not at.error
    buttons = _save_buttons(at)
    assert buttons and not buttons[0].disabled
    assert any("Current model: openai:gpt-5-mini" in c.value for c in at.caption)
