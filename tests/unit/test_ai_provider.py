"""Unit tests for uorca.ai_provider."""

import pytest
import yaml
from pathlib import Path

import uorca.ai_provider as ai_provider


@pytest.fixture(autouse=True)
def clear_caches(monkeypatch, tmp_path):
    """Clear LRU caches and set default env before each test."""
    ai_provider._load_ai_config.cache_clear()
    ai_provider.get_model.cache_clear()
    # Remove provider env var to avoid test interference
    monkeypatch.delenv("UORCA_AI_PROVIDER", raising=False)
    # Point CONFIG_PATH to a non-existent tmp path by default
    monkeypatch.setattr(ai_provider, "CONFIG_PATH", tmp_path / "config.yaml")
    yield
    ai_provider._load_ai_config.cache_clear()
    ai_provider.get_model.cache_clear()


def test_get_model_defaults_to_openai(monkeypatch, tmp_path):
    """With no config file and no env var, should return an openai: string."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # CONFIG_PATH already points to a missing file via autouse fixture
    model = ai_provider.get_model()
    assert isinstance(model, str)
    assert model.startswith("openai:")


def test_get_model_default_openai_model(monkeypatch):
    """With no config file and no env var, the shipped default model is used."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert ai_provider.get_model() == "openai:gpt-5.6-terra"


def test_get_model_openai_from_config(monkeypatch, tmp_path):
    """Config specifying openai with a specific model returns that model string."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "ai_provider": {
            "provider": "openai",
            "openai": {"model": "gpt-4-turbo"},
        }
    }))
    monkeypatch.setattr(ai_provider, "CONFIG_PATH", config_path)

    model = ai_provider.get_model()
    assert model == "openai:gpt-4-turbo"


def test_get_model_env_var_overrides_config(monkeypatch, tmp_path):
    """UORCA_AI_PROVIDER=openai overrides a config file that specifies bedrock."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("UORCA_AI_PROVIDER", "openai")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "ai_provider": {
            "provider": "bedrock",
            "bedrock": {"model": "anthropic.claude-3", "region": "us-east-1"},
        }
    }))
    monkeypatch.setattr(ai_provider, "CONFIG_PATH", config_path)

    model = ai_provider.get_model()
    assert isinstance(model, str)
    assert model.startswith("openai:")


def test_load_ai_config_missing_file(monkeypatch, tmp_path):
    """_load_ai_config returns an empty dict when the config file is absent."""
    monkeypatch.setattr(ai_provider, "CONFIG_PATH", tmp_path / "does_not_exist.yaml")
    config = ai_provider._load_ai_config()
    assert config == {}


def test_load_ai_config_reads_yaml(monkeypatch, tmp_path):
    """_load_ai_config correctly parses an existing YAML config file."""
    config_path = tmp_path / "config.yaml"
    expected = {
        "ai_provider": {
            "provider": "bedrock",
            "bedrock": {
                "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                "model_prefix": "au.",
                "region": "ap-southeast-2",
                "profile": "my-aws-profile",
            },
            "openai": {"model": "gpt-5.4-mini"},
        }
    }
    config_path.write_text(yaml.dump(expected))
    monkeypatch.setattr(ai_provider, "CONFIG_PATH", config_path)

    config = ai_provider._load_ai_config()
    assert config == expected


def test_get_model_name_openai(monkeypatch, tmp_path):
    """get_model_name returns a human-readable string for the openai provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "ai_provider": {
            "provider": "openai",
            "openai": {"model": "gpt-5.4-mini"},
        }
    }))
    monkeypatch.setattr(ai_provider, "CONFIG_PATH", config_path)

    name = ai_provider.get_model_name()
    assert "openai" in name.lower() or "gpt" in name.lower()
