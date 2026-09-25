"""
Centralised AI provider configuration for UORCA.

Creates pydantic-ai model instances from configuration. Supports:
- OpenAI (default, via "openai:<model>" string that pydantic-ai resolves)
- AWS Bedrock (via BedrockConverseModel)

Configuration priority:
  1. Environment variable UORCA_AI_PROVIDER
  2. ~/.uorca/config.yaml  ai_provider.provider
  3. Default: "openai"

Example config.yaml:
  ai_provider:
    provider: bedrock   # or "openai"
    bedrock:
      model: anthropic.claude-sonnet-4-5-20250929-v1:0
      model_prefix: "au."
      region: ap-southeast-2
      profile: my-aws-profile
    openai:
      model: gpt-5.6-terra
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".uorca" / "config.yaml"

DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
DEFAULT_BEDROCK_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_BEDROCK_REGION = "ap-southeast-2"
DEFAULT_BEDROCK_PROFILE: str | None = None
DEFAULT_BEDROCK_MODEL_PREFIX = "au."

# Environment variable names
UORCA_AI_PROVIDER = "UORCA_AI_PROVIDER"
UORCA_OPENAI_MODEL = "UORCA_OPENAI_MODEL"
UORCA_BEDROCK_MODEL = "UORCA_BEDROCK_MODEL"
UORCA_BEDROCK_REGION = "UORCA_BEDROCK_REGION"
UORCA_BEDROCK_PROFILE = "UORCA_BEDROCK_PROFILE"
UORCA_BEDROCK_MODEL_PREFIX = "UORCA_BEDROCK_MODEL_PREFIX"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_ai_config() -> dict[str, Any]:
    """Load UORCA configuration from CONFIG_PATH.

    Returns an empty dict if the file does not exist or is empty.
    The result is cached; call _load_ai_config.cache_clear() to invalidate.
    """
    path = CONFIG_PATH  # module-level name — monkeypatching updates this
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_config_value(
    config: dict[str, Any],
    *keys: str,
    env_var: str | None = None,
    default: str | None = None,
) -> str | None:
    """Navigate a nested dict, with optional environment-variable override.

    Args:
        config: Top-level config dict.
        *keys: Sequence of keys to traverse (e.g. "ai_provider", "bedrock", "model").
        env_var: If provided, check this env var first; its value wins if set.
        default: Value to return when the key path is absent.

    Returns:
        The resolved string value, or *default*.
    """
    if env_var and (val := os.environ.get(env_var)):
        return val

    current: Any = config
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default

    return current if current is not None else default


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _create_openai_model(config: dict[str, Any]) -> str:
    """Return an ``openai:<model>`` string for pydantic-ai.

    pydantic-ai resolves the string directly to an OpenAI model instance.
    """
    model = _get_config_value(
        config, "ai_provider", "openai", "model",
        env_var=UORCA_OPENAI_MODEL,
        default=DEFAULT_OPENAI_MODEL,
    )
    if not model.startswith("openai:"):  # type: ignore[union-attr]
        model = f"openai:{model}"
    return model  # type: ignore[return-value]


def _create_bedrock_model(config: dict[str, Any]):
    """Create and return a BedrockConverseModel instance."""
    # Lazy import so the module loads even when boto3 is absent
    import boto3
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

    model_id = _get_config_value(
        config, "ai_provider", "bedrock", "model",
        env_var=UORCA_BEDROCK_MODEL,
        default=DEFAULT_BEDROCK_MODEL,
    )
    region = _get_config_value(
        config, "ai_provider", "bedrock", "region",
        env_var=UORCA_BEDROCK_REGION,
        default=DEFAULT_BEDROCK_REGION,
    )
    profile = _get_config_value(
        config, "ai_provider", "bedrock", "profile",
        env_var=UORCA_BEDROCK_PROFILE,
        default=DEFAULT_BEDROCK_PROFILE,
    )
    model_prefix = _get_config_value(
        config, "ai_provider", "bedrock", "model_prefix",
        env_var=UORCA_BEDROCK_MODEL_PREFIX,
        default=DEFAULT_BEDROCK_MODEL_PREFIX,
    )

    full_model_id = f"{model_prefix}{model_id}" if model_prefix else model_id

    session = boto3.Session(profile_name=profile, region_name=region)
    bedrock_client = session.client("bedrock-runtime", region_name=region)

    return BedrockConverseModel(
        full_model_id,
        provider=BedrockProvider(bedrock_client=bedrock_client),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_model():
    """Return the configured pydantic-ai model.

    Priority order:
      1. ``UORCA_AI_PROVIDER`` environment variable
      2. ``~/.uorca/config.yaml``  →  ``ai_provider.provider``
      3. Default: ``"openai"``

    For OpenAI the return value is an ``"openai:<model>"`` string.
    For Bedrock the return value is a ``BedrockConverseModel`` instance.

    The result is cached; call ``get_model.cache_clear()`` to force reload.
    """
    config = _load_ai_config()
    provider = _get_config_value(
        config, "ai_provider", "provider",
        env_var=UORCA_AI_PROVIDER,
        default="openai",
    )

    if provider == "bedrock":
        return _create_bedrock_model(config)
    else:
        return _create_openai_model(config)


def get_model_name() -> str:
    """Return a human-readable name for the active model (for logging).

    Does **not** use the cache — reads config fresh each call.
    """
    config = _load_ai_config()
    provider = _get_config_value(
        config, "ai_provider", "provider",
        env_var=UORCA_AI_PROVIDER,
        default="openai",
    )

    if provider == "bedrock":
        model_id = _get_config_value(
            config, "ai_provider", "bedrock", "model",
            env_var=UORCA_BEDROCK_MODEL,
            default=DEFAULT_BEDROCK_MODEL,
        )
        prefix = _get_config_value(
            config, "ai_provider", "bedrock", "model_prefix",
            env_var=UORCA_BEDROCK_MODEL_PREFIX,
            default=DEFAULT_BEDROCK_MODEL_PREFIX,
        )
        full_id = f"{prefix}{model_id}" if prefix else model_id
        return f"bedrock:{full_id}"
    else:
        model = _get_config_value(
            config, "ai_provider", "openai", "model",
            env_var=UORCA_OPENAI_MODEL,
            default=DEFAULT_OPENAI_MODEL,
        )
        if not model.startswith("openai:"):  # type: ignore[union-attr]
            model = f"openai:{model}"
        return model  # type: ignore[return-value]


def save_ai_config(
    provider: str,
    openai_model: str | None = None,
    bedrock_config: dict[str, Any] | None = None,
) -> None:
    """Write AI provider settings to ``~/.uorca/config.yaml`` and clear caches.

    Merges with any existing config so other keys are preserved.

    Args:
        provider: ``"openai"`` or ``"bedrock"``.
        openai_model: Model name for OpenAI (e.g. ``"gpt-5.6-terra"``).
        bedrock_config: Dict with optional keys ``model``, ``model_prefix``,
            ``region``, ``profile``.
    """
    path = CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config to avoid clobbering unrelated keys
    existing: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            existing = yaml.safe_load(f) or {}

    ai_section: dict[str, Any] = existing.setdefault("ai_provider", {})
    ai_section["provider"] = provider

    if openai_model is not None:
        ai_section.setdefault("openai", {})["model"] = openai_model

    if bedrock_config:
        bedrock_section = ai_section.setdefault("bedrock", {})
        bedrock_section.update(bedrock_config)

    with open(path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False)

    # Invalidate caches so next call picks up new config
    _load_ai_config.cache_clear()
    get_model.cache_clear()
