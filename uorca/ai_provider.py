"""
Centralised AI provider configuration for UORCA.

Creates pydantic-ai model instances from configuration. Supports:

- ``openai`` (default). Returned as an ``"openai:<model>"`` string that pydantic-ai
  resolves itself, exactly as before.
- ``bedrock`` (AWS Bedrock, via ``BedrockConverseModel``), with region, profile and
  model prefix settings.
- A pydantic-ai model string ``"<provider>:<model>"`` for the hosted APIs
  ``anthropic``, ``google-gla`` (Gemini API), ``google-vertex``, ``groq``,
  ``mistral``, ``cohere`` and ``openai``.
- ``azure`` (Azure OpenAI): endpoint, deployment name and API version.
- ``openai_compatible``: any server that speaks the OpenAI chat-completions API
  (vLLM, Ollama, LM Studio, OpenRouter, Together, Fireworks, DeepSeek, ...), set by a
  base URL, a model name and, optionally, the name of the environment variable that
  holds its key.

Configuration priority for the provider:
  1. Environment variable UORCA_AI_PROVIDER
  2. ~/.uorca/config.yaml  ai_provider.provider
  3. Default: "openai"

Example config.yaml:
  ai_provider:
    provider: bedrock   # or "openai", "azure", "openai_compatible",
                        # or a model string such as "anthropic:claude-sonnet-4-5"
    bedrock:
      model: anthropic.claude-sonnet-4-5-20250929-v1:0
      model_prefix: "au."
      region: ap-southeast-2
      profile: my-aws-profile
    openai:
      model: gpt-5-mini
    anthropic:            # used when provider is the bare name "anthropic"
      model: claude-sonnet-4-5
    azure:
      endpoint: https://my-resource.openai.azure.com
      deployment: my-gpt-deployment
      api_version: "2024-10-21"
    openai_compatible:
      base_url: http://localhost:11434/v1
      model: qwen3:8b
      api_key_env: OLLAMA_API_KEY   # optional; omit for a server without a key
    google-vertex:
      model: gemini-2.5-flash
      project: my-gcp-project
      location: us-central1

Before the first call, :func:`check_provider_credentials` confirms that the selected
provider has its key or credentials, and raises :class:`MissingCredentialsError` naming
the environment variable when it does not.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import httpx
    from pydantic_ai.models import Model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".uorca" / "config.yaml"

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_BEDROCK_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_BEDROCK_REGION = "ap-southeast-2"
DEFAULT_BEDROCK_PROFILE: str | None = None
DEFAULT_BEDROCK_MODEL_PREFIX = "au."
DEFAULT_VERTEX_LOCATION = "us-central1"

# Environment variable names
UORCA_AI_PROVIDER = "UORCA_AI_PROVIDER"
UORCA_OPENAI_MODEL = "UORCA_OPENAI_MODEL"
UORCA_BEDROCK_MODEL = "UORCA_BEDROCK_MODEL"
UORCA_BEDROCK_REGION = "UORCA_BEDROCK_REGION"
UORCA_BEDROCK_PROFILE = "UORCA_BEDROCK_PROFILE"
UORCA_BEDROCK_MODEL_PREFIX = "UORCA_BEDROCK_MODEL_PREFIX"
UORCA_AZURE_DEPLOYMENT = "UORCA_AZURE_DEPLOYMENT"
UORCA_OPENAI_COMPATIBLE_BASE_URL = "UORCA_OPENAI_COMPATIBLE_BASE_URL"
UORCA_OPENAI_COMPATIBLE_MODEL = "UORCA_OPENAI_COMPATIBLE_MODEL"
UORCA_OPENAI_COMPATIBLE_API_KEY_ENV = "UORCA_OPENAI_COMPATIBLE_API_KEY_ENV"

# Provider-standard variables (the names pydantic-ai and the vendor SDKs read).
AZURE_OPENAI_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_API_KEY = "AZURE_OPENAI_API_KEY"
OPENAI_API_VERSION = "OPENAI_API_VERSION"
GOOGLE_CLOUD_PROJECT = "GOOGLE_CLOUD_PROJECT"
GOOGLE_CLOUD_LOCATION = "GOOGLE_CLOUD_LOCATION"

# Hosted providers selected by a "<provider>:<model>" string, and the environment
# variables that hold their key, in the order the installed code reads them. For
# google-gla, pydantic-ai 0.4.2 (providers/google.py) and google-genai 1.25.0
# (_api_client.py) both use GOOGLE_API_KEY first and GEMINI_API_KEY second.
API_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google-gla": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "cohere": ("CO_API_KEY",),
}
STRING_PROVIDERS = (*API_KEY_ENV_VARS, "google-vertex")
NAMED_PROVIDERS = ("bedrock", "azure", "openai_compatible")
SUPPORTED_PROVIDERS = (*STRING_PROVIDERS, *NAMED_PROVIDERS)

# Placeholder key for an OpenAI-compatible server that needs none (Ollama, a local
# vLLM without --api-key). It is passed explicitly so the OpenAI SDK never falls back
# to OPENAI_API_KEY and sends a real OpenAI key to a third-party server.
NO_KEY_PLACEHOLDER = "uorca-no-key"


class ProviderConfigError(ValueError):
    """The AI provider configuration is missing a value or names an unknown provider."""


class MissingCredentialsError(ProviderConfigError):
    """The selected AI provider has no key or credentials in the environment.

    Attributes:
        env_vars: The variable(s) to set, when a single key variable fixes it; empty
            when the fix is a login or a credentials file (Bedrock, Vertex AI).
    """

    def __init__(self, message: str, env_vars: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.env_vars = tuple(env_vars)


@dataclass(frozen=True)
class ModelSpec:
    """The resolved provider choice, before any client is built.

    Attributes:
        provider: One of :data:`SUPPORTED_PROVIDERS`.
        model: Model name as the provider expects it. For ``bedrock`` this is the full
            model ID including any prefix; for ``azure`` it is the deployment name.
        region: AWS region (``bedrock``).
        profile: AWS profile (``bedrock``).
        endpoint: Azure resource endpoint (``azure``).
        api_version: Azure OpenAI API version (``azure``).
        base_url: Server base URL (``openai_compatible``).
        api_key_env: Name of the environment variable holding the key
            (``openai_compatible``; ``None`` means the server needs no key).
        project: Google Cloud project (``google-vertex``).
        location: Google Cloud location (``google-vertex``).
    """

    provider: str
    model: str
    region: str | None = None
    profile: str | None = None
    endpoint: str | None = None
    api_version: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    project: str | None = None
    location: str | None = None

    @property
    def display_name(self) -> str:
        """``"<provider>:<model>"``, as shown in logs and the GUI."""
        return f"{self.provider}:{self.model}"


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


def _require(value: str | None, what: str, config_key: str, env_var: str | None = None) -> str:
    """Return *value*, or raise naming where it should have been set."""
    if value:
        return value
    where = f"`{config_key}` in {CONFIG_PATH}"
    if env_var:
        where += f" or the {env_var} environment variable"
    raise ProviderConfigError(f"The AI provider needs {what}. Set {where}.")


# ---------------------------------------------------------------------------
# Resolution: configuration -> ModelSpec
# ---------------------------------------------------------------------------

def _configured_provider(config: dict[str, Any]) -> str:
    provider = _get_config_value(
        config, "ai_provider", "provider",
        env_var=UORCA_AI_PROVIDER,
        default="openai",
    )
    return (provider or "openai").strip()


def _openai_model_name(config: dict[str, Any]) -> str:
    model = _get_config_value(
        config, "ai_provider", "openai", "model",
        env_var=UORCA_OPENAI_MODEL,
        default=DEFAULT_OPENAI_MODEL,
    ) or DEFAULT_OPENAI_MODEL
    return model.removeprefix("openai:")


def _bedrock_spec(config: dict[str, Any], model_id: str | None = None) -> ModelSpec:
    """Bedrock settings, unchanged from the original two-provider module.

    *model_id* comes from a ``"bedrock:<id>"`` string; it is used as given, without the
    configured prefix, because the string already names the full ID.
    """
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
    if model_id is None:
        base_id = _get_config_value(
            config, "ai_provider", "bedrock", "model",
            env_var=UORCA_BEDROCK_MODEL,
            default=DEFAULT_BEDROCK_MODEL,
        )
        model_prefix = _get_config_value(
            config, "ai_provider", "bedrock", "model_prefix",
            env_var=UORCA_BEDROCK_MODEL_PREFIX,
            default=DEFAULT_BEDROCK_MODEL_PREFIX,
        )
        model_id = f"{model_prefix}{base_id}" if model_prefix else base_id
    return ModelSpec(
        provider="bedrock",
        model=_require(model_id, "a Bedrock model ID", "ai_provider.bedrock.model"),
        region=region,
        profile=profile,
    )


def _azure_spec(config: dict[str, Any], deployment: str | None = None) -> ModelSpec:
    endpoint = _get_config_value(
        config, "ai_provider", "azure", "endpoint", env_var=AZURE_OPENAI_ENDPOINT,
    )
    api_version = _get_config_value(
        config, "ai_provider", "azure", "api_version", env_var=OPENAI_API_VERSION,
    )
    deployment = deployment or _get_config_value(
        config, "ai_provider", "azure", "deployment", env_var=UORCA_AZURE_DEPLOYMENT,
    )
    return ModelSpec(
        provider="azure",
        model=_require(
            deployment, "an Azure OpenAI deployment name",
            "ai_provider.azure.deployment", UORCA_AZURE_DEPLOYMENT,
        ),
        endpoint=_require(
            endpoint, "an Azure OpenAI endpoint",
            "ai_provider.azure.endpoint", AZURE_OPENAI_ENDPOINT,
        ),
        api_version=_require(
            api_version, "an Azure OpenAI API version",
            "ai_provider.azure.api_version", OPENAI_API_VERSION,
        ),
    )


def _openai_compatible_spec(config: dict[str, Any], model: str | None = None) -> ModelSpec:
    base_url = _get_config_value(
        config, "ai_provider", "openai_compatible", "base_url",
        env_var=UORCA_OPENAI_COMPATIBLE_BASE_URL,
    )
    model = model or _get_config_value(
        config, "ai_provider", "openai_compatible", "model",
        env_var=UORCA_OPENAI_COMPATIBLE_MODEL,
    )
    api_key_env = _get_config_value(
        config, "ai_provider", "openai_compatible", "api_key_env",
        env_var=UORCA_OPENAI_COMPATIBLE_API_KEY_ENV,
    )
    return ModelSpec(
        provider="openai_compatible",
        model=_require(
            model, "a model name for the OpenAI-compatible server",
            "ai_provider.openai_compatible.model", UORCA_OPENAI_COMPATIBLE_MODEL,
        ),
        base_url=_require(
            base_url, "a base URL for the OpenAI-compatible server",
            "ai_provider.openai_compatible.base_url", UORCA_OPENAI_COMPATIBLE_BASE_URL,
        ),
        api_key_env=api_key_env or None,
    )


def resolve_model_spec(config: dict[str, Any] | None = None) -> ModelSpec:
    """Resolve the configured provider and model without building any client.

    Args:
        config: Parsed config dict; defaults to :func:`_load_ai_config`.

    Raises:
        ProviderConfigError: The provider is unknown, or a required setting (model,
            endpoint, deployment, base URL) is missing. The message names the setting.
    """
    if config is None:
        config = _load_ai_config()
    raw = _configured_provider(config)

    if ":" in raw:
        provider, model = raw.split(":", 1)
        provider, model = provider.strip().lower(), model.strip()
        if not model:
            raise ProviderConfigError(
                f"AI provider {raw!r} has no model name after the ':'. "
                f"Use '<provider>:<model>', for example 'anthropic:claude-sonnet-4-5'."
            )
    else:
        provider, model = raw.lower(), None

    if provider == "openai":
        return ModelSpec(provider="openai", model=model or _openai_model_name(config))
    if provider == "bedrock":
        return _bedrock_spec(config, model)
    if provider == "azure":
        return _azure_spec(config, model)
    if provider == "openai_compatible":
        return _openai_compatible_spec(config, model)
    if provider in STRING_PROVIDERS:
        model = model or _get_config_value(config, "ai_provider", provider, "model")
        model = _require(
            model, f"a model name for provider {provider!r}",
            f"ai_provider.{provider}.model (or set provider to '{provider}:<model>')",
        )
        if provider == "google-vertex":
            return ModelSpec(
                provider=provider,
                model=model,
                project=_get_config_value(
                    config, "ai_provider", provider, "project", env_var=GOOGLE_CLOUD_PROJECT,
                ),
                location=_get_config_value(
                    config, "ai_provider", provider, "location",
                    env_var=GOOGLE_CLOUD_LOCATION, default=DEFAULT_VERTEX_LOCATION,
                ),
            )
        return ModelSpec(provider=provider, model=model)

    raise ProviderConfigError(
        f"Unknown AI provider {raw!r} (from {UORCA_AI_PROVIDER} or ai_provider.provider "
        f"in {CONFIG_PATH}). Use one of: {', '.join(SUPPORTED_PROVIDERS)}, "
        f"or a model string '<provider>:<model>'."
    )


# ---------------------------------------------------------------------------
# Credential check
# ---------------------------------------------------------------------------

def redact_url(url: str | None) -> str:
    """*url* with any ``user:password@`` replaced by ``***@``, for messages and logs."""
    if not url:
        return ""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit(parts._replace(netloc=f"***@{host}"))


def _first_set(names: tuple[str, ...]) -> str | None:
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    return None


def _check_bedrock_credentials(spec: ModelSpec) -> None:
    import boto3
    from botocore.exceptions import BotoCoreError

    try:
        session = boto3.Session(profile_name=spec.profile, region_name=spec.region)
        credentials = session.get_credentials()
    except BotoCoreError as exc:  # e.g. ProfileNotFound
        raise MissingCredentialsError(
            f"AWS credentials for Bedrock could not be loaded: {exc}. Check "
            f"ai_provider.bedrock.profile / {UORCA_BEDROCK_PROFILE}, then run "
            f"'aws sso login --profile <profile>'."
        ) from exc
    if credentials is None:
        raise MissingCredentialsError(
            "No AWS credentials were found for Bedrock. Set AWS_PROFILE (or "
            f"ai_provider.bedrock.profile / {UORCA_BEDROCK_PROFILE}) and run "
            "'aws sso login', or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        )


def _check_vertex_credentials(spec: ModelSpec) -> None:
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    try:
        _, detected_project = google.auth.default()
    except DefaultCredentialsError as exc:
        raise MissingCredentialsError(
            "No Google Cloud credentials were found for Vertex AI. Set "
            "GOOGLE_APPLICATION_CREDENTIALS to a service-account file, or run "
            "'gcloud auth application-default login'."
        ) from exc
    if not (spec.project or detected_project):
        raise MissingCredentialsError(
            f"No Google Cloud project is set for Vertex AI. Set {GOOGLE_CLOUD_PROJECT} "
            f"or ai_provider.google-vertex.project in {CONFIG_PATH}."
        )


def check_provider_credentials(spec: ModelSpec | None = None) -> ModelSpec:
    """Check that the selected provider has its key or credentials.

    Call this before the first LLM call, so a missing key stops the run at once with a
    message that names the variable, instead of failing deep inside a step.

    It reads only environment variables (and, for ``bedrock`` and ``google-vertex``,
    the local AWS / Google credential chain). It makes no network call.

    Args:
        spec: The resolved spec; defaults to :func:`resolve_model_spec`.

    Returns:
        The spec that was checked.

    Raises:
        ProviderConfigError: The configuration itself is incomplete.
        MissingCredentialsError: The key or credentials are missing. The message names
            the environment variable to set.
    """
    if spec is None:
        spec = resolve_model_spec()

    if spec.provider in API_KEY_ENV_VARS:
        names = API_KEY_ENV_VARS[spec.provider]
        if not _first_set(names):
            alternatives = " or ".join(names)
            raise MissingCredentialsError(
                f"The AI provider is {spec.display_name!r}, but {alternatives} is not set. "
                f"Add it to your .env file or export it, and try again.",
                env_vars=names[:1],
            )
    elif spec.provider == "azure":
        if not os.environ.get(AZURE_OPENAI_API_KEY):
            raise MissingCredentialsError(
                f"The AI provider is Azure OpenAI, but {AZURE_OPENAI_API_KEY} is not set. "
                f"Add it to your .env file or export it, and try again.",
                env_vars=(AZURE_OPENAI_API_KEY,),
            )
    elif spec.provider == "openai_compatible":
        if spec.api_key_env and not os.environ.get(spec.api_key_env):
            raise MissingCredentialsError(
                f"The OpenAI-compatible server at {redact_url(spec.base_url)} is configured to use the "
                f"key in {spec.api_key_env}, but {spec.api_key_env} is not set.",
                env_vars=(spec.api_key_env,),
            )
    elif spec.provider == "bedrock":
        _check_bedrock_credentials(spec)
    elif spec.provider == "google-vertex":
        _check_vertex_credentials(spec)
    else:  # pragma: no cover - resolve_model_spec only returns supported providers
        raise ProviderConfigError(f"No credential check for provider {spec.provider!r}.")
    return spec


# ---------------------------------------------------------------------------
# Pipeline containers
# ---------------------------------------------------------------------------

# AWS variables a Bedrock job may need. All are passed by name only.
AWS_ENV_VARS = (
    "AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_REGION", "AWS_DEFAULT_REGION",
)


@dataclass(frozen=True)
class ContainerEnv:
    """What a pipeline container needs to resolve the same model as the host.

    Attributes:
        settings: Non-secret settings, passed with their values. With only these set
            (and no ``~/.uorca/config.yaml``), :func:`resolve_model_spec` inside the
            container returns the host's spec.
        secret_names: Names of the variables that hold keys or credentials. They are
            passed **by name only**, so a value never reaches a command line, a
            generated script or a log; the container runtime copies each value from
            the caller's environment.
        needs_aws_config: True for Bedrock: mount ``~/.aws`` read-only, so profiles and
            AWS SSO tokens work inside the container.
    """

    settings: dict[str, str]
    secret_names: tuple[str, ...]
    needs_aws_config: bool = False


ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _url_for_display(url: str) -> str:
    """*url* with credentials, query string and fragment removed (safe to print)."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    host = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _check_container_url(url: str, what: str) -> None:
    """Refuse a URL that would carry a secret into a container setting.

    The URL is passed to the container by value, so it must hold no credentials:
    no ``user:password@``, no query string (``?api-key=...``) and no fragment.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if "@" in parts.netloc or parts.query or parts.fragment:
        raise ProviderConfigError(
            f"The {what} {_url_for_display(url)} has credentials, a query string or a "
            "fragment. Pipeline containers receive it as a plain setting, so it must "
            "hold no secret: remove them, and put the key in the key variable instead."
        )


def _container_unsupported(spec: ModelSpec) -> str | None:
    """Why *spec* works on this machine but cannot work inside a pipeline container."""
    from urllib.parse import urlsplit

    if spec.provider == "google-vertex":
        return (
            "Vertex AI is not supported for `uorca run` yet: its credentials (the "
            "GOOGLE_APPLICATION_CREDENTIALS file or the gcloud application-default "
            "login) are not mounted into the pipeline containers. Use it for "
            "`uorca identify`, or choose another provider for pipeline runs."
        )
    if spec.provider == "openai_compatible":
        host = (urlsplit(spec.base_url or "").hostname or "").lower()
        if host in _LOCAL_HOSTS:
            return (
                f"The OpenAI-compatible server is at {_url_for_display(spec.base_url or '')}, "
                f"but '{host}' inside a pipeline container is the container itself, not "
                "this machine. For `uorca run local` (Docker Desktop) use "
                "http://host.docker.internal:<port>/v1; for `uorca run slurm` use a host "
                "name the compute nodes can reach."
            )
    if spec.provider == "bedrock":
        defaults = {"AWS_CONFIG_FILE": "config", "AWS_SHARED_CREDENTIALS_FILE": "credentials"}
        for name, default in defaults.items():
            value = os.environ.get(name)
            if value and Path(value).expanduser() != Path.home() / ".aws" / default:
                return (
                    f"{name} is set, but pipeline containers only see the default ~/.aws "
                    f"directory. Unset {name} and keep the AWS profile in ~/.aws, or "
                    "choose another provider for pipeline runs."
                )
        import boto3

        from botocore.exceptions import BotoCoreError

        try:
            session = boto3.Session(profile_name=spec.profile, region_name=spec.region)
            scoped = session._session.get_scoped_config()  # noqa: SLF001
        except BotoCoreError:
            # e.g. ProfileNotFound: check_provider_credentials reports it by name.
            scoped = {}
        if "credential_process" in scoped:
            return (
                "The AWS profile uses credential_process, which runs a helper program "
                "that is not in the pipeline containers. Use an AWS SSO profile ('aws "
                "sso login') or access keys for pipeline runs."
            )
    return None


def container_env(spec: ModelSpec | None = None) -> ContainerEnv:
    """Describe the environment a pipeline container needs for *spec*.

    Args:
        spec: The resolved spec; defaults to :func:`resolve_model_spec`.

    Raises:
        ProviderConfigError: The setting cannot be carried into a container safely or
            would not work there: a base URL or endpoint holding credentials, a query
            string or a fragment; a key variable name that is not a plain identifier;
            Vertex AI; a localhost server; or an AWS setup the container cannot see.
    """
    if spec is None:
        spec = resolve_model_spec()
    reason = _container_unsupported(spec)
    if reason:
        raise ProviderConfigError(reason)
    provider = spec.provider
    settings: dict[str, str] = {}
    secrets: tuple[str, ...] = ()

    if provider in API_KEY_ENV_VARS:
        settings[UORCA_AI_PROVIDER] = spec.display_name
        secrets = API_KEY_ENV_VARS[provider]
    elif provider == "bedrock":
        settings[UORCA_AI_PROVIDER] = spec.display_name  # full model ID, prefix included
        if spec.region:
            settings[UORCA_BEDROCK_REGION] = spec.region
        if spec.profile:
            settings[UORCA_BEDROCK_PROFILE] = spec.profile
        secrets = AWS_ENV_VARS
    elif provider == "azure":
        _check_container_url(spec.endpoint or "", "Azure OpenAI endpoint")
        settings.update({
            UORCA_AI_PROVIDER: "azure",
            UORCA_AZURE_DEPLOYMENT: spec.model,
            AZURE_OPENAI_ENDPOINT: spec.endpoint or "",
            OPENAI_API_VERSION: spec.api_version or "",
        })
        secrets = (AZURE_OPENAI_API_KEY,)
    elif provider == "openai_compatible":
        _check_container_url(spec.base_url or "", "OpenAI-compatible base URL")
        settings.update({
            UORCA_AI_PROVIDER: "openai_compatible",
            UORCA_OPENAI_COMPATIBLE_BASE_URL: spec.base_url or "",
            UORCA_OPENAI_COMPATIBLE_MODEL: spec.model,
        })
        if spec.api_key_env:
            if not ENV_NAME_RE.match(spec.api_key_env):
                raise ProviderConfigError(
                    f"{UORCA_OPENAI_COMPATIBLE_API_KEY_ENV} must name an environment "
                    "variable (letters, digits and '_', not starting with a digit)."
                )
            settings[UORCA_OPENAI_COMPATIBLE_API_KEY_ENV] = spec.api_key_env
            secrets = (spec.api_key_env,)
    else:  # pragma: no cover - resolve_model_spec only returns supported providers
        raise ProviderConfigError(f"No container settings for provider {provider!r}.")

    return ContainerEnv(
        settings=settings, secret_names=secrets, needs_aws_config=provider == "bedrock",
    )


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _http_client(transport: httpx.AsyncBaseTransport | None) -> httpx.AsyncClient | None:
    """A client over *transport* (tests), or ``None`` for pydantic-ai's shared client."""
    if transport is None:
        return None
    import httpx

    return httpx.AsyncClient(transport=transport)


def create_model(
    spec: ModelSpec,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    bedrock_client: Any = None,
) -> Model:
    """Build the pydantic-ai model object for *spec*.

    The key is read from the provider's standard environment variable. Call
    :func:`check_provider_credentials` first for a clear error when it is missing.

    Args:
        spec: The resolved provider choice.
        transport: An httpx transport for every HTTP provider. ``None`` (production)
            uses pydantic-ai's shared client. Tests pass ``httpx.MockTransport``.
        bedrock_client: A ready ``bedrock-runtime`` boto3 client (``bedrock`` only).
            ``None`` builds one from the spec's profile and region.
    """
    http_client = _http_client(transport)
    provider = spec.provider

    if provider in ("openai", "openai_compatible", "azure"):
        from pydantic_ai.models.openai import OpenAIModel

        if provider == "openai":
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIModel(
                spec.model,
                provider=OpenAIProvider(
                    api_key=_first_set(API_KEY_ENV_VARS["openai"]), http_client=http_client,
                ),
            )
        if provider == "openai_compatible":
            from pydantic_ai.providers.openai import OpenAIProvider

            api_key = os.environ.get(spec.api_key_env) if spec.api_key_env else None
            return OpenAIModel(
                spec.model,
                provider=OpenAIProvider(
                    base_url=spec.base_url,
                    api_key=api_key or NO_KEY_PLACEHOLDER,
                    http_client=http_client,
                ),
            )
        from pydantic_ai.providers.azure import AzureProvider

        return OpenAIModel(
            spec.model,
            provider=AzureProvider(
                azure_endpoint=spec.endpoint,
                api_version=spec.api_version,
                api_key=os.environ.get(AZURE_OPENAI_API_KEY),
                http_client=http_client,
            ),
        )

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(
            spec.model,
            provider=AnthropicProvider(
                api_key=_first_set(API_KEY_ENV_VARS["anthropic"]), http_client=http_client,
            ),
        )

    if provider in ("google-gla", "google-vertex"):
        # pydantic-ai 0.4.2 maps both strings to GoogleModel, which uses the google-genai
        # SDK. That SDK takes no httpx client, only arguments for the one it builds, so a
        # test transport goes in through http_options.async_client_args.
        from google import genai
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        http_options: dict[str, Any] = {}
        if transport is not None:
            http_options["async_client_args"] = {"transport": transport}
        if provider == "google-gla":
            client = genai.Client(
                api_key=_first_set(API_KEY_ENV_VARS["google-gla"]),
                http_options=http_options or None,  # type: ignore[arg-type]
            )
        else:
            client = genai.Client(
                vertexai=True,
                project=spec.project,
                location=spec.location,
                http_options=http_options or None,  # type: ignore[arg-type]
            )
        return GoogleModel(spec.model, provider=GoogleProvider(client=client))

    if provider == "groq":
        from pydantic_ai.models.groq import GroqModel
        from pydantic_ai.providers.groq import GroqProvider

        return GroqModel(
            spec.model,
            provider=GroqProvider(
                api_key=_first_set(API_KEY_ENV_VARS["groq"]), http_client=http_client,
            ),
        )

    if provider == "mistral":
        from pydantic_ai.models.mistral import MistralModel
        from pydantic_ai.providers.mistral import MistralProvider

        return MistralModel(
            spec.model,
            provider=MistralProvider(
                api_key=_first_set(API_KEY_ENV_VARS["mistral"]), http_client=http_client,
            ),
        )

    if provider == "cohere":
        try:
            from pydantic_ai.models.cohere import CohereModel
        except ImportError as exc:
            # pydantic-ai 0.4.2 imports cohere.TextAssistantMessageContentItem, which
            # cohere 5.16 renamed; the locked cohere is 5.16.1. pydantic-ai's own message
            # ("Please install cohere") is wrong here, so say what is actually broken.
            import importlib.metadata as md

            raise ProviderConfigError(
                "The Cohere provider cannot be loaded: pydantic-ai "
                f"{md.version('pydantic-ai-slim')} is not compatible with the installed "
                f"cohere {md.version('cohere')} ({exc.__cause__ or exc}). Choose another "
                "provider, or pin a cohere version that this pydantic-ai supports."
            ) from exc
        from pydantic_ai.providers.cohere import CohereProvider

        return CohereModel(
            spec.model,
            provider=CohereProvider(
                api_key=_first_set(API_KEY_ENV_VARS["cohere"]), http_client=http_client,
            ),
        )

    if provider == "bedrock":
        from pydantic_ai.models.bedrock import BedrockConverseModel
        from pydantic_ai.providers.bedrock import BedrockProvider

        if bedrock_client is None:
            import boto3

            session = boto3.Session(profile_name=spec.profile, region_name=spec.region)
            bedrock_client = session.client("bedrock-runtime", region_name=spec.region)
        return BedrockConverseModel(
            spec.model, provider=BedrockProvider(bedrock_client=bedrock_client),
        )

    raise ProviderConfigError(f"Cannot build a model for provider {provider!r}.")


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

    For OpenAI the return value is an ``"openai:<model>"`` string (unchanged).
    For Bedrock it is a ``BedrockConverseModel`` instance (unchanged). For every other
    provider it is a pydantic-ai ``Model`` instance; for those, the credentials are
    checked first and a missing key raises :class:`MissingCredentialsError`.

    The result is cached; call ``get_model.cache_clear()`` to force reload.
    """
    config = _load_ai_config()
    spec = resolve_model_spec(config)

    if spec.provider == "openai":
        return f"openai:{spec.model}"
    if spec.provider == "bedrock":
        # No credential check here: building the client was never gated on it, and the
        # AWS SSO gate covers it. Callers may call check_provider_credentials() first.
        return create_model(spec)

    check_provider_credentials(spec)
    return create_model(spec)


def get_model_name() -> str:
    """Return a human-readable name for the active model (for logging).

    Does **not** use the cache — reads config fresh each call. Returns
    ``"<provider>:<model>"``; for Azure the model part is the deployment name.
    """
    return resolve_model_spec(_load_ai_config()).display_name


def save_ai_config(
    provider: str,
    openai_model: str | None = None,
    bedrock_config: dict[str, Any] | None = None,
    provider_config: dict[str, Any] | None = None,
) -> None:
    """Write AI provider settings to ``~/.uorca/config.yaml`` and clear caches.

    Merges with any existing config so other keys are preserved.

    Args:
        provider: A provider name (``"openai"``, ``"bedrock"``, ``"azure"``,
            ``"openai_compatible"``, ...) or a ``"<provider>:<model>"`` string.
        openai_model: Model name for OpenAI (e.g. ``"gpt-5-mini"``).
        bedrock_config: Dict with optional keys ``model``, ``model_prefix``,
            ``region``, ``profile``.
        provider_config: Settings merged into ``ai_provider.<provider>`` for the
            other providers, e.g. ``{"base_url": ..., "model": ...}`` for
            ``openai_compatible``. Ignored when *provider* is a model string.
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

    if provider_config and ":" not in provider:
        ai_section.setdefault(provider, {}).update(provider_config)

    with open(path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False)

    # Invalidate caches so next call picks up new config
    _load_ai_config.cache_clear()
    get_model.cache_clear()
