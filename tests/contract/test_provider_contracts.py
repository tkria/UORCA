"""Provider contract tests: no keys, no network (plan 7, step 6).

For each provider, UORCA's own ``uorca.ai_provider`` resolves the configuration and
builds the real pydantic-ai model with a fake key. Every HTTP request goes to an
``httpx.MockTransport``; Bedrock goes through a botocore ``Stubber``. A pydantic-ai
``Agent`` with a small structured ``output_type`` then runs against canned responses in
the provider's own wire format (see ``wire_fixtures.py`` for their sources).

Each provider is checked for:

* the request: endpoint, model name, and the structured-output tool with its schema;
* a good response parses into the output model;
* a 401, a 429 and a schema-invalid response each raise a clear error, never a default.

What this does NOT prove: that the remote service accepts the request. That needs a
real call (plan 7 step 6, "real smoke test").
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

import uorca.ai_provider as ai_provider

from . import wire_fixtures as wf

pytestmark = pytest.mark.contract

FAKE_KEY = "fake-contract-key"
OUTPUT_TOOL = "final_result"  # pydantic-ai 0.4.2 default name of the output tool


class Verdict(BaseModel):
    """Small structured output, shaped like a relevance score."""

    relevant: bool
    score: int = Field(ge=1, le=5)
    reason: str


GOOD_ARGS = {"relevant": True, "score": 4, "reason": "Matches the query."}
INVALID_ARGS = {"relevant": "perhaps", "score": 99}  # wrong type, out of range, missing field


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------

_PROVIDER_ENV = (
    "UORCA_AI_PROVIDER", "UORCA_OPENAI_MODEL", "UORCA_BEDROCK_MODEL", "UORCA_BEDROCK_REGION",
    "UORCA_BEDROCK_PROFILE", "UORCA_BEDROCK_MODEL_PREFIX", "UORCA_AZURE_DEPLOYMENT",
    "UORCA_OPENAI_COMPATIBLE_BASE_URL", "UORCA_OPENAI_COMPATIBLE_MODEL",
    "UORCA_OPENAI_COMPATIBLE_API_KEY_ENV", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "GROQ_BASE_URL",
    "MISTRAL_API_KEY", "CO_API_KEY", "CO_BASE_URL", "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION", "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION", "LOCAL_LLM_KEY", "AWS_PROFILE", "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    """No real keys, no ~/.uorca, no ~/.aws: every test starts from nothing."""
    for name in _PROVIDER_ENV:
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


# ---------------------------------------------------------------------------
# HTTP providers
# ---------------------------------------------------------------------------

@dataclass
class HttpCase:
    """One HTTP provider: how UORCA selects it, and what its wire format looks like."""

    name: str
    env: dict[str, str]                     # selects the provider, incl. the fake key
    key_var: str | None                     # the variable check_provider_credentials names
    url: str                                # expected request URL (without query string)
    model: str                              # model name the request must carry
    ok: Callable[[str, str, dict[str, Any]], dict[str, Any]]
    error: Callable[[int], dict[str, Any]]
    model_in: Callable[[httpx.Request, dict[str, Any]], str]
    tools_in: Callable[[dict[str, Any]], dict[str, dict[str, Any]]]  # name -> JSON schema
    http_error_type: type[Exception] = ModelHTTPError
    extra_checks: Callable[[httpx.Request], None] | None = None
    # Error a 200 response with a non-JSON body raises (measured with the locked SDKs).
    # The OpenAI and Anthropic SDKs return the body as a str and pydantic-ai then fails
    # with AttributeError: loud, but not a clear message.
    malformed_error: str = "AttributeError"
    query: dict[str, str] = field(default_factory=dict)


def _openai_tools(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {t["function"]["name"]: t["function"]["parameters"] for t in body.get("tools", [])}


def _anthropic_tools(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {t["name"]: t["input_schema"] for t in body.get("tools", [])}


def _google_tools(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tool in body.get("tools", []):
        for decl in tool.get("functionDeclarations", []):
            out[decl["name"]] = decl.get("parameters") or decl.get("parametersJsonSchema") or {}
    return out


def _body_model(request: httpx.Request, body: dict[str, Any]) -> str:
    return body["model"]


def _bearer(expected: str) -> Callable[[httpx.Request], None]:
    def check(request: httpx.Request) -> None:
        assert request.headers["authorization"] == f"Bearer {expected}"
    return check


def _header(name: str, expected: str) -> Callable[[httpx.Request], None]:
    def check(request: httpx.Request) -> None:
        assert request.headers.get(name) == expected, f"{name} header"
    return check


HTTP_CASES = [
    HttpCase(
        name="openai",
        env={"UORCA_AI_PROVIDER": "openai", "UORCA_OPENAI_MODEL": "gpt-5-mini",
             "OPENAI_API_KEY": FAKE_KEY},
        key_var="OPENAI_API_KEY",
        url="https://api.openai.com/v1/chat/completions",
        model="gpt-5-mini",
        ok=lambda m, t, a: wf.openai_chat(m, t, a), error=wf.openai_error,
        model_in=_body_model, tools_in=_openai_tools, extra_checks=_bearer(FAKE_KEY),
    ),
    HttpCase(
        name="azure",
        env={"UORCA_AI_PROVIDER": "azure", "UORCA_AZURE_DEPLOYMENT": "uorca-gpt",
             "AZURE_OPENAI_ENDPOINT": "https://contract.openai.azure.com",
             "OPENAI_API_VERSION": "2024-10-21", "AZURE_OPENAI_API_KEY": FAKE_KEY},
        key_var="AZURE_OPENAI_API_KEY",
        url="https://contract.openai.azure.com/openai/deployments/uorca-gpt/chat/completions",
        model="uorca-gpt",
        ok=lambda m, t, a: wf.openai_chat(m, t, a), error=wf.openai_error,
        model_in=_body_model, tools_in=_openai_tools,
        query={"api-version": "2024-10-21"},
        extra_checks=_header("api-key", FAKE_KEY),
    ),
    HttpCase(
        name="openai_compatible",
        env={"UORCA_AI_PROVIDER": "openai_compatible",
             "UORCA_OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:11434/v1",
             "UORCA_OPENAI_COMPATIBLE_MODEL": "qwen3:8b",
             "UORCA_OPENAI_COMPATIBLE_API_KEY_ENV": "LOCAL_LLM_KEY",
             "LOCAL_LLM_KEY": FAKE_KEY,
             # Set on purpose: the Bearer check proves it never reaches a third-party server.
             "OPENAI_API_KEY": "sk-openai-key-must-not-leak"},
        key_var="LOCAL_LLM_KEY",
        url="http://127.0.0.1:11434/v1/chat/completions",
        model="qwen3:8b",
        ok=lambda m, t, a: wf.openai_chat(m, t, a), error=wf.openai_error,
        model_in=_body_model, tools_in=_openai_tools, extra_checks=_bearer(FAKE_KEY),
    ),
    HttpCase(
        name="groq",
        env={"UORCA_AI_PROVIDER": "groq:llama-3.3-70b-versatile", "GROQ_API_KEY": FAKE_KEY},
        key_var="GROQ_API_KEY",
        url="https://api.groq.com/openai/v1/chat/completions",
        model="llama-3.3-70b-versatile",
        ok=lambda m, t, a: wf.openai_chat(m, t, a), error=wf.openai_error,
        model_in=_body_model, tools_in=_openai_tools, extra_checks=_bearer(FAKE_KEY),
    ),
    HttpCase(
        name="mistral",
        env={"UORCA_AI_PROVIDER": "mistral:mistral-small-latest", "MISTRAL_API_KEY": FAKE_KEY},
        key_var="MISTRAL_API_KEY",
        url="https://api.mistral.ai/v1/chat/completions",
        model="mistral-small-latest",
        ok=wf.mistral_chat, error=wf.plain_message_error, malformed_error="SDKError",
        model_in=_body_model, tools_in=_openai_tools, extra_checks=_bearer(FAKE_KEY),
    ),
    HttpCase(
        name="anthropic",
        env={"UORCA_AI_PROVIDER": "anthropic:claude-sonnet-4-5", "ANTHROPIC_API_KEY": FAKE_KEY},
        key_var="ANTHROPIC_API_KEY",
        url="https://api.anthropic.com/v1/messages",
        model="claude-sonnet-4-5",
        ok=wf.anthropic_message, error=wf.anthropic_error,
        model_in=_body_model, tools_in=_anthropic_tools,
        extra_checks=_header("x-api-key", FAKE_KEY),
    ),
    HttpCase(
        name="google-gla",
        env={"UORCA_AI_PROVIDER": "google-gla:gemini-2.5-flash", "GOOGLE_API_KEY": FAKE_KEY},
        key_var="GOOGLE_API_KEY",
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        model="gemini-2.5-flash",
        ok=wf.gemini_generate, error=wf.google_error, malformed_error="JSONDecodeError",
        model_in=lambda r, b: r.url.path.rsplit("/", 1)[-1].split(":")[0],
        tools_in=_google_tools,
        # google-genai raises its own errors.APIError; pydantic-ai 0.4.2 does not wrap it.
        http_error_type=Exception,
        extra_checks=_header("x-goog-api-key", FAKE_KEY),
    ),
    HttpCase(
        name="cohere",
        env={"UORCA_AI_PROVIDER": "cohere:command-r-plus", "CO_API_KEY": FAKE_KEY},
        key_var="CO_API_KEY",
        url="https://api.cohere.com/v2/chat",
        model="command-r-plus",
        ok=wf.cohere_chat, error=wf.plain_message_error,
        model_in=_body_model, tools_in=_openai_tools, extra_checks=_bearer(FAKE_KEY),
    ),
]


# Cases that build and run the model. Cohere cannot be imported with the locked versions
# (pydantic-ai 0.4.2 imports cohere.TextAssistantMessageContentItem, renamed in cohere
# 5.16; uv.lock has 5.16.1). UORCA raises ProviderConfigError saying so. strict=True makes
# these XPASS -> FAIL once the dependency is fixed, so the marker cannot go stale.
COHERE_BROKEN = pytest.mark.xfail(
    raises=ai_provider.ProviderConfigError, strict=True,
    reason="pydantic-ai 0.4.2 is incompatible with the locked cohere 5.16.1",
)
WIRE_CASES = [
    pytest.param(c, id=c.name, marks=COHERE_BROKEN if c.name == "cohere" else ())
    for c in HTTP_CASES
]


class Recorder:
    """An ``httpx.MockTransport`` handler that replays queued responses."""

    def __init__(self, responses: list[httpx.Response]):
        self._responses = responses
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        if not self._responses:
            raise AssertionError(f"Unexpected extra request #{len(self.requests)} to {request.url}")
        if len(self._responses) == 1:
            return self._responses[0]  # repeat the last one (SDK retries)
        return self._responses.pop(0)


def _json_response(status: int, body: dict[str, Any]) -> httpx.Response:
    # retry-after-ms keeps the Stainless SDKs' 429 back-off to 1 ms in tests.
    return httpx.Response(status, json=body, headers={"retry-after-ms": "1"})


def _build(case: HttpCase, monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]):
    for name, value in case.env.items():
        monkeypatch.setenv(name, value)
    spec = ai_provider.check_provider_credentials()  # resolves and checks, as production does
    recorder = Recorder(responses)
    model = ai_provider.create_model(spec, transport=httpx.MockTransport(recorder))
    agent = Agent(model, output_type=Verdict, system_prompt="Score the dataset.")
    return spec, agent, recorder


@pytest.mark.parametrize("case", WIRE_CASES)
def test_request_and_structured_output(case: HttpCase, monkeypatch: pytest.MonkeyPatch) -> None:
    ok = _json_response(200, case.ok(case.model, OUTPUT_TOOL, GOOD_ARGS))
    spec, agent, recorder = _build(case, monkeypatch, [ok])
    assert spec.provider == case.name
    assert spec.model == case.model

    result = agent.run_sync("Is GSE000001 relevant to fibrosis?")

    assert result.output == Verdict(**GOOD_ARGS)
    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url.copy_with(query=None)) == case.url
    for key, value in case.query.items():
        assert request.url.params.get(key) == value
    body = json.loads(request.content)
    assert case.model_in(request, body) == case.model
    tools = case.tools_in(body)
    assert OUTPUT_TOOL in tools, f"structured-output tool missing; tools sent: {sorted(tools)}"
    schema = tools[OUTPUT_TOOL]
    assert set(schema.get("properties", {})) == {"relevant", "score", "reason"}
    if case.extra_checks:
        case.extra_checks(request)


@pytest.mark.parametrize("status", [401, 429])
@pytest.mark.parametrize("case", WIRE_CASES)
def test_http_errors_raise(case: HttpCase, status: int, monkeypatch: pytest.MonkeyPatch) -> None:
    _, agent, recorder = _build(case, monkeypatch, [_json_response(status, case.error(status))])

    with pytest.raises(case.http_error_type) as excinfo:
        agent.run_sync("Is GSE000001 relevant to fibrosis?")

    err = excinfo.value
    if isinstance(err, ModelHTTPError):
        assert err.status_code == status
        assert err.model_name == case.model
    else:
        assert str(status) in str(err), f"{type(err).__name__} does not name the status: {err}"
    assert recorder.requests, "no request was sent"


@pytest.mark.parametrize("case", WIRE_CASES)
def test_schema_invalid_output_raises(case: HttpCase, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _json_response(200, case.ok(case.model, OUTPUT_TOOL, INVALID_ARGS))
    _, agent, recorder = _build(case, monkeypatch, [bad])

    with pytest.raises(UnexpectedModelBehavior, match="retries"):
        agent.run_sync("Is GSE000001 relevant to fibrosis?")

    # pydantic-ai sent the validation error back and retried before giving up.
    assert len(recorder.requests) >= 2


@pytest.mark.parametrize("case", WIRE_CASES)
def test_malformed_body_raises(case: HttpCase, monkeypatch: pytest.MonkeyPatch) -> None:
    garbage = httpx.Response(200, content=b"<html>proxy error</html>",
                             headers={"content-type": "text/html"})
    _, agent, _ = _build(case, monkeypatch, [garbage])

    with pytest.raises(Exception) as excinfo:
        agent.run_sync("Is GSE000001 relevant to fibrosis?")
    # The run fails; it never returns a default Verdict. The error is not always clear,
    # though: pinned here so a change in the SDK or pydantic-ai shows up.
    assert type(excinfo.value).__name__ == case.malformed_error, (
        f"{case.name}: {type(excinfo.value).__name__}: {excinfo.value}"
    )


@pytest.mark.parametrize("case", HTTP_CASES, ids=lambda c: c.name)
def test_missing_key_is_named(case: HttpCase, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in case.env.items():
        if name != case.key_var:
            monkeypatch.setenv(name, value)
    with pytest.raises(ai_provider.MissingCredentialsError, match=str(case.key_var)):
        ai_provider.check_provider_credentials()


# ---------------------------------------------------------------------------
# Bedrock (boto3, not httpx): botocore Stubber
# ---------------------------------------------------------------------------

BEDROCK_MODEL = "au.anthropic.claude-sonnet-4-5-20250929-v1:0"


@pytest.fixture
def bedrock(monkeypatch: pytest.MonkeyPatch):
    """A stubbed bedrock-runtime client, plus the request parameters it receives."""
    import boto3
    from botocore.stub import Stubber

    monkeypatch.setenv("UORCA_AI_PROVIDER", "bedrock")
    client = boto3.client(
        "bedrock-runtime", region_name="ap-southeast-2",
        aws_access_key_id="fake", aws_secret_access_key="fake",
    )
    seen: list[dict[str, Any]] = []
    client.meta.events.register(
        "provide-client-params.bedrock-runtime.Converse",
        lambda params, **_: seen.append(json.loads(json.dumps(params, default=str))),
    )
    stubber = Stubber(client)
    spec = ai_provider.resolve_model_spec()
    model = ai_provider.create_model(spec, bedrock_client=client)
    agent = Agent(model, output_type=Verdict, system_prompt="Score the dataset.")
    with stubber:
        yield spec, agent, stubber, seen


def test_bedrock_request_and_structured_output(bedrock) -> None:
    spec, agent, stubber, seen = bedrock
    assert spec.model == BEDROCK_MODEL  # default model with the default "au." prefix
    stubber.add_response("converse", wf.bedrock_converse(OUTPUT_TOOL, GOOD_ARGS))

    result = agent.run_sync("Is GSE000001 relevant to fibrosis?")

    assert result.output == Verdict(**GOOD_ARGS)
    stubber.assert_no_pending_responses()
    params = seen[0]
    assert params["modelId"] == BEDROCK_MODEL
    tools = {t["toolSpec"]["name"]: t["toolSpec"]["inputSchema"]["json"]
             for t in params["toolConfig"]["tools"]}
    assert set(tools[OUTPUT_TOOL]["properties"]) == {"relevant", "score", "reason"}


@pytest.mark.parametrize(
    ("status", "code"), [(403, "AccessDeniedException"), (429, "ThrottlingException")]
)
def test_bedrock_errors_raise(bedrock, status: int, code: str) -> None:
    from botocore.exceptions import ClientError

    _, agent, stubber, _ = bedrock
    stubber.add_client_error("converse", service_error_code=code, http_status_code=status)

    # pydantic-ai 0.4.2 does not wrap Bedrock errors in ModelHTTPError.
    with pytest.raises(ClientError) as excinfo:
        agent.run_sync("Is GSE000001 relevant to fibrosis?")
    assert excinfo.value.response["Error"]["Code"] == code
    assert excinfo.value.response["ResponseMetadata"]["HTTPStatusCode"] == status


def test_bedrock_schema_invalid_output_raises(bedrock) -> None:
    _, agent, stubber, seen = bedrock
    for _ in range(2):  # first answer + one output retry (pydantic-ai default)
        stubber.add_response("converse", wf.bedrock_converse(OUTPUT_TOOL, INVALID_ARGS))

    with pytest.raises(UnexpectedModelBehavior, match="retries"):
        agent.run_sync("Is GSE000001 relevant to fibrosis?")
    assert len(seen) == 2


def test_bedrock_missing_credentials_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UORCA_AI_PROVIDER", "bedrock")
    with pytest.raises(ai_provider.MissingCredentialsError, match="AWS_PROFILE"):
        ai_provider.check_provider_credentials()


# ---------------------------------------------------------------------------
# The production OpenAI path is a string; it must build the same model as above.
# ---------------------------------------------------------------------------

def test_openai_string_resolves_to_same_model_class(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai.models import infer_model
    from pydantic_ai.models.openai import OpenAIModel

    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    model = ai_provider.get_model()
    assert model == "openai:gpt-5-mini"
    inferred = infer_model(model)
    assert isinstance(inferred, OpenAIModel)
    assert inferred.model_name == "gpt-5-mini"
    assert inferred.base_url == "https://api.openai.com/v1/"


def test_cohere_fails_loudly_with_locked_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Until the dependency is fixed, selecting Cohere must say why it cannot work."""
    monkeypatch.setenv("UORCA_AI_PROVIDER", "cohere:command-r-plus")
    monkeypatch.setenv("CO_API_KEY", FAKE_KEY)
    with pytest.raises(ai_provider.ProviderConfigError, match="not compatible with the installed cohere"):
        ai_provider.get_model()
