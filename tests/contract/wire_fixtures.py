"""Canned provider responses for the provider contract tests.

Each builder returns a response body in the provider's own wire format. ``SOURCES``
records where each shape comes from and when it was written down.

The shapes were constructed on 2026-10-27, without network access, from two places that
must agree for a test to pass: the response types of the vendor SDK pinned in
``uv.lock``, and the pydantic-ai 0.4.2 parser that reads them. The documentation URL is
the provider's public reference for the same object; it was not re-fetched on that date.
Replace a shape with a recorded real response when a key becomes available (plan 7,
step 6, "real smoke test").
"""

from __future__ import annotations

import json
from typing import Any

FIXTURE_DATE = "2026-10-27"

SOURCES: dict[str, dict[str, str]] = {
    "openai": {
        "docs": "https://platform.openai.com/docs/api-reference/chat/object",
        "constructed_from": "openai 1.95.0 openai/types/chat/chat_completion.py; "
        "pydantic-ai 0.4.2 pydantic_ai/models/openai.py OpenAIModel._process_response",
    },
    "azure": {
        "docs": "https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference",
        "constructed_from": "same body as openai (AsyncAzureOpenAI shares the openai 1.95.0 types); "
        "pydantic-ai 0.4.2 pydantic_ai/providers/azure.py",
    },
    "openai_compatible": {
        "docs": "https://platform.openai.com/docs/api-reference/chat/object "
        "(vLLM, Ollama, OpenRouter etc. return this shape on /v1/chat/completions)",
        "constructed_from": "openai 1.95.0 openai/types/chat/chat_completion.py; "
        "pydantic-ai 0.4.2 pydantic_ai/models/openai.py",
    },
    "groq": {
        "docs": "https://console.groq.com/docs/api-reference#chat-create",
        "constructed_from": "groq 0.29.0 groq/types/chat/chat_completion.py; "
        "pydantic-ai 0.4.2 pydantic_ai/models/groq.py GroqModel._process_response",
    },
    "mistral": {
        "docs": "https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post",
        "constructed_from": "mistralai 1.9.2 mistralai/models/chatcompletionresponse.py; "
        "pydantic-ai 0.4.2 pydantic_ai/models/mistral.py MistralModel._process_response",
    },
    "anthropic": {
        "docs": "https://docs.anthropic.com/en/api/messages",
        "constructed_from": "anthropic 0.57.1 anthropic/types/message.py; "
        "pydantic-ai 0.4.2 pydantic_ai/models/anthropic.py AnthropicModel._process_response",
    },
    "google-gla": {
        "docs": "https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse",
        "constructed_from": "google-genai 1.25.0 google/genai/types.py GenerateContentResponse; "
        "pydantic-ai 0.4.2 pydantic_ai/models/google.py GoogleModel._process_response",
    },
    "cohere": {
        "docs": "https://docs.cohere.com/reference/chat",
        "constructed_from": "cohere 5.16.1 cohere/v2/types/v2chat_response.py; "
        "pydantic-ai 0.4.2 pydantic_ai/models/cohere.py CohereModel._process_response",
    },
    "bedrock": {
        "docs": "https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html",
        "constructed_from": "botocore 1.39.4 bedrock-runtime service model (Stubber validates the "
        "shape against it); pydantic-ai 0.4.2 pydantic_ai/models/bedrock.py",
    },
}


# ---------------------------------------------------------------------------
# Successful responses carrying one structured-output tool call
# ---------------------------------------------------------------------------

def openai_chat(model: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """OpenAI chat-completions body (also Azure, Groq and OpenAI-compatible servers)."""
    return {
        "id": "chatcmpl-contract-1",
        "object": "chat.completion",
        "created": 1761523200,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_contract_1",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59},
    }


def mistral_chat(model: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Mistral chat-completions body. Tool-call IDs are 9 alphanumeric characters."""
    return {
        "id": "cmpl-contract-1",
        "object": "chat.completion",
        "created": 1761523200,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "abc123def",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59},
    }


def anthropic_message(model: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "msg_contract_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {"type": "tool_use", "id": "toolu_contract_1", "name": tool_name, "input": args}
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 42, "output_tokens": 17},
    }


def gemini_generate(model: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": tool_name, "args": args}}],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 42,
            "candidatesTokenCount": 17,
            "totalTokenCount": 59,
        },
        "modelVersion": model,
        "responseId": "contract-1",
    }


def cohere_chat(model: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    del model  # Cohere v2 /chat does not echo the model
    return {
        "id": "contract-1",
        "finish_reason": "TOOL_CALL",
        "message": {
            "role": "assistant",
            "tool_plan": "I will return the structured answer.",
            "tool_calls": [
                {
                    "id": "final_result_contract_1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(args)},
                }
            ],
        },
        "usage": {
            "billed_units": {"input_tokens": 42, "output_tokens": 17},
            "tokens": {"input_tokens": 42, "output_tokens": 17},
        },
    }


def bedrock_converse(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "tooluse_contract_1", "name": tool_name, "input": args}}
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 42, "outputTokens": 17, "totalTokens": 59},
        "metrics": {"latencyMs": 5},
    }


# ---------------------------------------------------------------------------
# Error bodies (401 and 429)
# ---------------------------------------------------------------------------

def openai_error(status: int) -> dict[str, Any]:
    if status == 401:
        return {"error": {"message": "Incorrect API key provided: fake-key.", "type": "invalid_request_error",
                          "param": None, "code": "invalid_api_key"}}
    return {"error": {"message": "Rate limit reached for requests.", "type": "requests",
                      "param": None, "code": "rate_limit_exceeded"}}


def anthropic_error(status: int) -> dict[str, Any]:
    kind = "authentication_error" if status == 401 else "rate_limit_error"
    return {"type": "error", "error": {"type": kind, "message": f"contract {kind}"}}


def google_error(status: int) -> dict[str, Any]:
    state = "UNAUTHENTICATED" if status == 401 else "RESOURCE_EXHAUSTED"
    return {"error": {"code": status, "message": f"contract {state}", "status": state}}


def plain_message_error(status: int) -> dict[str, Any]:
    """Mistral and Cohere return a flat ``{"message": ...}`` error body."""
    return {"message": "Unauthorized" if status == 401 else "Requests rate limit exceeded"}
