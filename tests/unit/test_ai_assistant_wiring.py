"""Unit tests for AI assistant wiring in the UORCA Explorer.

Covers three defects that make the AI Assistant tab unusable or under-powered:

1. ``agent_factory.create_uorca_agent`` builds the MCP server script path from a
   directory name that does not exist, so the stdio subprocess dies at connect
   time with ``McpError: Connection closed``.
2. ``ai_assistant_tab._execute_ai_analysis`` references ``concurrent`` in an
   ``except`` clause while only importing it inside one branch, so evaluating the
   handler raises ``UnboundLocalError`` and replaces the real failure.
3. The per-run analysis prompt tells the agent it has "four tools" and lists only
   four, so the two remaining MCP tools are never called even though the system
   prompt describes six and instructs the agent to use them.
"""

import inspect
from pathlib import Path

import pytest


def _registered_tool_names():
    """Tool names the MCP server actually exposes."""
    from uorca.gui.mcp_server import server_core

    return sorted(
        name
        for name, obj in vars(server_core).items()
        if inspect.iscoroutinefunction(obj) and not name.startswith("_")
    )


@pytest.mark.unit
def test_mcp_server_script_path_exists(monkeypatch):
    """The script handed to MCPServerStdio must be a real file on disk."""
    from uorca.gui.ai import agent_factory

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")

    captured = {}

    class FakeMCPServer:
        def __init__(self, command, args, env=None, timeout=None):
            captured["args"] = list(args)

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(agent_factory, "MCPServerStdio", FakeMCPServer)
    monkeypatch.setattr(agent_factory, "Agent", FakeAgent)

    create = getattr(
        agent_factory.create_uorca_agent, "__wrapped__", agent_factory.create_uorca_agent
    )
    create("")

    assert captured, "create_uorca_agent never constructed an MCP server"

    script_args = [a for a in captured["args"] if a.endswith("server_core.py")]
    assert script_args, f"no server script in MCP args: {captured['args']}"

    script = Path(script_args[0])
    assert script.exists(), f"MCP server script does not exist: {script}"


@pytest.mark.unit
def test_execute_ai_analysis_reports_original_error(monkeypatch):
    """A failure inside the analysis must surface its own message, not be masked."""
    from uorca.gui.components import ai_assistant_tab as tab

    errors = []

    class Stopped(Exception):
        """Stand-in for streamlit's control-flow stop."""

    def fake_stop():
        raise Stopped()

    monkeypatch.setattr(tab.st, "error", lambda msg: errors.append(str(msg)))
    monkeypatch.setattr(tab.st, "stop", fake_stop)

    class ExplodingAgent:
        def run_mcp_servers(self):
            raise RuntimeError("MCP sentinel failure: connection closed")

    with pytest.raises(Stopped):
        tab._execute_ai_analysis(ExplodingAgent(), "analyse these contrasts")

    assert errors, "the failure was never reported to the user"
    message = errors[-1]
    assert "MCP sentinel failure" in message, f"original error lost, got: {message}"
    assert "concurrent" not in message, f"error was masked by import bug: {message}"


@pytest.mark.unit
@pytest.mark.parametrize("enhanced", [True, False])
def test_analysis_prompt_offers_every_registered_tool(enhanced):
    """The agent must be told about every tool the MCP server exposes."""
    from uorca.gui.components import ai_assistant_tab as tab

    contrasts = [{"analysis_id": "GSE1", "contrast_id": "treated_vs_control"}]
    prompt = tab._build_analysis_prompt(
        research_query="drivers of differentiation",
        selected_contrast_dicts=contrasts,
        enhanced=enhanced,
    )

    missing = [name for name in _registered_tool_names() if name not in prompt]
    assert not missing, f"prompt never mentions registered tools: {missing}"

    assert "four tools" not in prompt.lower(), (
        "prompt still claims the agent has four tools"
    )


# Phrases that push the agent toward a single prescribed pass instead of
# exploring the toolbox until it has enough evidence.
PRESCRIPTIVE_PHRASES = (
    "four tools",
    "sparingly",
    "only for final validation",
)


@pytest.mark.unit
@pytest.mark.parametrize("enhanced", [True, False])
def test_analysis_prompt_invites_iterative_tool_use(enhanced):
    """The prompt must offer a toolbox to explore, not a fixed one-pass recipe."""
    from uorca.gui.components import ai_assistant_tab as tab
    from uorca.gui.ai.config_loader import get_ai_agent_config

    prompt = tab._build_analysis_prompt(
        research_query="drivers of differentiation",
        selected_contrast_dicts=[{"analysis_id": "GSE1", "contrast_id": "t_vs_c"}],
        enhanced=enhanced,
    )
    lowered = prompt.lower()

    found = [p for p in PRESCRIPTIVE_PHRASES if p in lowered]
    assert not found, f"prompt still constrains tool use: {found}"

    budget = get_ai_agent_config().request_limit
    assert str(budget) in prompt, (
        f"prompt does not tell the agent its actual call budget ({budget})"
    )

    assert "repeat" in lowered or "iterat" in lowered, (
        "prompt does not invite repeated/iterative tool calls"
    )


@pytest.mark.unit
def test_ai_config_loads_the_shipped_config_file():
    """The loader must read uorca/config/ai_assistant_config.json, not fall back to defaults."""
    import json

    from uorca.gui.ai.config_loader import AIAssistantConfigLoader, get_ai_agent_config

    assert AIAssistantConfigLoader.DEFAULT_CONFIG_PATH.exists(), (
        f"config loader points at a non-existent file: "
        f"{AIAssistantConfigLoader.DEFAULT_CONFIG_PATH}"
    )

    shipped = json.loads(
        (Path(__file__).parents[2] / "uorca/config/ai_assistant_config.json").read_text()
    )
    expected = shipped["ai_agent"]["request_limit"]
    assert get_ai_agent_config().request_limit == expected, (
        "loaded request_limit does not match the shipped config"
    )


@pytest.mark.unit
def test_system_prompt_does_not_discourage_tool_use():
    """The system prompt must not tell the agent to ration its tools."""
    from uorca.gui.ai.agent_factory import load_system_prompt

    lowered = load_system_prompt().lower()
    found = [p for p in PRESCRIPTIVE_PHRASES if p in lowered]
    assert not found, f"system prompt still rations tool use: {found}"
