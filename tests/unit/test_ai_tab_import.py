"""Importing the AI assistant tab must not load the MCP server or the ``mcp`` package.

``mcp`` pulls in pywin32 on Windows; the tab should load without it and import
it only when the agent actually runs. Each check uses a fresh interpreter, so
modules imported by other tests do not leak in.
"""

import os
import subprocess
import sys

import pytest

_PROBE = (
    "import sys\n"
    "import uorca.gui.components.ai_assistant_tab\n"
    "print(','.join(sorted(m for m in sys.modules if m == {name!r} or m.startswith({name!r} + '.'))))\n"
)


def _loaded_after_import(module_name: str) -> str:
    env = dict(os.environ, OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY", "test-key"))
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(name=module_name)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""


@pytest.mark.unit
def test_ai_tab_import_does_not_load_mcp_server():
    assert _loaded_after_import("uorca.gui.mcp_server") == ""


@pytest.mark.unit
@pytest.mark.xfail(
    reason="uorca/gui/ai/agent_factory.py imports pydantic_ai.mcp at module level; "
           "moved into create_uorca_agent() on branch fix/windows-install-and-ci (7c55e76). "
           "Remove this marker once that branch is merged.",
    strict=True,
)
def test_ai_tab_import_does_not_load_mcp_package():
    assert _loaded_after_import("mcp") == ""
