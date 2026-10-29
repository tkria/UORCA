"""``python -m uorca`` runs the CLI without the ``uorca`` script shim (``uorca.exe`` on Windows).

Some Windows application-control policies block ``.exe`` files in a user's venv but allow
``python.exe``; ``uv run python -m uorca ...`` is the documented workaround.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_python_dash_m_uorca_help_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "uorca", "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "usage: uorca" in proc.stdout
    assert "explore" in proc.stdout


def test_python_dash_m_uorca_unknown_command_fails():
    proc = subprocess.run([sys.executable, "-m", "uorca", "no-such-command"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr
