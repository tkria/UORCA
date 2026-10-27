"""Guard: no GUI string says "p-value" without saying which one.

The explorer thresholds on limma's Benjamini-Hochberg adjusted p-value
(``adj.P.Val``). A label that just says "p-value" lets a reader assume a raw
p-value. Every string literal under ``uorca/gui/`` that mentions a p-value
must also say "adjusted", "adj" or "raw" (plan 2026-10-27-02, section 3C).

Docstrings are developer documentation and are skipped, except in the MCP
server, whose tool docstrings are the descriptions the LLM reads.
"""

import ast
import re
from pathlib import Path

import pytest

GUI_ROOT = Path(__file__).resolve().parents[2] / "uorca" / "gui"

PVALUE = re.compile(r"p[- ]?value", re.IGNORECASE)
QUALIFIED = re.compile(r"adjusted|adj|raw", re.IGNORECASE)

# Internal identifiers (dict keys, widget keys, CLI flags, column names of the
# MCP long table) that contain "pvalue" but are never shown as a label. They
# are removed from a literal before it is checked. The MCP field ``pvalue`` is
# renamed in a later plan; until then it holds adj.P.Val.
INTERNAL_IDENTIFIERS = (
    "heatmap_combined_pvalue_threshold",
    "pvalue_threshold",
    "pvalue_thresh",
    "pvalue",
)
_INTERNAL = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(INTERNAL_IDENTIFIERS) + r")(?![A-Za-z])"
)

# Packages whose docstrings are user- or LLM-facing text.
DOCSTRINGS_CHECKED_IN = ("mcp_server",)


def _is_unqualified(literal: str) -> bool:
    """True when the literal mentions a p-value but not which kind."""
    text = _INTERNAL.sub("", literal)
    return bool(PVALUE.search(text)) and not QUALIFIED.search(text)


def _docstring_nodes(tree: ast.AST) -> set:
    nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                nodes.add(id(body[0].value))
    return nodes


def _violations(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    check_docstrings = any(part in DOCSTRINGS_CHECKED_IN for part in path.parts)
    docstrings = set() if check_docstrings else _docstring_nodes(tree)
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in docstrings:
            continue
        if _is_unqualified(node.value):
            rel = path.relative_to(GUI_ROOT.parent.parent)
            found.append(f"{rel}:{node.lineno}: {node.value[:80]!r}")
    return found


@pytest.mark.unit
def test_every_gui_p_value_label_says_adjusted_or_raw():
    files = sorted(GUI_ROOT.rglob("*.py"))
    assert files, f"no Python files found under {GUI_ROOT}"
    violations = [v for f in files for v in _violations(f)]
    assert not violations, (
        "These string literals mention a p-value without saying 'adjusted' or 'raw':\n"
        + "\n".join(violations)
    )


@pytest.mark.unit
def test_guard_catches_an_unqualified_label(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text('label = "P-value threshold"\nok = "Adjusted p-value (BH)"\nkey = "pvalue_thresh"\n')
    tree = ast.parse(probe.read_text())
    texts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    flagged = [t for t in texts if _is_unqualified(t)]
    assert flagged == ["P-value threshold"]
