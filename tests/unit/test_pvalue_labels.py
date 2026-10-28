"""Guard: every mention of a p-value says which one.

The explorer thresholds on limma's Benjamini-Hochberg adjusted p-value
(``adj.P.Val``). Text that just says "p-value" lets a reader, or the LLM, assume
a raw p-value. Each p-value mention in a string literal or docstring under
``uorca/gui/`` and ``uorca/core/``, and in the AI prompt files, must say
"adjusted"/"adj" or "raw" next to it (plan 2026-10-27-02, section 3C).

A mention is any spelling of p-value (``p-value``, ``p value``, ``pvalue``,
``p_value``, ``P.Value``, ``pval``) in any case. ``adj.P.Val`` and ``padj``
are adjusted by name. A mention is qualified when "adjusted", "adj" or "raw"
(whole words) appears within ``BEFORE`` characters before it or ``AFTER``
characters after it. Each mention is checked, not each literal.
"""

import ast
import re
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

REPO = Path(__file__).resolve().parents[2]
PY_ROOTS = (REPO / "uorca" / "gui", REPO / "uorca" / "core")
TEXT_FILES_GLOB = (REPO / "uorca" / "gui" / "ai" / "prompts", "*.txt")

# One p-value mention, in any spelling and case.
MENTION = re.compile(r"\bp[-_ .]?val(?:ue)?s?\b|\bpval\w*|pvalue\w*", re.IGNORECASE)
# Mentions that are adjusted by name.
ADJUSTED_BY_NAME = re.compile(r"\badj\.p\.val\b|\bpadj\b", re.IGNORECASE)
QUALIFIER = re.compile(r"\badj(?:usted)?\b|\braw\b", re.IGNORECASE)
BEFORE = 40
AFTER = 20

# Internal identifiers (dict keys, widget keys, CLI flags, function parameters,
# column names of the MCP long table) that contain a p-value spelling but are
# never shown as text. They are removed before checking. The MCP field
# ``pvalue`` holds adj.P.Val and is renamed in a later plan.
INTERNAL_IDENTIFIERS = (
    "heatmap_combined_pvalue_threshold",
    "pvalue_threshold",
    "pvalue_thresh",
    "p_value_threshold",
    "p_value_col",
    "pval_val",
    "pvalue_value",
    "MissingAdjustedPValueError",
)
# ``pvalue`` only as a code reference: data$pvalue, df['pvalue'], df.pvalue, `pvalue`
_INTERNAL = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(INTERNAL_IDENTIFIERS) + r")(?![A-Za-z])"
    r"""|(?<=[$'"`\[.])pvalue(?=['"`\]]|\b)"""
)

# Whole literals that are the raw limma column name used as a dict/DataFrame
# key, not as text a person reads.
COLUMN_KEYS = {"P.Value", "pvalue"}


def unqualified_mentions(text: str) -> List[str]:
    """Return each p-value mention in ``text`` that does not say which kind."""
    if text in COLUMN_KEYS:
        return []
    cleaned = _INTERNAL.sub("", ADJUSTED_BY_NAME.sub("", text))
    found = []
    for m in MENTION.finditer(cleaned):
        window = cleaned[max(0, m.start() - BEFORE): m.end() + AFTER]
        if not QUALIFIER.search(window):
            found.append(cleaned[max(0, m.start() - 25): m.end() + 10])
    return found


def _python_texts(path: Path) -> Iterator[Tuple[int, str]]:
    """Yield (line, text) for every string literal and docstring.

    An f-string is checked as one text with ``{}`` for each placeholder, so a
    qualifier and a mention in different pieces still count together.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    inside_fstring = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                    inside_fstring.add(id(value))
                else:
                    parts.append("{}")
            yield node.lineno, "".join(parts)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in inside_fstring:
            yield node.lineno, node.value


def _sources() -> Iterator[Tuple[str, Iterator[Tuple[int, str]]]]:
    for root in PY_ROOTS:
        for path in sorted(root.rglob("*.py")):
            yield str(path.relative_to(REPO)), _python_texts(path)
    folder, pattern = TEXT_FILES_GLOB
    for path in sorted(folder.glob(pattern)):
        lines = path.read_text(encoding="utf-8").splitlines()
        yield str(path.relative_to(REPO)), iter(list(enumerate(lines, start=1)))


@pytest.mark.unit
def test_every_p_value_mention_says_adjusted_or_raw():
    scanned = 0
    violations = []
    for name, texts in _sources():
        scanned += 1
        for line, text in texts:
            for snippet in unqualified_mentions(text):
                violations.append(f"{name}:{line}: ...{snippet!r}...")
    assert scanned > 10, "guard scanned too few files; check the paths"
    assert not violations, (
        "These p-value mentions do not say 'adjusted' or 'raw':\n" + "\n".join(violations)
    )


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "P-value threshold",
    "pvalue < 0.05",
    "PVAL cutoff",
    "Use the P.Value column here",
    "Adjusted p-value (BH) is in the first column; the second column of the table shows the p-value",
    "radjusted p-value",
    "drawn p-value",
])
def test_guard_flags_unqualified_mentions(text):
    assert unqualified_mentions(text), text


@pytest.mark.unit
@pytest.mark.parametrize("text", [
    "Adjusted p-value (BH) threshold",
    "Raw p-value",
    "adj.P.Val < 0.05",
    "padj",
    "p-value (adjusted, BH)",
    "data['pvalue'] < 0.05",
    "data$pvalue < 0.05",
    "pvalue_thresh",
    "P.Value",
])
def test_guard_accepts_qualified_mentions(text):
    assert not unqualified_mentions(text), text
