"""Deterministic checks for contrast expressions (no LLM, no R).

The contrast design step returns expressions such as ``"A - B"`` or
``"(A + B)/2 - C"``. ``RNAseq.R`` turns these into a contrast matrix with
``limma::makeContrasts``. This module checks each expression before R sees it
(plan 2026-10-27-03, section 3.2):

- Every token is a real group level, after the same renaming that R applies
  (``make.names`` on the levels, then a literal, longest-first replacement of
  each raw level in the expression; see ``RNAseq.R`` around line 411 and 524).
- The weights sum to zero and the positive weights sum to 1, so an unscaled
  ``A + B - C - D`` is rejected. A difference of two balanced contrasts,
  ``(A - B) - (C - D)``, is accepted as an interaction.
- A contrast that compares groups in different levels of a block that is
  determined by the group (for example a batch that each group lies in)
  gets a warning.

Results are structured (:class:`ContrastCheck`), never printed.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Literal, Mapping, Sequence

import pandas as pd

from .design_validation import r_make_names

WEIGHT_TOLERANCE = 1e-6
"""Tolerance for the zero-sum and unit-positive-sum weight checks."""

R_SYNTACTIC_NAME = re.compile(r"^(?:[A-Za-z]|\.(?![0-9]))[A-Za-z0-9._]*$")

ContrastShape = Literal["pairwise", "average", "interaction", "other", "invalid"]
IssueCode = Literal[
    "invalid_name",
    "duplicate_name",
    "parse_error",
    "unknown_level",
    "nonlinear",
    "constant_term",
    "no_levels",
    "weights_not_zero_sum",
    "unscaled",
    "crosses_nested_block",
]
Severity = Literal["error", "warning"]


class ContrastParseError(ValueError):
    """Raised internally when an expression cannot be parsed."""


@dataclass(frozen=True)
class ContrastIssue:
    """One problem found in a contrast."""

    code: IssueCode
    severity: Severity
    message: str


@dataclass
class ContrastCheck:
    """Result of checking one contrast."""

    name: str
    expression: str
    r_expression: str
    """The expression after R's level renaming, as makeContrasts sees it."""
    weights: dict[str, float]
    """Weight of each design level (R names); zero weights are omitted."""
    shape: ContrastShape
    issues: list[ContrastIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ContrastIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ContrastIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        """True when the contrast has no errors (warnings are allowed)."""
        return not self.errors

    def canonical_pair(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return ``(positive levels, negative levels)``, each sorted.

        Two expressions that weight the same levels the same way get the same
        pair, for example ``A - B`` and ``-B + A``. The evaluation set compares
        these pairs.
        """
        pos = tuple(sorted(k for k, w in self.weights.items() if w > 0))
        neg = tuple(sorted(k for k, w in self.weights.items() if w < 0))
        return pos, neg

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "expression": self.expression,
            "r_expression": self.r_expression,
            "weights": dict(self.weights),
            "shape": self.shape,
            "is_valid": self.is_valid,
            "issues": [
                {"code": i.code, "severity": i.severity, "message": i.message}
                for i in self.issues
            ],
        }


@dataclass
class ContrastCheckReport:
    """Checks for every contrast of one dataset."""

    level_names: dict[str, str]
    """Raw group level -> name in the design matrix (R ``make.names``)."""
    checks: list[ContrastCheck]

    @property
    def all_valid(self) -> bool:
        return all(c.is_valid for c in self.checks)

    def pairs(self) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
        """Canonical pairs of the valid contrasts."""
        return {c.canonical_pair() for c in self.checks if c.is_valid}

    def shape_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.checks:
            counts[c.shape] = counts.get(c.shape, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "all_valid": self.all_valid,
            "level_names": dict(self.level_names),
            "shape_counts": self.shape_counts(),
            "contrasts": [c.to_dict() for c in self.checks],
        }


# --- level naming -----------------------------------------------------------


def level_name_map(levels: Iterable[object]) -> dict[str, str]:
    """Map each raw group level to its R design-matrix name.

    Raises:
        ValueError: if two different levels get the same R name. R would merge
            them into one group without an error, which must not happen.
    """
    mapping: dict[str, str] = {}
    by_r_name: dict[str, str] = {}
    for raw in levels:
        raw_s = str(raw)
        r_name = r_make_names(raw_s)
        other = by_r_name.get(r_name)
        if other is not None and other != raw_s:
            raise ValueError(
                f"Group levels {other!r} and {raw_s!r} both become {r_name!r} in R; "
                "they would be merged into one group."
            )
        by_r_name[r_name] = raw_s
        mapping[raw_s] = r_name
    return mapping


def to_r_expression(expression: str, level_names: Mapping[str, str]) -> str:
    """Rewrite an expression the way ``RNAseq.R`` does before makeContrasts.

    Backticks are removed, then each raw level that differs from its R name is
    replaced literally, longest first.
    """
    expr = re.sub(r"`([^`]+)`", r"\1", expression)
    for raw in sorted(level_names, key=len, reverse=True):
        r_name = level_names[raw]
        if raw != r_name:
            expr = expr.replace(raw, r_name)
    return expr


# --- parser -----------------------------------------------------------------

_NUMBER = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_IDENT = re.compile(r"(?:[^\W\d]|\.(?!\d))[\w.]*")


def _tokenise(expr: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "+-*/()":
            tokens.append(("op", ch))
            i += 1
            continue
        m = _NUMBER.match(expr, i)
        if m and (ch.isdigit() or ch == "."):
            # A leading "." followed by a digit is a number; otherwise it may
            # start an identifier such as ".a".
            if not (ch == "." and not expr[i + 1 : i + 2].isdigit()):
                tokens.append(("num", m.group()))
                i = m.end()
                continue
        m = _IDENT.match(expr, i)
        if m:
            tokens.append(("ident", m.group()))
            i = m.end()
            continue
        raise ContrastParseError(f"unexpected character {ch!r} at position {i}")
    return tokens


# A linear combination: level weights plus a constant.
_Lin = tuple[dict[str, Fraction], Fraction]


def _add(a: _Lin, b: _Lin, sign: int) -> _Lin:
    coeffs = dict(a[0])
    for k, v in b[0].items():
        coeffs[k] = coeffs.get(k, Fraction(0)) + sign * v
    return coeffs, a[1] + sign * b[1]


def _scale(a: _Lin, factor: Fraction) -> _Lin:
    return {k: v * factor for k, v in a[0].items()}, a[1] * factor


class _Parser:
    """Recursive-descent parser for linear contrast expressions."""

    def __init__(self, tokens: list[tuple[str, str]], known: set[str]):
        self.tokens = tokens
        self.pos = 0
        self.known = known
        self.unknown: list[str] = []

    def _peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self) -> tuple[str, str]:
        tok = self._peek()
        if tok is None:
            raise ContrastParseError("unexpected end of expression")
        self.pos += 1
        return tok

    def parse(self) -> tuple[_Lin, list[_Lin]]:
        """Return the whole combination and its signed top-level terms."""
        total, terms = self._expr()
        leftover = self._peek()
        if leftover is not None:
            raise ContrastParseError(f"unexpected token {leftover[1]!r}")
        return total, terms

    def _expr(self) -> tuple[_Lin, list[_Lin]]:
        first = self._term()
        terms = [first]
        total = first
        while self._peek() in (("op", "+"), ("op", "-")):
            sign = 1 if self._next()[1] == "+" else -1
            term = self._term()
            signed = _scale(term, Fraction(sign))
            terms.append(signed)
            total = _add(total, term, sign)
        return total, terms

    def _term(self) -> _Lin:
        left = self._factor()
        while self._peek() in (("op", "*"), ("op", "/")):
            op = self._next()[1]
            right = self._factor()
            if op == "*":
                if not left[0]:
                    left = _scale(right, left[1])
                elif not right[0]:
                    left = _scale(left, right[1])
                else:
                    raise ContrastParseError("product of two levels is not linear")
            else:
                if right[0]:
                    raise ContrastParseError("division by a level is not linear")
                if right[1] == 0:
                    raise ContrastParseError("division by zero")
                left = _scale(left, 1 / right[1])
        return left

    def _factor(self) -> _Lin:
        kind, value = self._next()
        if (kind, value) == ("op", "-"):
            return _scale(self._factor(), Fraction(-1))
        if (kind, value) == ("op", "+"):
            return self._factor()
        if (kind, value) == ("op", "("):
            inner, _ = self._expr()
            if self._next() != ("op", ")"):
                raise ContrastParseError("missing closing parenthesis")
            return inner
        if kind == "num":
            return {}, Fraction(value)
        if kind == "ident":
            if value not in self.known:
                self.unknown.append(value)
            return {value: Fraction(1)}, Fraction(0)
        raise ContrastParseError(f"unexpected token {value!r}")


def _is_balanced_unit(weights: Mapping[str, Fraction]) -> bool:
    nonzero = [w for w in weights.values() if w != 0]
    if not nonzero:
        return False
    pos = sum((w for w in nonzero if w > 0), Fraction(0))
    return abs(float(sum(nonzero, Fraction(0)))) <= WEIGHT_TOLERANCE and abs(
        float(pos) - 1
    ) <= WEIGHT_TOLERANCE


def _shape(weights: Mapping[str, Fraction], terms: Sequence[_Lin]) -> ContrastShape:
    pos = [w for w in weights.values() if w > 0]
    neg = [w for w in weights.values() if w < 0]
    if _is_balanced_unit(weights):
        return "pairwise" if len(pos) == 1 and len(neg) == 1 else "average"
    # A difference of balanced contrasts, such as (A - B) - (C - D). Each
    # signed top-level term must itself be a balanced contrast.
    if len(terms) >= 2 and all(
        _is_balanced_unit(t[0]) or _is_balanced_unit(_scale(t, Fraction(-1))[0]) for t in terms
    ):
        return "interaction"
    return "other"


# --- public checks ----------------------------------------------------------


def check_contrast(
    name: str,
    expression: str,
    level_names: Mapping[str, str],
    *,
    block_of_group: Mapping[str, Mapping[str, str]] | None = None,
) -> ContrastCheck:
    """Check one contrast against the design levels.

    Args:
        name: Contrast name; becomes an R argument name in makeContrasts.
        expression: Contrast expression, as the LLM wrote it.
        level_names: Raw group level -> R name (see :func:`level_name_map`).
        block_of_group: ``{block: {raw group level: block level}}`` for blocks
            determined by the group (from
            ``DesignValidationResult.block_of_group()``).
    """
    issues: list[ContrastIssue] = []
    if not R_SYNTACTIC_NAME.match(name) or r_make_names(name) != name:
        issues.append(
            ContrastIssue(
                "invalid_name", "error",
                f"Contrast name {name!r} is not a valid R name (letters, digits, "
                "'.' and '_', starting with a letter).",
            )
        )

    r_expression = to_r_expression(expression, level_names)
    known = set(level_names.values())
    try:
        parser = _Parser(_tokenise(r_expression), known)
        (coeffs, const), terms = parser.parse()
    except ContrastParseError as exc:
        issues.append(ContrastIssue("parse_error", "error", f"Cannot parse {expression!r}: {exc}."))
        return ContrastCheck(name, expression, r_expression, {}, "invalid", issues)

    for token in dict.fromkeys(parser.unknown):
        close = difflib.get_close_matches(token, sorted(known), n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        issues.append(
            ContrastIssue("unknown_level", "error", f"Token {token!r} is not a group level.{hint}")
        )
    if const != 0:
        issues.append(
            ContrastIssue("constant_term", "error", f"Expression has a constant term ({float(const):g}).")
        )
    weights = {k: v for k, v in coeffs.items() if v != 0}
    if not weights:
        issues.append(ContrastIssue("no_levels", "error", "All level weights are zero."))

    shape: ContrastShape = "invalid"
    if weights:
        total = float(sum(weights.values(), Fraction(0)))
        pos_sum = float(sum((w for w in weights.values() if w > 0), Fraction(0)))
        shape = _shape(weights, terms)
        if abs(total) > WEIGHT_TOLERANCE:
            issues.append(
                ContrastIssue(
                    "weights_not_zero_sum", "error",
                    f"Weights sum to {total:g}, not 0, so the contrast is not a comparison.",
                )
            )
        elif shape != "interaction" and abs(pos_sum - 1) > WEIGHT_TOLERANCE:
            issues.append(
                ContrastIssue(
                    "unscaled", "error",
                    f"Positive weights sum to {pos_sum:g}, not 1. Scale averages, "
                    "for example (A + B)/2 - C; write an interaction as "
                    "(A - B) - (C - D).",
                )
            )

    if block_of_group and weights:
        r_to_raw = {r: raw for raw, r in level_names.items()}
        for block, mapping in block_of_group.items():
            block_levels = sorted(
                {mapping[r_to_raw[k]] for k in weights if k in r_to_raw and r_to_raw[k] in mapping}
            )
            if len(block_levels) > 1:
                issues.append(
                    ContrastIssue(
                        "crosses_nested_block", "warning",
                        f"Compares groups in different levels of '{block}' "
                        f"({', '.join(block_levels)}); the block effect is not "
                        "separable from the contrast.",
                    )
                )

    float_weights = {k: float(v) for k, v in weights.items()}
    if any(i.severity == "error" for i in issues):
        shape = "invalid"
    return ContrastCheck(name, expression, r_expression, float_weights, shape, issues)


def check_contrasts(
    contrasts: pd.DataFrame | Iterable[Mapping[str, str]],
    levels: Iterable[object],
    *,
    block_of_group: Mapping[str, Mapping[str, str]] | None = None,
) -> ContrastCheckReport:
    """Check every contrast of a dataset.

    Args:
        contrasts: Rows with ``name`` and ``expression`` (a DataFrame such as
            ``contrasts.csv``, or a list of dicts).
        levels: The raw group levels of the design.
        block_of_group: See :func:`check_contrast`.

    Raises:
        ValueError: if a row lacks ``name`` or ``expression``, or if two levels
            collide under R naming.
    """
    rows = (
        contrasts.to_dict(orient="records") if isinstance(contrasts, pd.DataFrame) else list(contrasts)
    )
    names = level_name_map(levels)
    checks: list[ContrastCheck] = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        if "name" not in row or "expression" not in row:
            raise ValueError(f"Contrast row {i} needs 'name' and 'expression': {row}")
        check = check_contrast(
            str(row["name"]), str(row["expression"]), names, block_of_group=block_of_group
        )
        if check.name in seen:
            check.issues.append(
                ContrastIssue("duplicate_name", "error", f"Contrast name {check.name!r} is used twice.")
            )
            check.shape = "invalid"
        seen.add(check.name)
        checks.append(check)
    return ContrastCheckReport(level_names=names, checks=checks)
