"""Assign R names to group levels and check contrast names before R runs.

``RNAseq.R`` turns each group level into a design-matrix column with R's
``make.names`` and pastes contrast names into R code. Done in R, that renaming
merges distinct levels (``EGFP+`` and ``EGFP-`` both become ``EGFP.``),
corrupts expressions by substring replacement and fails late, after Kallisto
(renaming audit 2026-10-28, failure modes F1-F11).

This module moves the naming to Python, before Kallisto:

- :func:`assign_level_names` gives every raw level a valid, unique, ASCII R
  name. A level keeps its ``make.names`` form when that form is safe, so
  existing R output keeps its names. Levels whose ``make.names`` forms collide,
  or are not plain ASCII, get an ID plus a readable slug (``g01_EGFPpos``).
  ``RNAseq.R`` then receives names on which ``make.names`` is a no-op.
- :func:`contrast_design_issues` checks contrast names (valid R names that are
  also safe folder names, unique ignoring case) and expressions (only known
  level names, linear, weights that form a comparison), using the
  deterministic parser in :mod:`uorca.core.contrast_checks`.
- :func:`build_naming_record` builds the ``metadata/naming.json`` record.

Pure: no LLM, no R, no pandas.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Iterable, Mapping, Sequence

from unidecode import unidecode

from .contrast_checks import check_contrasts, level_name_map
from .design_validation import R_RESERVED_WORDS, r_make_names

logger = logging.getLogger(__name__)

NAMING_SCHEMA_VERSION = 1

R_GROUP_COLUMN = "uorca_group"
"""Sample-table column that holds the R name of each sample's group. RNAseq.R
is given this column, so it never renames a level."""

CONTRAST_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
"""A contrast name is an R argument name and, unchanged, a folder name. Dots
are excluded because RNAseq.R maps them to ``_`` in the folder name, which
would let ``KO.vs.WT`` and ``KO_vs_WT`` share a folder."""

RESERVED_CONTRAST_NAMES = frozenset({"levels", "contrasts"})
"""Argument names of ``limma::makeContrasts``; a contrast cannot use them."""

SAFE_LEVEL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9._]*$")
"""An R level name that is ASCII, starts with a letter and so reads the same
in every locale (F18)."""

SLUG_MAX_LENGTH = 32


class NamingError(ValueError):
    """Raised when group levels or contrasts cannot be named safely.

    The error is deterministic: running Kallisto again cannot fix it. A new
    metadata selection or contrast design can.
    """


def _slug(raw: str) -> str:
    """Return a short ASCII slug that keeps the sign of ``+``/``-`` markers."""
    # unidecode maps Greek mu to "m", which would read as milli; both mu code
    # points (U+00B5 micro sign, U+03BC Greek mu) mean micro in GEO metadata.
    s = unidecode(raw.replace("\u00b5", "u").replace("\u03bc", "u"))
    s = s.replace("+", "pos")
    s = re.sub(r"-(?=$|[^A-Za-z0-9])", "neg", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:SLUG_MAX_LENGTH].rstrip("_")


def _is_safe_level_name(name: str) -> bool:
    return (
        name.isascii()
        and bool(SAFE_LEVEL_NAME.match(name))
        and name not in R_RESERVED_WORDS
        and r_make_names(name) == name
    )


def assign_level_names(raw_levels: Iterable[object]) -> dict[str, str]:
    """Map each raw group level to a valid, unique R name.

    A level keeps ``make.names(raw)`` when that name is ASCII, starts with a
    letter and no other level gets the same name. Every other level gets
    ``g<NN>_<slug>``, where ``NN`` is its 1-based position in ``raw_levels``.
    All levels of a colliding set are renamed this way, so neither keeps the
    shared name.

    Args:
        raw_levels: Distinct raw levels, in a stable order. Values are compared
            as strings (``str(value)``).

    Returns:
        ``{raw level: R name}`` in input order. The R names are distinct and
        ``make.names`` leaves each unchanged.

    Raises:
        NamingError: if two inputs have the same string form.
    """
    raws = [str(v) for v in raw_levels]
    dupes = sorted({r for r in raws if raws.count(r) > 1})
    if dupes:
        raise NamingError(f"Group levels are listed more than once: {dupes}.")

    candidates = {raw: r_make_names(raw) for raw in raws}
    by_candidate: dict[str, list[str]] = {}
    for raw, cand in candidates.items():
        by_candidate.setdefault(cand, []).append(raw)

    width = max(2, len(str(len(raws))))
    mapping: dict[str, str] = {}
    used: set[str] = set()
    needs_id: list[tuple[int, str, str]] = []
    for idx, raw in enumerate(raws, start=1):
        cand = candidates[raw]
        if len(by_candidate[cand]) == 1 and _is_safe_level_name(cand):
            mapping[raw] = cand
            used.add(cand)
        else:
            reason = (
                f"make.names gives {cand!r} for {by_candidate[cand]}"
                if len(by_candidate[cand]) > 1
                else f"make.names gives {cand!r}, which is not a plain ASCII name"
            )
            needs_id.append((idx, raw, reason))

    for idx, raw, reason in needs_id:
        slug = _slug(raw)
        name = f"g{idx:0{width}d}" + (f"_{slug}" if slug else "")
        while name in used:
            name += "_x"
        mapping[raw] = name
        used.add(name)
        logger.info("Group level %r named %r (%s)", raw, name, reason)

    ordered = {raw: mapping[raw] for raw in raws}
    # Defence in depth: the result must be identity under R's own rules.
    level_name_map(ordered.values())
    bad = [n for n in ordered.values() if r_make_names(n) != n]
    if bad or len(set(ordered.values())) != len(ordered):
        raise NamingError(f"Internal error: assigned names are not valid and unique: {ordered}")
    return ordered


def legacy_r_naming_hazards(raw_levels: Iterable[object]) -> list[str]:
    """Describe what RNAseq.R's own renaming would have done to raw levels.

    For logging only: since :func:`assign_level_names` runs first, R no longer
    renames anything. Each entry names a collision (F1, F2) or a level whose
    text occurs inside another one (F3).
    """
    raws = [str(v) for v in raw_levels]
    hazards: list[str] = []
    by_r: dict[str, list[str]] = {}
    for raw in raws:
        by_r.setdefault(r_make_names(raw), []).append(raw)
    for r_name, members in by_r.items():
        if len(members) > 1:
            hazards.append(f"make.names would merge {members} into {r_name!r}")
    changed = [r for r in raws if r_make_names(r) != r]
    for short in changed:
        for other in raws:
            if other != short and short in other:
                hazards.append(f"level {short!r} occurs inside level {other!r}")
    return hazards


def check_joined_levels(rows: Iterable[Sequence[object]], separator: str = "_") -> None:
    """Raise when joining several group columns merges different groups (F17).

    ``a_b`` + ``c`` and ``a`` + ``b_c`` both join to ``a_b_c``.

    Args:
        rows: One tuple of column values per sample.
        separator: The join separator used for the merged column.
    """
    by_joined: dict[str, set[tuple[str, ...]]] = {}
    for row in rows:
        combo = tuple(str(v) for v in row)
        by_joined.setdefault(separator.join(combo), set()).add(combo)
    clashes = {j: sorted(c) for j, c in by_joined.items() if len(c) > 1}
    if clashes:
        raise NamingError(
            f"Joining the group columns with {separator!r} merges different groups: {clashes}. "
            "Choose other columns or a single column."
        )


def contrast_name_issues(names: Sequence[str]) -> list[str]:
    """Return problems with contrast names (empty when all are usable).

    Each name must match ``^[A-Za-z][A-Za-z0-9_]*$``, be a valid R name (not a
    reserved word), not be a ``makeContrasts`` argument name, and be unique
    ignoring case (macOS file systems ignore case, so ``KO_vs_WT`` and
    ``ko_vs_wt`` would share a folder).
    """
    issues: list[str] = []
    seen: dict[str, str] = {}
    for name in names:
        if not CONTRAST_NAME.match(name):
            issues.append(
                f"Contrast name {name!r} is not allowed: use only letters, digits and "
                "underscores, starting with a letter (e.g. "
                f"{_suggest_contrast_name(name)!r})."
            )
        elif name in R_RESERVED_WORDS or r_make_names(name) != name:
            issues.append(f"Contrast name {name!r} is an R reserved word.")
        elif name in RESERVED_CONTRAST_NAMES:
            issues.append(f"Contrast name {name!r} is reserved by makeContrasts.")
        key = name.lower()
        if key in seen:
            issues.append(
                f"Contrast names {seen[key]!r} and {name!r} are the same ignoring case; "
                "each contrast needs its own name."
            )
        else:
            seen[key] = name
    return issues


def _suggest_contrast_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", unidecode(name)).strip("_")
    if not s or not s[0].isalpha():
        s = "c_" + s
    return s


def contrast_design_issues(
    contrasts: Sequence[Mapping[str, str]],
    level_names: Mapping[str, str],
) -> list[str]:
    """Return every blocking problem with a contrast design (empty when valid).

    Args:
        contrasts: Rows with ``name`` and ``expression``. Expressions must use
            the R level names (the values of ``level_names``).
        level_names: ``{raw level: R name}`` from :func:`assign_level_names`.
    """
    for i, row in enumerate(contrasts):
        if "name" not in row or "expression" not in row:
            raise ValueError(f"Contrast row {i} needs 'name' and 'expression': {dict(row)}")
    issues = contrast_name_issues([str(c["name"]) for c in contrasts])
    r_names = list(level_names.values())
    report = check_contrasts(
        [{"name": str(c["name"]), "expression": str(c["expression"])} for c in contrasts],
        r_names,
    )
    for check in report.checks:
        for issue in check.errors:
            if issue.code in ("invalid_name", "duplicate_name"):
                continue  # covered, more strictly, by contrast_name_issues
            hint = _raw_level_hint(check.expression, level_names)
            issues.append(f"Contrast {check.name!r} ({check.expression!r}): {issue.message}{hint}")
    return issues


def _raw_level_hint(expression: str, level_names: Mapping[str, str]) -> str:
    used_raw = [
        (raw, r) for raw, r in level_names.items() if raw != r and raw in expression
    ]
    if not used_raw:
        return ""
    shown = ", ".join(f"{r!r} for {raw!r}" for raw, r in used_raw)
    return f" Write the R level names, not the raw values: {shown}."


def contrast_weights(
    contrasts: Sequence[Mapping[str, str]], level_names: Mapping[str, str]
) -> dict[str, dict[str, float]]:
    """Return ``{contrast name: {R level name: weight}}`` for a valid design.

    Raises:
        NamingError: if any contrast has a blocking issue.
    """
    issues = contrast_design_issues(contrasts, level_names)
    if issues:
        raise NamingError("Invalid contrast design:\n- " + "\n- ".join(issues))
    report = check_contrasts(
        [{"name": str(c["name"]), "expression": str(c["expression"])} for c in contrasts],
        list(level_names.values()),
    )
    return {c.name: dict(c.weights) for c in report.checks}


def build_naming_record(
    *,
    group_column: str,
    source_columns: Sequence[str],
    level_names: Mapping[str, str],
    level_counts: Mapping[str, int],
    contrasts: Sequence[Mapping[str, str]],
) -> dict:
    """Return the ``metadata/naming.json`` record (schema version 1).

    The record maps each raw level to its R name, and each contrast to its R
    name and folder (the same string) and its weights.

    Raises:
        NamingError: if the contrast design is invalid.
    """
    weights = contrast_weights(contrasts, level_names)
    return {
        "schema_version": NAMING_SCHEMA_VERSION,
        "created_by": {"module": __name__, "date": date.today().isoformat()},
        "group_column": {
            "raw": group_column,
            "r_name": R_GROUP_COLUMN,
            "source_columns": list(source_columns),
        },
        "levels": [
            {"raw": raw, "r_name": r, "n_samples": int(level_counts.get(raw, 0))}
            for raw, r in level_names.items()
        ],
        "contrasts": [
            {
                "name": str(c["name"]),
                "r_name": str(c["name"]),
                "folder": str(c["name"]),
                "expression": str(c["expression"]),
                "weights": weights[str(c["name"])],
            }
            for c in contrasts
        ],
    }
