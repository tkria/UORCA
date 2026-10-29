"""Deterministic validation of a limma/edgeR design with blocking factors.

Given a sample table, the columns that define the groups of interest and the
columns proposed as blocking factors (cell line, donor, batch), this module
decides which terms can enter a ``~0 + group + <blocks>`` design and why the
others cannot. It calls no LLM and no R.

Steps (plan 2026-10-27-03, section 3.1):

1. Drop group and block columns that have one level only, logging each one.
2. For each block, cross-tabulate group x block. A block that is aliased with
   the group, or nested with it in either direction, cannot be estimated next
   to ``~0 + group`` and is dropped with a reason.
3. Build the design matrix and check that it has full column rank. A crossed
   block that still makes the matrix rank-deficient is dropped as confounded.
4. Check that the residual degrees of freedom are positive and that every
   group has a name of its own in R. A group smaller than
   ``MIN_REPLICATES_PER_GROUP`` does not raise; it is listed in
   ``small_groups`` and becomes a high review flag (owner decision pending).

The result serialises (``to_dict``) to the ``metadata/design.json`` structure.
Impossible designs raise :class:`DesignValidationError`; nothing degenerate is
returned silently.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_REPLICATES_PER_GROUP = 2
"""Groups smaller than this cannot contribute a within-group variance. They are
reported in ``small_groups`` (and flagged for review), not rejected."""

DEFAULT_GROUP_VARIABLE = "group"
"""Name of the merged group factor in the formula."""

GROUP_LEVEL_SEPARATOR = "_"
"""Separator used to join several group columns into one level.

Matches ``uorca/graph/nodes/metadata.py``, which joins the selected columns
with ``"_".join``.
"""

# R reserved words. ``make.names`` appends "." to a name that equals one.
R_RESERVED_WORDS = frozenset(
    {
        "if", "else", "repeat", "while", "function", "for", "next", "break",
        "in", "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA", "NA_integer_",
        "NA_real_", "NA_character_", "NA_complex_",
    }
)

DropReason = Literal[
    "single_level",
    "aliased_with_group",
    "group_nested_in_block",
    "block_nested_in_group",
    "implied_by_block",
    "confounded",
]
TermRole = Literal["group", "block"]


class DesignValidationError(ValueError):
    """Raised when no valid design can be built from the inputs."""


def _r_name_char(ch: str) -> bool:
    """True when R keeps ``ch`` in a name (Unicode letter, letter-number or
    decimal digit, ``.`` or ``_``). Other numbers such as ``²`` and ``½`` and
    combining marks become ``.``, as in R 4.x in a UTF-8 locale."""
    return ch in "._" or unicodedata.category(ch)[0] == "L" or unicodedata.category(ch) in ("Nd", "Nl")


def _r_name_start(s: str) -> bool:
    ch = s[0]
    if ch == ".":
        return not (len(s) > 1 and s[1] in "0123456789")
    if ch in "0123456789":
        return False
    # R accepts a leading letter, letter-number or non-ASCII decimal digit.
    return unicodedata.category(ch)[0] == "L" or unicodedata.category(ch) in ("Nd", "Nl")


def r_make_names(value: object) -> str:
    """Return what R's ``make.names(value)`` returns for one string.

    ``RNAseq.R`` applies ``make.names`` to the group levels (and ``read.csv``
    applies it to column headers), so Python checks must use the same names.

    Rules (R ``?make.names``): every character that is not a letter, a digit,
    ``.`` or ``_`` becomes ``.``; ``X`` is prepended when the name does not
    start with a letter or with a ``.`` that is not followed by a digit; a
    reserved word gets a trailing ``.``. Non-ASCII handling was checked
    against R 4.x on macOS in a UTF-8 locale (``x²`` -> ``x.``, ``½dose`` ->
    ``X.dose``, a leading Arabic-Indic digit is kept without ``X``). Other
    locales or platforms may differ.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA."
    s = str(value)
    first_ok = bool(s) and _r_name_start(s)
    cleaned = "".join(ch if _r_name_char(ch) else "." for ch in s)
    if not first_ok:
        cleaned = "X" + cleaned
    if cleaned in R_RESERVED_WORDS:
        cleaned += "."
    return cleaned


@dataclass(frozen=True)
class KeptTerm:
    """A design term that enters the formula."""

    column: str
    r_name: str
    role: TermRole
    n_levels: int
    levels: list[str]


@dataclass(frozen=True)
class DroppedTerm:
    """A requested term that was removed, with the reason."""

    column: str
    role: TermRole
    reason_code: DropReason
    reason: str
    n_levels: int
    group_to_block: dict[str, str] | None = None
    """For a block that is a function of the group (aliased, or the group is
    nested in it): the block level of each group level. Used to warn about
    contrasts that cross block levels."""


@dataclass
class DesignValidationResult:
    """Outcome of :func:`validate_design`; serialise with :meth:`to_dict`."""

    formula: str
    group_variable: str
    group_columns: list[str]
    block_columns: list[str]
    group_levels: list[str]
    group_sizes: dict[str, int]
    kept_terms: list[KeptTerm]
    dropped_terms: list[DroppedTerm]
    n_samples: int
    n_coefficients: int
    rank: int
    residual_df: int
    requested_group_columns: list[str]
    requested_block_columns: list[str]
    design_columns: list[str] = field(default_factory=list)
    small_groups: dict[str, int] = field(default_factory=dict)
    """Groups with fewer than ``MIN_REPLICATES_PER_GROUP`` samples."""
    sample_groups: list[str] = field(default_factory=list)
    """Group level of each sample, in the row order of the input table."""

    def block_of_group(self) -> dict[str, dict[str, str]]:
        """Return ``{block column: {group level: block level}}``.

        Only blocks that are determined by the group (dropped as aliased or
        as the group nested in the block) appear. A contrast between groups in
        different levels of such a block also compares those block levels.
        """
        return {
            d.column: dict(d.group_to_block)
            for d in self.dropped_terms
            if d.group_to_block is not None
        }

    def to_dict(self) -> dict:
        """Return the ``metadata/design.json`` structure."""
        return {
            "formula": self.formula,
            "group_variable": self.group_variable,
            "group_columns": list(self.group_columns),
            "block_columns": list(self.block_columns),
            "group_levels": list(self.group_levels),
            "group_sizes": dict(self.group_sizes),
            "small_groups": dict(self.small_groups),
            "design_columns": list(self.design_columns),
            "terms": {
                "kept": [asdict(t) for t in self.kept_terms],
                "dropped": [asdict(t) for t in self.dropped_terms],
            },
            "counts": {
                "n_samples": self.n_samples,
                "group_columns_before": len(self.requested_group_columns),
                "group_columns_after": len(self.group_columns),
                "block_columns_before": len(self.requested_block_columns),
                "block_columns_after": len(self.block_columns),
                "n_groups": len(self.group_levels),
                "n_coefficients": self.n_coefficients,
                "rank": self.rank,
                "residual_df": self.residual_df,
            },
        }


def string_column(samples: pd.DataFrame, column: str) -> pd.Series:
    """Return one column as strings.

    Raises:
        DesignValidationError: if the column name is duplicated, which makes
            ``samples[column]`` a DataFrame instead of one column.
    """
    values = samples[column]
    if not isinstance(values, pd.Series):
        raise DesignValidationError(f"Column name {column!r} appears more than once.")
    return values.astype(str)


def _levels(series: pd.Series) -> list[str]:
    return sorted(series.astype(str).unique().tolist())


def build_design_matrix(
    group: pd.Series, blocks: Mapping[str, pd.Series]
) -> tuple[np.ndarray, list[str]]:
    """Build the ``~0 + group + blocks`` model matrix.

    The group is one-hot coded (one column per level, no intercept). Each
    block uses treatment coding with its first sorted level as reference, as
    R's ``model.matrix`` does by default.
    """
    group_str = group.astype(str)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for level in _levels(group_str):
        columns.append((group_str == level).to_numpy(dtype=float))
        names.append(f"group:{level}")
    for block_name, block in blocks.items():
        block_str = block.astype(str)
        for level in _levels(block_str)[1:]:
            columns.append((block_str == level).to_numpy(dtype=float))
            names.append(f"{block_name}:{level}")
    matrix = np.column_stack(columns) if columns else np.zeros((len(group), 0))
    return matrix, names


def _matrix_rank(matrix: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(matrix)) if matrix.size else 0


def _determines(fine: pd.Series, coarse: pd.Series) -> bool:
    """True when every level of ``fine`` lies in one level of ``coarse``."""
    table = pd.crosstab(fine.astype(str), coarse.astype(str)) > 0
    return bool((table.sum(axis=1) == 1).all())


def _classify_block(
    group: pd.Series, block: pd.Series
) -> tuple[DropReason | None, dict[str, str] | None]:
    """Classify a block against the group from their crosstab."""
    table = pd.crosstab(group.astype(str), block.astype(str)) > 0
    blocks_per_group = table.sum(axis=1)
    groups_per_block = table.sum(axis=0)
    group_in_one_block = bool((blocks_per_group == 1).all())
    block_in_one_group = bool((groups_per_block == 1).all())
    if group_in_one_block:
        mapping = {str(g): str(table.columns[row.to_numpy().argmax()]) for g, row in table.iterrows()}
        if block_in_one_group:
            return "aliased_with_group", mapping
        return "group_nested_in_block", mapping
    if block_in_one_group:
        return "block_nested_in_group", None
    return None, None


def _drop_reason_text(code: DropReason, column: str, detail: str = "") -> str:
    texts: dict[DropReason, str] = {
        "single_level": f"'{column}' has one level only, so it cannot vary in the design.",
        "aliased_with_group": (
            f"Each level of '{column}' matches exactly one group, so the block is "
            "aliased with the group and cannot be estimated."
        ),
        "group_nested_in_block": (
            f"Every group lies in one level of '{column}' (the groups are nested in "
            "the block), so the block is a sum of group coefficients and cannot be "
            "a covariate. Contrasts across its levels also compare the block."
        ),
        "block_nested_in_group": (
            f"Each level of '{column}' lies in one group only (the block is nested "
            "in the group), so it cannot be a fixed covariate next to ~0 + group. "
            "A random-effect method such as duplicateCorrelation would be needed."
        ),
        "implied_by_block": (
            f"'{column}' is determined by another kept block{detail} (each level of "
            "that block lies in one level of this one), so its effect is already "
            "in the design. Example: cell line inside a cell line x replicate pair."
        ),
        "confounded": (
            f"Adding '{column}' makes the design matrix rank-deficient{detail}; it is "
            "partly confounded with the group or with an earlier block."
        ),
    }
    return texts[code]


def validate_design(
    samples: pd.DataFrame,
    group_columns: Sequence[str],
    block_columns: Sequence[str] = (),
    *,
    group_variable: str = DEFAULT_GROUP_VARIABLE,
    min_replicates: int = MIN_REPLICATES_PER_GROUP,
) -> DesignValidationResult:
    """Validate a ``~0 + group + blocks`` design for a sample table.

    Args:
        samples: One row per sample.
        group_columns: Columns joined (with ``"_"``) into the group factor.
        block_columns: Candidate blocking factors, tried in the given order.
        group_variable: Name of the group factor in the formula.
        min_replicates: Groups below this size are listed in ``small_groups``.

    Returns:
        The validated design, with kept and dropped terms and their reasons.

    Raises:
        DesignValidationError: if a column is missing or has missing values,
            if fewer than two groups remain, if joining the group columns or
            R's ``make.names`` would merge two different groups, or if no
            residual degrees of freedom remain.
    """
    requested_groups = list(group_columns)
    requested_blocks = list(block_columns)
    if samples.empty:
        raise DesignValidationError("The sample table is empty.")
    if not requested_groups:
        raise DesignValidationError("At least one group column is required.")
    overlap = sorted(set(requested_groups) & set(requested_blocks))
    if overlap:
        raise DesignValidationError(
            f"Columns cannot be both a group and a block column: {overlap}"
        )
    missing = [c for c in requested_groups + requested_blocks if c not in samples.columns]
    if missing:
        raise DesignValidationError(
            f"Design columns not found in the sample table: {missing}. "
            f"Available: {list(samples.columns)}"
        )
    for col in requested_groups + requested_blocks:
        string_column(samples, col)  # raises on a duplicated column name
    with_na = [c for c in requested_groups + requested_blocks if bool(samples[c].isna().any())]
    if with_na:
        raise DesignValidationError(
            f"Design columns have missing values: {with_na}. Fill or remove "
            "those samples before building the design."
        )

    n_samples = len(samples)
    dropped: list[DroppedTerm] = []

    kept_group_cols: list[str] = []
    for col in requested_groups:
        n_levels = int(string_column(samples, col).nunique())
        if n_levels < 2:
            reason = _drop_reason_text("single_level", col)
            logger.info("Design: dropping group column %r: %s", col, reason)
            dropped.append(DroppedTerm(col, "group", "single_level", reason, n_levels))
        else:
            kept_group_cols.append(col)
    if not kept_group_cols:
        raise DesignValidationError(
            f"No group column varies across samples (requested: {requested_groups}); "
            "differential expression needs at least two groups."
        )

    group_parts = [string_column(samples, c) for c in kept_group_cols]
    group = group_parts[0]
    for part in group_parts[1:]:
        group = group + GROUP_LEVEL_SEPARATOR + part
    if len(kept_group_cols) > 1:
        combos = set(zip(*(string_column(samples, c).tolist() for c in kept_group_cols)))
        by_joined: dict[str, list[tuple[str, ...]]] = {}
        for combo in sorted(combos):
            by_joined.setdefault(GROUP_LEVEL_SEPARATOR.join(combo), []).append(combo)
        clashes = {j: c for j, c in by_joined.items() if len(c) > 1}
        if clashes:
            raise DesignValidationError(
                f"Joining {kept_group_cols} with {GROUP_LEVEL_SEPARATOR!r} merges "
                f"different groups: {clashes}."
            )
    group_levels = _levels(group)
    by_r_name: dict[str, list[str]] = {}
    for lvl in group_levels:
        by_r_name.setdefault(r_make_names(lvl), []).append(lvl)
    r_clashes = {r: lvls for r, lvls in by_r_name.items() if len(lvls) > 1}
    if r_clashes:
        raise DesignValidationError(
            "Different groups get the same R name (make.names), so RNAseq.R would "
            f"merge them silently: {r_clashes}."
        )

    kept_blocks: dict[str, pd.Series] = {}
    candidates: dict[str, pd.Series] = {}
    for col in requested_blocks:
        block = string_column(samples, col)
        n_levels = int(block.nunique())
        if n_levels < 2:
            reason = _drop_reason_text("single_level", col)
            logger.info("Design: dropping block %r: %s", col, reason)
            dropped.append(DroppedTerm(col, "block", "single_level", reason, n_levels))
            continue
        code, mapping = _classify_block(group, block)
        if code is not None:
            reason = _drop_reason_text(code, col)
            logger.info("Design: dropping block %r: %s", col, reason)
            dropped.append(DroppedTerm(col, "block", code, reason, n_levels, mapping))
            continue
        candidates[col] = block

    # A block that another candidate block determines (cell line, when a
    # cell line x replicate pair block is also requested) adds nothing; drop
    # it so the finer block is kept. For identical partitions keep the first.
    implied: dict[str, str] = {}
    names = list(candidates)
    for col in names:
        for other in names:
            if other == col or other in implied:
                continue
            if _determines(candidates[other], candidates[col]) and (
                not _determines(candidates[col], candidates[other])
                or names.index(other) < names.index(col)
            ):
                implied[col] = other
                break
    for col, block in candidates.items():
        if col in implied:
            reason = _drop_reason_text("implied_by_block", col, f" ('{implied[col]}')")
            logger.info("Design: dropping block %r: %s", col, reason)
            dropped.append(
                DroppedTerm(col, "block", "implied_by_block", reason, int(block.nunique()))
            )
            continue
        n_levels = int(block.nunique())
        trial = {**kept_blocks, col: block}
        matrix, _ = build_design_matrix(group, trial)
        if _matrix_rank(matrix) < matrix.shape[1]:
            detail = f" (rank {_matrix_rank(matrix)} < {matrix.shape[1]} columns)"
            reason = _drop_reason_text("confounded", col, detail)
            logger.info("Design: dropping block %r: %s", col, reason)
            dropped.append(DroppedTerm(col, "block", "confounded", reason, n_levels))
            continue
        kept_blocks[col] = block

    matrix, design_names = build_design_matrix(group, kept_blocks)
    rank = _matrix_rank(matrix)
    if rank < matrix.shape[1]:
        # Every block was checked above; reaching here is a bug, not a user error.
        raise DesignValidationError(
            f"Design matrix is rank-deficient (rank {rank} < {matrix.shape[1]})."
        )
    residual_df = n_samples - rank
    if residual_df <= 0:
        raise DesignValidationError(
            f"No residual degrees of freedom: {n_samples} samples and {rank} "
            "coefficients. Replicates are required to estimate variance."
        )
    group_sizes = {lvl: int((group == lvl).sum()) for lvl in group_levels}
    small = {lvl: n for lvl, n in group_sizes.items() if n < min_replicates}
    if small:
        logger.warning(
            "Design: groups with fewer than %d samples (no within-group variance): %s",
            min_replicates, small,
        )

    kept_terms = [
        KeptTerm(
            column=GROUP_LEVEL_SEPARATOR.join(kept_group_cols),
            r_name=group_variable,
            role="group",
            n_levels=len(group_levels),
            levels=group_levels,
        )
    ] + [
        KeptTerm(
            column=col,
            r_name=r_make_names(col),
            role="block",
            n_levels=int(block.nunique()),
            levels=_levels(block),
        )
        for col, block in kept_blocks.items()
    ]
    formula = "~0 + " + " + ".join(t.r_name for t in kept_terms)
    logger.info(
        "Design: %s, %d samples, rank %d, residual df %d",
        formula, n_samples, rank, residual_df,
    )
    return DesignValidationResult(
        formula=formula,
        group_variable=group_variable,
        group_columns=kept_group_cols,
        block_columns=list(kept_blocks),
        group_levels=group_levels,
        group_sizes=group_sizes,
        kept_terms=kept_terms,
        dropped_terms=dropped,
        n_samples=n_samples,
        n_coefficients=int(matrix.shape[1]),
        rank=rank,
        residual_df=residual_df,
        requested_group_columns=requested_groups,
        requested_block_columns=requested_blocks,
        design_columns=design_names,
        small_groups=small,
        sample_groups=group.tolist(),
    )
