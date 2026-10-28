"""Deterministic "needs review" flags for a dataset's automatic design.

UORCA marks datasets whose automatic grouping or contrasts are likely to be
wrong, so a user can check those first (plan 2026-10-27-03, section 3.4).
Every signal is a fixed rule on the metadata, the design or the contrast
checks. No LLM is involved, and the same input always gives the same flags.

Signals computed here:

- ``unused_varying_column`` (high): a metadata column varies, is crossed with
  the groups (so it could be a block) and is not in the design. A column
  whose levels are fully determined by the group (for example a source
  column of a merged group) is not flagged.
- ``candidate_missing_values`` (medium): a candidate column has missing
  values, so it was not tested.
- ``group_name_collision`` (high): two group labels get the same R name and
  ``RNAseq.R`` would merge them.
- ``single_sample_group`` (high): a group has fewer than
  ``MIN_REPLICATES_PER_GROUP`` samples, so its variance cannot be estimated.
- ``many_groups``: more than ``MAX_GROUPS`` groups.
- ``few_replicates`` (medium, one per group): a group has exactly
  ``WEAK_REPLICATE_COUNT`` (2) replicates.
- ``unbalanced_groups``: the largest group is more than
  ``MAX_GROUP_SIZE_RATIO`` times the smallest.
- ``sentence_group_labels``: group labels are long free text rather than
  short condition names.
- ``block_removed``: design validation removed a block as aliased, nested or
  confounded (needs a :class:`DesignValidationResult`).
- ``contrast_rejected`` / ``contrast_warning``: a contrast check gave an error
  or a warning (needs :class:`ContrastCheck` results).

``needs_review`` is true when at least ``HIGH_SIGNALS_FOR_REVIEW`` high
signals or ``MEDIUM_SIGNALS_FOR_REVIEW`` medium signals fire. The result
serialises (``to_dict``) to the ``metadata/review_flags.json`` structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Sequence

import pandas as pd

from .contrast_checks import ContrastCheck
from .design_validation import (
    MIN_REPLICATES_PER_GROUP,
    DesignValidationResult,
    r_make_names,
    string_column,
)

MAX_GROUPS = 8
WEAK_REPLICATE_COUNT = 2
"""A group with exactly this many samples gets a medium flag (owner rule,
2026-10-28). Fewer than ``MIN_REPLICATES_PER_GROUP`` is high."""
MAX_GROUP_SIZE_RATIO = 3.0
SENTENCE_LABEL_MIN_WORDS = 8
SENTENCE_LABEL_MIN_CHARS = 60
HIGH_SIGNALS_FOR_REVIEW = 1
MEDIUM_SIGNALS_FOR_REVIEW = 2

IGNORED_COLUMN_PATTERN = re.compile(
    r"^(?:GSM|SRX|SRR|geo_accession|title|description|abundance_file|Unnamed: \d+"
    r"|status|submission_date|last_update_date|type|channel_count|taxid_ch1"
    r"|\w+_protocol_ch1|data_processing|platform_id|contact_.*|instrument_model"
    r"|library_selection|library_source|library_strategy|relation|supplementary_file_\d+"
    r"|series_id|data_row_count)$"
    r"|replicate|(?:^|[_.\s])rep(?:$|[_.\s])",
    re.IGNORECASE,
)
"""Columns that are identifiers or submission details, never design factors.

Replicate numbers are ignored too: in GEO they label samples within a group
and are rarely a real pairing factor.
"""

Severity = Literal["high", "medium", "low"]
DROP_REASONS_FLAGGED = ("aliased_with_group", "group_nested_in_block", "block_nested_in_group", "confounded")


@dataclass(frozen=True)
class ReviewSignal:
    signal: str
    severity: Severity
    reason: str


@dataclass
class ReviewFlags:
    needs_review: bool
    signals: list[ReviewSignal] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "needs_review": self.needs_review,
            "rule": (
                f">= {HIGH_SIGNALS_FOR_REVIEW} high or >= {MEDIUM_SIGNALS_FOR_REVIEW} "
                "medium signals"
            ),
            "signals": [
                {"signal": s.signal, "severity": s.severity, "reason": s.reason}
                for s in self.signals
            ],
            "reviewed_by_user": False,
        }


def _partition(series: pd.Series) -> frozenset[frozenset[int]]:
    codes = pd.factorize(series.astype(str))[0]
    groups: dict[int, set[int]] = {}
    for i, c in enumerate(codes):
        groups.setdefault(int(c), set()).add(i)
    return frozenset(frozenset(g) for g in groups.values())


def is_sentence_like(label: str) -> bool:
    """True when a group label reads like a sentence rather than a level."""
    return len(label.split()) >= SENTENCE_LABEL_MIN_WORDS or len(label) >= SENTENCE_LABEL_MIN_CHARS


def candidate_factor_columns(samples: pd.DataFrame) -> list[str]:
    """Columns that could be design factors: not ignored, 2..n-1 levels."""
    n = len(samples)
    out = []
    for col in samples.columns:
        if IGNORED_COLUMN_PATTERN.search(str(col)):
            continue
        n_levels = int(string_column(samples, str(col)).nunique())
        if 2 <= n_levels < n:
            out.append(str(col))
    return out


def metadata_signals(
    samples: pd.DataFrame,
    group_column: str,
    design_columns: Sequence[str],
    *,
    candidate_columns: Sequence[str] | None = None,
) -> list[ReviewSignal]:
    """Signals that need only the sample table and the chosen grouping.

    Args:
        samples: One row per sample.
        group_column: Column holding each sample's group label.
        design_columns: Source columns that were used or considered for the
            design (group and block columns). They never count as unused.
        candidate_columns: Columns to test for ``unused_varying_column``.
            Defaults to :func:`candidate_factor_columns`.

    Raises:
        KeyError: if ``group_column`` or a design column is missing.
    """
    missing = [c for c in [group_column, *design_columns] if c not in samples.columns]
    if missing:
        raise KeyError(f"Columns not found in the sample table: {missing}")
    if bool(samples[group_column].isna().any()):
        raise ValueError(f"Group column {group_column!r} has missing values.")

    signals: list[ReviewSignal] = []
    group = string_column(samples, group_column)
    sizes = {str(k): int(v) for k, v in group.value_counts().items()}

    used_partitions = {
        _partition(string_column(samples, c)) for c in [group_column, *design_columns]
    }
    candidates = (
        list(candidate_columns) if candidate_columns is not None else candidate_factor_columns(samples)
    )
    for col in candidates:
        if col == group_column or col in design_columns:
            continue
        n_missing = int(samples[col].isna().sum())
        if n_missing:
            signals.append(
                ReviewSignal(
                    "candidate_missing_values", "medium",
                    f"'{col}' has missing values in {n_missing} of {len(samples)} "
                    "samples, so it was not tested as a design factor.",
                )
            )
            continue
        values = string_column(samples, col)
        if _partition(values) in used_partitions:
            continue  # a copy of a used column, such as source_name = cell line
        crossed = bool((pd.crosstab(group, values) > 0).sum(axis=1).gt(1).any())
        if not crossed:
            continue  # determined by the group, e.g. a source column of a merged group
        signals.append(
            ReviewSignal(
                "unused_varying_column", "high",
                f"'{col}' varies ({int(values.nunique())} levels) and is crossed with "
                "the groups, but it is not in the design. It may be a blocking factor.",
            )
        )

    by_r_name: dict[str, list[str]] = {}
    for lvl in sizes:
        by_r_name.setdefault(r_make_names(lvl), []).append(lvl)
    for r_name, lvls in by_r_name.items():
        if len(lvls) > 1:
            signals.append(
                ReviewSignal(
                    "group_name_collision", "high",
                    f"Groups {lvls} all become {r_name!r} in R (make.names); "
                    "RNAseq.R would merge them into one group.",
                )
            )

    if len(sizes) > MAX_GROUPS:
        signals.append(
            ReviewSignal("many_groups", "medium", f"{len(sizes)} groups (more than {MAX_GROUPS}).")
        )
    for lvl, n in sizes.items():
        if n < MIN_REPLICATES_PER_GROUP:
            signals.append(
                ReviewSignal(
                    "single_sample_group", "high",
                    f"Group {lvl!r} has {n} sample; its variance cannot be estimated.",
                )
            )
        elif n == WEAK_REPLICATE_COUNT:
            signals.append(
                ReviewSignal(
                    "few_replicates", "medium",
                    f"Group {lvl!r} has only {n} replicates; variance estimates are weak.",
                )
            )
    largest, smallest = max(sizes.values()), min(sizes.values())
    if largest / smallest > MAX_GROUP_SIZE_RATIO:
        signals.append(
            ReviewSignal(
                "unbalanced_groups", "medium",
                f"Largest group has {largest} samples, smallest {smallest} "
                f"(more than {MAX_GROUP_SIZE_RATIO:g} times).",
            )
        )
    sentences = [lvl for lvl in sizes if is_sentence_like(lvl)]
    if sentences:
        signals.append(
            ReviewSignal(
                "sentence_group_labels", "medium",
                f"{len(sentences)} of {len(sizes)} group labels are long free text "
                "rather than short condition names; check that the grouping is "
                "right and that the labels are readable.",
            )
        )
    return signals


def design_signals(design: DesignValidationResult) -> list[ReviewSignal]:
    """Signals from :func:`uorca.core.design_validation.validate_design`."""
    return [
        ReviewSignal("block_removed", "medium", f"Block '{d.column}' removed ({d.reason_code}): {d.reason}")
        for d in design.dropped_terms
        if d.role == "block" and d.reason_code in DROP_REASONS_FLAGGED
    ]


def contrast_signals(checks: Sequence[ContrastCheck]) -> list[ReviewSignal]:
    """Signals from :mod:`uorca.core.contrast_checks` results."""
    signals: list[ReviewSignal] = []
    rejected = [c.name for c in checks if not c.is_valid]
    if rejected:
        signals.append(
            ReviewSignal("contrast_rejected", "high", f"Contrast checks rejected: {', '.join(rejected)}.")
        )
    warned = [c.name for c in checks if c.is_valid and c.warnings]
    if warned:
        signals.append(
            ReviewSignal("contrast_warning", "medium", f"Contrast checks warned about: {', '.join(warned)}.")
        )
    return signals


def decide(signals: Sequence[ReviewSignal]) -> ReviewFlags:
    """Apply the fixed rule that turns signals into ``needs_review``."""
    n_high = sum(s.severity == "high" for s in signals)
    n_medium = sum(s.severity == "medium" for s in signals)
    needs = n_high >= HIGH_SIGNALS_FOR_REVIEW or n_medium >= MEDIUM_SIGNALS_FOR_REVIEW
    return ReviewFlags(needs_review=needs, signals=list(signals))


def compute_review_flags(
    samples: pd.DataFrame,
    group_column: str,
    design_columns: Sequence[str],
    *,
    design: DesignValidationResult | None = None,
    contrast_checks: Sequence[ContrastCheck] | None = None,
    candidate_columns: Sequence[str] | None = None,
) -> ReviewFlags:
    """Compute all available signals and the ``needs_review`` decision."""
    signals = metadata_signals(
        samples, group_column, design_columns, candidate_columns=candidate_columns
    )
    if design is not None:
        signals += design_signals(design)
    if contrast_checks is not None:
        signals += contrast_signals(contrast_checks)
    return decide(signals)
