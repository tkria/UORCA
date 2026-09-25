"""Pure derivation functions for the dataset-identification scoring pipeline.

The two LLM stages return raw axis scores (BiologyScore, DesignScore); this
module converts those into the user-facing RelevanceScore and Justification
fields written to the results CSV. Single source of truth for the
derivation; importable both by the CLI scoring pipeline and any future GUI
display logic.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from pydantic import BaseModel


class BiologyAssessment(BaseModel):
    """Stage 1 LLM output schema — biology only."""

    ID: str
    BiologyScore: int
    BiologyJustification: str


class BiologyAssessments(BaseModel):
    """Stage 1 LLM batch output."""

    assessments: list[BiologyAssessment]


class RelevanceAssessment(BaseModel):
    """Stage 2 LLM output schema — biology + design."""

    ID: str
    BiologyScore: int
    DesignScore: int
    BiologyJustification: str
    DesignJustification: str


class RelevanceAssessments(BaseModel):
    """Stage 2 LLM batch output."""

    assessments: list[RelevanceAssessment]


def compute_relevance(
    *,
    biology: Optional[float],
    design: Optional[float],
    stage1_biology: int,
    biology_weight: float,
    design_min: int,
) -> float:
    """Derive the user-facing RelevanceScore.

    For Stage-2-scored datasets:
        relevance = biology_weight * biology + (1 - biology_weight) * design
        if design_min > 0 and design < design_min:
            relevance = min(relevance, design_min)

    For Stage-1-only datasets (biology or design is None):
        relevance = stage1_biology
    """
    if biology is None or design is None:
        return float(stage1_biology)

    design_weight = 1.0 - biology_weight
    relevance = biology_weight * biology + design_weight * design

    if design_min > 0 and design < design_min:
        relevance = min(relevance, float(design_min))

    return relevance


def compute_justification(
    *,
    biology_just: Optional[str],
    design_just: Optional[str],
    stage1_biology_just: str,
    design: Optional[float],
    design_min: int,
) -> str:
    """Derive the user-facing Justification field.

    Stage-1-only: returns Stage 1 justification.
    Stage 2 reached + design_min cap fired: returns DesignJustification (explains the cap).
    Stage 2 reached, no cap: returns BiologyJustification (the dominant axis on
    default weights — most informative for a researcher scanning results).
    """
    if biology_just is None:
        return stage1_biology_just

    if (
        design_min > 0
        and design is not None
        and design < design_min
        and design_just is not None
    ):
        return design_just

    return biology_just


# ---------------------------------------------------------------------------
# Row adapters
# ---------------------------------------------------------------------------
# Applied per-DataFrame-row at the end of an identification run. Every field may
# be missing: Stage-2 columns are NaN for rows that never graduated, and Stage-1
# columns are NaN when the LLM omitted a dataset from a batch. Note that NaN is
# truthy in Python, so `value or default` does NOT guard against it — these
# helpers must test for NaN explicitly.


def _as_number(value: Any) -> Optional[float]:
    """Return *value* as a float, or None when missing, NaN or non-numeric."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _as_text(value: Any) -> Optional[str]:
    """Return *value* as a non-empty string, or None when missing or NaN."""
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def row_relevance(
    row: Mapping[str, Any],
    *,
    biology_weight: float,
    design_min: int,
) -> float:
    """Derive RelevanceScore for one result row, tolerating missing scores.

    A dataset that Stage 1 never scored has no evidence of relevance, so it
    derives 0.0 rather than aborting the run.
    """
    stage1 = _as_number(row.get("Stage1BiologyScore"))
    return compute_relevance(
        biology=_as_number(row.get("BiologyScore")),
        design=_as_number(row.get("DesignScore")),
        stage1_biology=int(stage1) if stage1 is not None else 0,
        biology_weight=biology_weight,
        design_min=design_min,
    )


def row_justification(row: Mapping[str, Any], *, design_min: int) -> str:
    """Derive Justification for one result row, tolerating missing text."""
    return compute_justification(
        biology_just=_as_text(row.get("BiologyJustification")),
        design_just=_as_text(row.get("DesignJustification")),
        stage1_biology_just=_as_text(row.get("Stage1BiologyJustification")) or "",
        design=_as_number(row.get("DesignScore")),
        design_min=design_min,
    )
