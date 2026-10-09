"""Table-driven tests for the RelevanceScore and Justification derivation."""

import math

import pytest


# --- compute_relevance ---


@pytest.mark.parametrize(
    "biology, design, stage1, w_bio, design_min, expected",
    [
        # Default weights, no design cap
        (8.0, 5.0, 0, 0.8, 0, 0.8 * 8.0 + 0.2 * 5.0),
        # Biology-only weighting
        (7.0, 0.0, 0, 1.0, 0, 7.0),
        # Design-only weighting (niche but allowed)
        (0.0, 7.0, 0, 0.0, 0, 7.0),
        # Equal weighting
        (6.0, 4.0, 0, 0.5, 0, 5.0),
        # design_min cap fires (DesignScore < design_min)
        (10.0, 1.0, 0, 0.8, 5, 5.0),  # raw 8.2, capped at 5
        # design_min set but DesignScore meets it — no cap
        (6.0, 5.0, 0, 0.8, 5, 0.8 * 6.0 + 0.2 * 5.0),
        # design_min set but raw relevance is already below cap — passes through
        (1.0, 1.0, 0, 0.8, 5, 1.0),
    ],
)
def test_compute_relevance_stage2(biology, design, stage1, w_bio, design_min, expected):
    from uorca.identification.scoring import compute_relevance

    result = compute_relevance(
        biology=biology,
        design=design,
        stage1_biology=stage1,
        biology_weight=w_bio,
        design_min=design_min,
    )
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_compute_relevance_stage1_only_uses_stage1_score():
    """When biology and design are None (Stage-1-only dataset), use stage1_biology."""
    from uorca.identification.scoring import compute_relevance

    result = compute_relevance(
        biology=None,
        design=None,
        stage1_biology=4,
        biology_weight=0.8,
        design_min=0,
    )
    assert result == 4.0


def test_compute_relevance_partial_stage2_treated_as_stage1_only():
    """If only one of biology/design is None, fall back to Stage 1."""
    from uorca.identification.scoring import compute_relevance

    result = compute_relevance(
        biology=8.0,
        design=None,
        stage1_biology=3,
        biology_weight=0.8,
        design_min=0,
    )
    assert result == 3.0


# --- compute_justification ---


def test_justification_stage1_only_uses_stage1_just():
    from uorca.identification.scoring import compute_justification

    result = compute_justification(
        biology_just=None,
        design_just=None,
        stage1_biology_just="Biology unclear from title",
        design=None,
        design_min=0,
    )
    assert result == "Biology unclear from title"


def test_justification_default_uses_biology_just():
    """Default: biology is the dominant axis, so Justification = BiologyJustification."""
    from uorca.identification.scoring import compute_justification

    result = compute_justification(
        biology_just="Biology matches",
        design_just="Design has 3 reps",
        stage1_biology_just="(unused)",
        design=8.0,
        design_min=0,
    )
    assert result == "Biology matches"


def test_justification_design_cap_fired_uses_design_just():
    """When design_min cap fires, Justification = DesignJustification (explains why)."""
    from uorca.identification.scoring import compute_justification

    result = compute_justification(
        biology_just="Biology matches",
        design_just="Only n=1 per group",
        stage1_biology_just="(unused)",
        design=1.0,
        design_min=5,
    )
    assert result == "Only n=1 per group"


def test_justification_design_just_missing_falls_back_to_biology():
    """If DesignJustification is None and design_min cap fires, fall back to biology."""
    from uorca.identification.scoring import compute_justification

    result = compute_justification(
        biology_just="Biology matches",
        design_just=None,
        stage1_biology_just="(unused)",
        design=1.0,
        design_min=5,
    )
    assert result == "Biology matches"
