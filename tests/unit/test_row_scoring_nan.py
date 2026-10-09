"""Row-level score derivation must tolerate missing/NaN fields.

Stage 1 does not always return a score for every dataset — if the LLM omits one
entry from a batch, that row reaches the final derivation with NaN in its score
and justification columns. That must not abort the run: a whole identification
run's results are written only after this step, so one unscored dataset out of
thousands used to discard the entire run.
"""

import math

import pandas as pd
import pytest

from uorca.identification.scoring import row_justification, row_relevance


@pytest.mark.unit
def test_row_relevance_survives_unscored_stage1_row():
    """A row Stage 1 never scored must derive a score, not raise."""
    row = pd.Series(
        {
            "BiologyScore": float("nan"),
            "DesignScore": float("nan"),
            "Stage1BiologyScore": float("nan"),
            "Stage1BiologyJustification": float("nan"),
        }
    )

    score = row_relevance(row, biology_weight=0.8, design_min=0)

    assert isinstance(score, float)
    assert not math.isnan(score), "unscored row produced a NaN relevance score"
    assert score == 0.0, "an unscored dataset should rank lowest, not crash"


@pytest.mark.unit
def test_row_justification_survives_unscored_stage1_row():
    """Justification must be a string even when every source field is NaN."""
    row = pd.Series(
        {
            "BiologyJustification": float("nan"),
            "DesignJustification": float("nan"),
            "Stage1BiologyJustification": float("nan"),
            "DesignScore": float("nan"),
        }
    )

    justification = row_justification(row, design_min=0)

    assert isinstance(justification, str), (
        f"justification must be text, got {type(justification).__name__}"
    )


@pytest.mark.unit
def test_row_relevance_stage1_only_row_uses_stage1_score():
    """A row scored by Stage 1 but not Stage 2 falls back to the Stage 1 score."""
    row = pd.Series(
        {
            "BiologyScore": float("nan"),
            "DesignScore": float("nan"),
            "Stage1BiologyScore": 4.0,
            "Stage1BiologyJustification": "plausible on title alone",
        }
    )

    assert row_relevance(row, biology_weight=0.8, design_min=0) == 4.0


@pytest.mark.unit
def test_row_relevance_fully_scored_row_uses_both_axes():
    """A Stage 2 row combines both axes on the configured weighting."""
    row = pd.Series(
        {
            "BiologyScore": 8.0,
            "DesignScore": 5.0,
            "Stage1BiologyScore": 6.0,
            "Stage1BiologyJustification": "",
        }
    )

    expected = 0.8 * 8.0 + 0.2 * 5.0
    assert row_relevance(row, biology_weight=0.8, design_min=0) == pytest.approx(expected)


@pytest.mark.unit
def test_row_scoring_over_a_frame_with_one_unscored_row():
    """The real failure: apply() across a frame where one row escaped Stage 1."""
    df = pd.DataFrame(
        [
            {"BiologyScore": 8.0, "DesignScore": 5.0, "Stage1BiologyScore": 6.0,
             "BiologyJustification": "on target", "DesignJustification": "solid design",
             "Stage1BiologyJustification": "looks relevant"},
            {"BiologyScore": float("nan"), "DesignScore": float("nan"),
             "Stage1BiologyScore": float("nan"),
             "BiologyJustification": float("nan"), "DesignJustification": float("nan"),
             "Stage1BiologyJustification": float("nan")},
        ]
    )

    scores = df.apply(lambda r: row_relevance(r, biology_weight=0.8, design_min=0), axis=1)
    texts = df.apply(lambda r: row_justification(r, design_min=0), axis=1)

    assert not scores.isna().any(), "derivation produced NaN scores"
    assert all(isinstance(t, str) for t in texts)
