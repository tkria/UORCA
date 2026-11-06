"""Tests for uorca.core.review_flags (no LLM, no R)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from uorca.core.contrast_checks import check_contrasts
from uorca.core.design_validation import validate_design
from uorca.core.review_flags import (
    MAX_GROUPS,
    ReviewSignal,
    candidate_factor_columns,
    compute_review_flags,
    decide,
    is_sentence_like,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "contrast_design"

# Source columns the 2026-10-07 run joined into its group column.
PIPELINE_DESIGN_COLUMNS = {
    "GSE298581": ["characteristics_ch1.3.treatment"],
    "GSE295568": [
        "characteristics_ch1.0.tissue",
        "characteristics_ch1.1.cell line",
        "characteristics_ch1.2.cell type",
        "characteristics_ch1.3.treatment",
    ],
    "GSE166308": ["characteristics_ch1.1.genotype"],
    "GSE324244": ["characteristics_ch1.1.genotype"],
    "GSE132902": ["characteristics_ch1.2.treatment"],
}


def _samples(accession: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / f"{accession}_samples.csv")


def _pipeline_flags(accession: str):
    return compute_review_flags(
        _samples(accession), "uorca_group", PIPELINE_DESIGN_COLUMNS[accession]
    )


def _codes(flags):
    return {s.signal for s in flags.signals}


def test_gse295568_is_flagged():
    flags = _pipeline_flags("GSE295568")
    assert flags.needs_review
    assert _codes(flags) == {"many_groups", "few_replicates", "sentence_group_labels"}
    sentence = next(s for s in flags.signals if s.signal == "sentence_group_labels")
    assert "joined" not in sentence.reason


def test_columns_determined_by_the_group_are_not_flagged():
    # Without knowing the source columns, the merged group's sources (tissue
    # ... treatment) and batch are all determined by the group.
    flags = compute_review_flags(_samples("GSE295568"), "uorca_group", [])
    assert "unused_varying_column" not in _codes(flags)


def test_gse161060_group_name_collision_is_flagged():
    flags = compute_review_flags(_samples("GSE161060"), "uorca_group", [])
    assert flags.needs_review
    collision = [s for s in flags.signals if s.signal == "group_name_collision"]
    assert len(collision) == 1 and collision[0].severity == "high"
    assert "CD45.F4.80.CD11b.EGFP." in collision[0].reason


def test_gse288554_single_sample_groups_are_high():
    flags = compute_review_flags(_samples("GSE288554"), "uorca_group", [])
    single = [s for s in flags.signals if s.signal == "single_sample_group"]
    assert len(single) == 6 and {s.severity for s in single} == {"high"}
    assert "variance cannot be estimated" in single[0].reason
    assert flags.needs_review


def test_missing_values_in_candidate_column_are_reported():
    samples = pd.DataFrame({"g": ["A"] * 3 + ["B"] * 3, "c": ["x", None, "y", "x", "y", "x"]})
    flags = compute_review_flags(samples, "g", ["g"])
    assert [s.signal for s in flags.signals] == ["candidate_missing_values"]
    assert "'c'" in flags.signals[0].reason


def test_gse298581_is_flagged_for_unused_cell_line():
    flags = _pipeline_flags("GSE298581")
    assert flags.needs_review
    unused = [s for s in flags.signals if s.signal == "unused_varying_column"]
    assert [s.severity for s in unused] == ["high"]
    assert "cell line" in unused[0].reason


@pytest.mark.parametrize("accession", ["GSE166308", "GSE324244", "GSE132902"])
def test_simple_two_group_datasets_are_not_flagged(accession):
    flags = _pipeline_flags(accession)
    assert not flags.needs_review
    assert flags.signals == []


def test_gse298581_with_cell_line_block_is_not_flagged():
    samples = _samples("GSE298581")
    design = validate_design(
        samples, ["characteristics_ch1.3.treatment"],
        ["characteristics_ch1.0.cell line", "characteristics_ch1.6.batch"],
    )
    flags = compute_review_flags(
        samples.assign(design_group=design.sample_groups),
        "design_group",
        ["characteristics_ch1.3.treatment", "characteristics_ch1.0.cell line",
         "characteristics_ch1.6.batch"],
        design=design,
    )
    assert flags.signals == []
    assert not flags.needs_review


def test_gse295568_design_signal_for_removed_batch():
    samples = _samples("GSE295568")
    cols = PIPELINE_DESIGN_COLUMNS["GSE295568"]
    design = validate_design(samples, cols, ["characteristics_ch1.4.batch"])
    flags = compute_review_flags(
        samples.assign(design_group=design.sample_groups),
        "design_group",
        cols + ["characteristics_ch1.4.batch"],
        design=design,
    )
    assert "block_removed" in _codes(flags)
    assert "unused_varying_column" not in _codes(flags)
    removed = [s for s in flags.signals if s.signal == "block_removed"]
    assert "group_nested_in_block" in removed[0].reason
    # Single-level columns are dropped too, but they are not review signals.
    assert len(removed) == 1


def test_contrast_signals_from_checks():
    samples = pd.DataFrame({"g": ["A"] * 3 + ["B"] * 3 + ["C"] * 3})
    checks = check_contrasts(
        [{"name": "ok", "expression": "A - B"}, {"name": "bad", "expression": "A + B - C"}],
        ["A", "B", "C"],
    ).checks
    flags = compute_review_flags(samples, "g", ["g"], contrast_checks=checks)
    assert flags.needs_review
    assert [s.signal for s in flags.signals] == ["contrast_rejected"]
    assert "bad" in flags.signals[0].reason


def test_unbalanced_and_small_groups():
    samples = pd.DataFrame({"g": ["A"] * 10 + ["B"] * 2})
    flags = compute_review_flags(samples, "g", ["g"])
    assert {"unbalanced_groups", "few_replicates"} == _codes(flags)
    samples = pd.DataFrame({"g": ["A"] * 4 + ["B"]})
    assert _codes(compute_review_flags(samples, "g", ["g"])) == {"unbalanced_groups", "single_sample_group"}
    assert flags.needs_review  # two medium signals


def test_many_groups_threshold():
    n = MAX_GROUPS + 1
    samples = pd.DataFrame({"g": [f"G{i}" for i in range(n) for _ in range(3)]})
    assert _codes(compute_review_flags(samples, "g", ["g"])) == {"many_groups"}
    samples = pd.DataFrame({"g": [f"G{i}" for i in range(MAX_GROUPS) for _ in range(3)]})
    assert _codes(compute_review_flags(samples, "g", ["g"])) == set()


def test_decision_rule():
    med = ReviewSignal("x", "medium", "r")
    high = ReviewSignal("y", "high", "r")
    low = ReviewSignal("z", "low", "r")
    assert not decide([med]).needs_review
    assert decide([med, med]).needs_review
    assert decide([high]).needs_review
    assert not decide([low, low, low]).needs_review


def test_sentence_like_labels():
    assert not is_sentence_like("0.5 uM T0070907")
    assert not is_sentence_like("Pim1-/-")
    assert is_sentence_like(
        "TSCs were infected with PPARG KD1 shRNA lentivirus and maintained under self-renewal conditions."
    )


def test_candidate_columns_skip_ids_replicates_and_constants():
    cols = candidate_factor_columns(_samples("GSE298581"))
    assert "characteristics_ch1.0.cell line" in cols
    assert "characteristics_ch1.5.replicate" not in cols
    assert "characteristics_ch1.6.batch" not in cols  # one level
    assert "title" not in cols and "GSM" not in cols


def test_missing_column_raises():
    with pytest.raises(KeyError):
        compute_review_flags(pd.DataFrame({"g": ["A", "B"]}), "g", ["nope"])


def test_to_dict_is_json_serialisable():
    payload = json.loads(json.dumps(_pipeline_flags("GSE295568").to_dict()))
    assert payload["needs_review"] is True
    assert payload["reviewed_by_user"] is False
    assert all({"signal", "severity", "reason"} == set(s) for s in payload["signals"])


def test_two_replicate_group_is_medium_with_owner_wording():
    samples = pd.DataFrame({"g": ["A"] * 3 + ["B"] * 2})
    flags = compute_review_flags(samples, "g", ["g"])
    assert [(s.signal, s.severity) for s in flags.signals] == [("few_replicates", "medium")]
    assert flags.signals[0].reason == "Group 'B' has only 2 replicates; variance estimates are weak."
    assert not flags.needs_review  # one medium signal
    samples = pd.DataFrame({"g": ["A"] * 2 + ["B"] * 2})
    assert compute_review_flags(samples, "g", ["g"]).needs_review  # two medium signals
    samples = pd.DataFrame({"g": ["A"] * 3 + ["B"] * 3})
    assert compute_review_flags(samples, "g", ["g"]).signals == []
