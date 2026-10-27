"""Tests for uorca.core.design_validation (no LLM, no R)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from uorca.core.design_validation import (
    DesignValidationError,
    r_make_names,
    validate_design,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "contrast_design"

CELL_LINE = "characteristics_ch1.0.cell line"
GSE298581_TREATMENT = "characteristics_ch1.3.treatment"
GSE298581_BATCH = "characteristics_ch1.6.batch"
GSE295568_GROUP_COLS = [
    "characteristics_ch1.0.tissue",
    "characteristics_ch1.1.cell line",
    "characteristics_ch1.2.cell type",
    "characteristics_ch1.3.treatment",
]
GSE295568_BATCH = "characteristics_ch1.4.batch"


def _samples(accession: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / f"{accession}_samples.csv")


# --- r_make_names -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("DMSO", "DMSO"),
        ("0.5 uM T0070907", "X0.5.uM.T0070907"),
        ("Pim1-/-", "Pim1..."),
        ("B4galt1-HKO", "B4galt1.HKO"),
        ("_a", "X_a"),
        (".5x", "X.5x"),
        (".a", ".a"),
        ("", "X"),
        ("TRUE", "TRUE."),
        ("characteristics_ch1.0.cell line", "characteristics_ch1.0.cell.line"),
        ("3–20 h", "X3.20.h"),
        ("1 μM", "X1.μM"),
    ],
)
def test_r_make_names_matches_r(raw, expected):
    assert r_make_names(raw) == expected


# --- real fixtures ----------------------------------------------------------


def test_gse298581_keeps_cell_line_block_with_42_residual_df():
    result = validate_design(
        _samples("GSE298581"),
        group_columns=[GSE298581_TREATMENT],
        block_columns=[CELL_LINE, GSE298581_BATCH],
    )
    assert result.block_columns == [CELL_LINE]
    assert result.formula == "~0 + group + characteristics_ch1.0.cell.line"
    assert result.n_samples == 48
    assert result.rank == 6
    assert result.residual_df == 42
    assert set(result.group_sizes.values()) == {12}
    dropped = {d.column: d.reason_code for d in result.dropped_terms}
    assert dropped == {GSE298581_BATCH: "single_level"}


def test_gse298581_treatment_only_has_44_residual_df():
    """The current pipeline's design, for comparison."""
    result = validate_design(_samples("GSE298581"), [GSE298581_TREATMENT])
    assert result.formula == "~0 + group"
    assert result.residual_df == 44


def test_gse295568_drops_batch_because_groups_are_nested_in_it():
    result = validate_design(
        _samples("GSE295568"),
        group_columns=GSE295568_GROUP_COLS,
        block_columns=[GSE295568_BATCH],
    )
    assert result.formula == "~0 + group"
    assert result.block_columns == []
    assert result.group_columns == ["characteristics_ch1.3.treatment"]
    assert len(result.group_levels) == 22
    assert result.residual_df == 22
    codes = {d.column: d.reason_code for d in result.dropped_terms}
    assert codes[GSE295568_BATCH] == "group_nested_in_block"
    for col in GSE295568_GROUP_COLS[:3]:
        assert codes[col] == "single_level"
    batch_map = result.block_of_group()[GSE295568_BATCH]
    assert len(batch_map) == 22
    assert sorted(set(batch_map.values())) == [f"Batch{i}" for i in range(1, 6)]


@pytest.mark.parametrize(
    "accession,group_col",
    [
        ("GSE166308", "characteristics_ch1.1.genotype"),
        ("GSE324244", "characteristics_ch1.1.genotype"),
        ("GSE132902", "characteristics_ch1.2.treatment"),
    ],
)
def test_simple_two_group_datasets(accession, group_col):
    df = _samples(accession)
    result = validate_design(df, [group_col])
    assert result.formula == "~0 + group"
    assert len(result.group_levels) == 2
    assert result.residual_df == len(df) - 2


# --- synthetic edge cases ---------------------------------------------------


def _table(groups, blocks=None):
    data = {"g": groups}
    if blocks is not None:
        data["b"] = blocks
    return pd.DataFrame(data)


def test_aliased_block_is_dropped():
    df = _table(["A", "A", "B", "B"], ["x", "x", "y", "y"])
    result = validate_design(df, ["g"], ["b"])
    assert result.dropped_terms[0].reason_code == "aliased_with_group"
    assert result.block_of_group() == {"b": {"A": "x", "B": "y"}}


def test_block_nested_in_group_is_dropped():
    df = _table(["A"] * 4 + ["B"] * 4, ["d1", "d1", "d2", "d2", "d3", "d3", "d4", "d4"])
    result = validate_design(df, ["g"], ["b"])
    dropped = result.dropped_terms[0]
    assert dropped.reason_code == "block_nested_in_group"
    assert "duplicateCorrelation" in dropped.reason
    assert result.block_of_group() == {}


def test_partially_confounded_block_is_dropped_by_rank_check():
    # Both blocks are crossed with the group, but b2 has the same partition
    # as b, so adding it after b makes the matrix rank-deficient.
    df = pd.DataFrame(
        {
            "g": ["A", "A", "A", "B", "B", "B"],
            "b": ["x", "y", "y", "x", "y", "y"],
            "b2": ["p", "q", "q", "p", "q", "q"],
        }
    )
    result = validate_design(df, ["g"], ["b", "b2"])
    assert result.block_columns == ["b"]
    assert {d.column: d.reason_code for d in result.dropped_terms} == {"b2": "confounded"}


def test_multiple_group_columns_are_joined_like_the_pipeline():
    df = pd.DataFrame({"t": ["a", "a", "b", "b"], "s": ["m", "m", "m", "m"], "u": ["1", "1", "2", "2"]})
    result = validate_design(df, ["t", "s", "u"])
    assert result.group_columns == ["t", "u"]
    assert result.group_levels == ["a_1", "b_2"]


def test_singleton_group_raises():
    with pytest.raises(DesignValidationError, match="fewer than 2 replicates"):
        validate_design(_table(["A", "A", "B"]), ["g"])


def test_no_residual_df_raises():
    with pytest.raises(DesignValidationError, match="residual degrees of freedom"):
        validate_design(_table(["A", "B"]), ["g"], min_replicates=1)


def test_constant_group_raises():
    with pytest.raises(DesignValidationError, match="No group column varies"):
        validate_design(_table(["A", "A", "A"]), ["g"])


def test_missing_column_raises():
    with pytest.raises(DesignValidationError, match="not found"):
        validate_design(_table(["A", "B"]), ["nope"])


def test_missing_values_raise():
    with pytest.raises(DesignValidationError, match="missing values"):
        validate_design(_table(["A", "A", None, "B"]), ["g"])


def test_column_in_both_roles_raises():
    with pytest.raises(DesignValidationError, match="both a group and a block"):
        validate_design(_table(["A", "A", "B", "B"]), ["g"], ["g"])


def test_to_dict_is_json_serialisable_and_has_counts():
    result = validate_design(
        _samples("GSE298581"), [GSE298581_TREATMENT], [CELL_LINE, GSE298581_BATCH]
    )
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["formula"] == result.formula
    assert payload["counts"]["block_columns_before"] == 2
    assert payload["counts"]["block_columns_after"] == 1
    assert payload["counts"]["residual_df"] == 42
    assert payload["terms"]["dropped"][0]["reason_code"] == "single_level"
    assert [t["role"] for t in payload["terms"]["kept"]] == ["group", "block"]


def test_duplicated_column_name_raises():
    df = pd.concat(
        [pd.Series(["A", "A", "B", "B"], name="g"), pd.Series(["x", "y", "x", "y"], name="g")],
        axis=1,
    )
    with pytest.raises(DesignValidationError, match="more than once"):
        validate_design(df, ["g"])
