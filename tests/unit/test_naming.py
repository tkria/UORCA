"""Tests for uorca.core.naming: R names for group levels and contrast checks.

Real cases come from ``tests/fixtures/contrast_design`` (PPARg run,
2026-10-07): GSE161060, whose two cell types R's ``make.names`` merged, and
GSE210935, whose digit-leading contrast names and prefix-sharing levels made
R fail six times.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from uorca.core.contrast_checks import level_name_map
from uorca.core.design_validation import r_make_names
from uorca.core.naming import (
    R_GROUP_COLUMN,
    NamingError,
    assign_level_names,
    build_naming_record,
    check_joined_levels,
    contrast_design_issues,
    contrast_name_issues,
    contrast_weights,
    is_missing_value,
    legacy_r_naming_hazards,
    missing_group_signals,
    normalise_contrasts,
    normalise_expression,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "contrast_design"


def _samples(gse: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / f"{gse}_samples.csv", dtype=str)


def _contrasts(gse: str) -> list[dict[str, str]]:
    return pd.read_csv(FIXTURES / f"{gse}_contrasts.csv", dtype=str).to_dict(orient="records")


def _assert_valid_unique(mapping: dict[str, str]) -> None:
    names = list(mapping.values())
    assert len(set(names)) == len(names)
    for n in names:
        assert r_make_names(n) == n, n
        assert n.isascii(), n


# --- GSE161060: full collision (F1) ----------------------------------------


def test_gse161060_levels_get_distinct_valid_names():
    levels = _samples("GSE161060")["uorca_group"].unique().tolist()
    assert len(levels) == 2
    with pytest.raises(ValueError, match="merged"):
        level_name_map(levels)  # what RNAseq.R's make.names did

    mapping = assign_level_names(levels)

    _assert_valid_unique(mapping)
    assert mapping["CD45+F4/80+CD11b+EGFP-"] == "g01_CD45posF4_80posCD11bposEGFPneg"
    assert mapping["CD45+F4/80+CD11b+EGFP+"] == "g02_CD45posF4_80posCD11bposEGFPpos"
    hazards = legacy_r_naming_hazards(levels)
    assert any("merge" in h for h in hazards)


def test_gse161060_contrasts_on_assigned_names_are_valid():
    levels = _samples("GSE161060")["uorca_group"].unique().tolist()
    mapping = assign_level_names(levels)
    neg, pos = mapping["CD45+F4/80+CD11b+EGFP-"], mapping["CD45+F4/80+CD11b+EGFP+"]
    contrasts = [{"name": "EGFP_pos_vs_EGFP_neg", "expression": f"{pos} - {neg}"}]

    assert contrast_design_issues(contrasts, mapping) == []
    assert contrast_weights(contrasts, mapping) == {"EGFP_pos_vs_EGFP_neg": {pos: 1.0, neg: -1.0}}


def test_gse161060_original_raw_expressions_are_rejected_with_hint():
    levels = _samples("GSE161060")["uorca_group"].unique().tolist()
    mapping = assign_level_names(levels)
    issues = contrast_design_issues(_contrasts("GSE161060"), mapping)
    assert issues
    assert all("Write the R level names" in i for i in issues)


# --- GSE210935: digit-leading names (F4) and prefix corruption (F3) --------


def test_gse210935_levels_keep_readable_names():
    levels = _samples("GSE210935")["uorca_group"].unique().tolist()
    assert len(levels) == 9
    mapping = assign_level_names(levels)
    _assert_valid_unique(mapping)
    assert mapping["8h_GM"] == "X8h_GM"
    assert mapping["8h_GMTGW"] == "X8h_GMTGW"
    assert mapping["MoDC_GMT"] == "MoDC_GMT"
    assert any("occurs inside" in h for h in legacy_r_naming_hazards(levels))


def test_gse210935_original_design_is_rejected_before_kallisto():
    levels = _samples("GSE210935")["uorca_group"].unique().tolist()
    mapping = assign_level_names(levels)
    issues = contrast_design_issues(_contrasts("GSE210935"), mapping)
    bad_names = [i for i in issues if i.startswith("Contrast name")]
    # The five 8h_* names start with a digit and broke R's parse().
    assert len(bad_names) == 5
    assert all("'8h_" in i for i in bad_names)
    # Raw tokens such as 8h_GM are not design levels.
    assert any("Write the R level names" in i for i in issues)


def test_gse210935_corrected_design_has_correct_weights():
    levels = _samples("GSE210935")["uorca_group"].unique().tolist()
    mapping = assign_level_names(levels)
    original = _contrasts("GSE210935")
    corrected = []
    for row in original:
        target, reference = (t.strip() for t in row["expression"].split(" - "))
        corrected.append(
            {
                "name": "c_" + row["name"] if row["name"][0].isdigit() else row["name"],
                "expression": f"{mapping[target]} - {mapping[reference]}",
                "target": mapping[target],
                "reference": mapping[reference],
            }
        )

    assert contrast_design_issues(corrected, mapping) == []
    weights = contrast_weights(corrected, mapping)
    assert len(weights) == 10
    for row in corrected:
        assert weights[row["name"]] == {row["target"]: 1.0, row["reference"]: -1.0}


# --- constructed failure modes ---------------------------------------------


def test_partial_collision_keeps_three_groups():
    """F2: Tg+ and Tg- both become Tg. in R; WT is untouched."""
    mapping = assign_level_names(["Tg+", "Tg-", "WT"])
    _assert_valid_unique(mapping)
    assert mapping == {"Tg+": "g01_Tgpos", "Tg-": "g02_Tgneg", "WT": "WT"}


def test_gse224129_like_genotypes_stay_three():
    mapping = assign_level_names(["cMyBP-C+/+", "cMyBP-C+/-", "cMyBP-C-/-"])
    _assert_valid_unique(mapping)
    assert len(set(mapping.values())) == 3


def test_numeric_levels_and_divisor():
    """F5: a level named 2 must not capture the divisor in (2 + 4)/2."""
    mapping = assign_level_names(["0", "2", "4"])
    assert mapping == {"0": "X0", "2": "X2", "4": "X4"}
    good = [{"name": "dose_avg_vs_zero", "expression": "(X2 + X4)/2 - X0"}]
    assert contrast_weights(good, mapping) == {"dose_avg_vs_zero": {"X2": 0.5, "X4": 0.5, "X0": -1.0}}
    raw = [{"name": "dose_avg_vs_zero", "expression": "(2 + 4)/2 - 0"}]
    assert contrast_design_issues(raw, mapping)


def test_hyphenated_raw_level_is_not_read_as_minus():
    """F6: the LLM must use the R name, not '0.5 uM FX-909'."""
    mapping = assign_level_names(["DMSO", "0.5 uM FX-909"])
    fx = mapping["0.5 uM FX-909"]
    assert fx == "X0.5.uM.FX.909"
    assert contrast_design_issues([{"name": "FX_vs_DMSO", "expression": f"{fx} - DMSO"}], mapping) == []
    issues = contrast_design_issues([{"name": "FX_vs_DMSO", "expression": "FX-909 - DMSO"}], mapping)
    assert issues


def test_non_ascii_levels_get_ascii_ids():
    """F18: make.names keeps 'μ' only in a UTF-8 locale."""
    mapping = assign_level_names(["1 μM T0070907", "α-SMA", "DMSO", "5 \u00b5M X"])
    _assert_valid_unique(mapping)
    assert mapping["1 μM T0070907"] == "g01_1_uM_T0070907"
    assert mapping["α-SMA"] == "g02_a_SMA"
    assert mapping["DMSO"] == "DMSO"
    assert mapping["5 \u00b5M X"] == "g04_5_uM_X"  # micro, not milli


def test_reserved_and_empty_levels():
    mapping = assign_level_names(["TRUE", "NA", "", "nan", "if"])
    _assert_valid_unique(mapping)
    assert mapping["TRUE"] == "TRUE."
    assert mapping["NA"] == "NA."
    assert mapping[""] == "X"


def test_long_colliding_levels_get_bounded_slugs():
    base = "very long cell population name " * 7
    mapping = assign_level_names([base + "+", base + "-"])
    _assert_valid_unique(mapping)
    assert all(len(n) <= 3 + 1 + 32 for n in mapping.values())


def test_assigned_name_never_clashes_with_a_kept_name():
    mapping = assign_level_names(["A+", "A-", "g01_Apos"])
    _assert_valid_unique(mapping)


def test_duplicate_input_levels_raise():
    with pytest.raises(NamingError, match="more than once"):
        assign_level_names(["A", "B", "A"])


def test_joined_column_collision_raises():
    """F17: a_b + c and a + b_c both join to a_b_c."""
    with pytest.raises(NamingError, match="merges different groups"):
        check_joined_levels([("a_b", "c"), ("a", "b_c")])
    check_joined_levels([("a", "c"), ("b", "c")])


# --- contrast names (F4, F7, F9, F10, F11) ---------------------------------


@pytest.mark.parametrize(
    "names, fragment",
    [
        (["8h_GMIL4_vs_8h_GM"], "not allowed"),
        (["high-dose_vs_saline"], "not allowed"),
        (["KO.vs.WT", "KO_vs_WT"], "not allowed"),
        (["KO_vs_WT", "ko_vs_wt"], "same ignoring case"),
        (["KO_vs_WT", "KO_vs_WT"], "same ignoring case"),
        (["TRUE"], "reserved"),
        (["levels"], "reserved"),
        (['x = "A - B", y'], "not allowed"),
        (["a`b"], "not allowed"),
    ],
)
def test_bad_contrast_names(names, fragment):
    issues = contrast_name_issues(names)
    assert issues and any(fragment in i for i in issues)


def test_good_contrast_names():
    assert contrast_name_issues(["KO_vs_WT", "Treated_vs_Control_24h"]) == []


@pytest.mark.parametrize(
    "expression",
    [
        'A - B + 0*nchar(Sys.setenv(X="1"))',
        'A - "B"',
        "A - B\\",
        "`A` - system('ls')",
        "A == B",
    ],
)
def test_hostile_expressions_are_rejected(expression):
    mapping = assign_level_names(["A", "B"])
    assert contrast_design_issues([{"name": "A_vs_B", "expression": expression}], mapping)


def test_unscaled_and_non_comparison_weights_are_rejected():
    mapping = assign_level_names(["A", "B", "C"])
    assert contrast_design_issues([{"name": "x", "expression": "A + B - C"}], mapping)
    assert contrast_design_issues([{"name": "x", "expression": "2*A - 2*B"}], mapping)
    assert contrast_design_issues([{"name": "x", "expression": "A - B - C"}], mapping)


# --- naming.json ----------------------------------------------------------


def test_naming_record_round_trips_through_json():
    mapping = assign_level_names(["Tg+", "Tg-", "WT"])
    contrasts = [
        {"name": "Tgpos_vs_WT", "expression": "g01_Tgpos - WT", "description": "d"},
        {"name": "Tgneg_vs_WT", "expression": "g02_Tgneg - WT", "description": "d"},
    ]
    record = build_naming_record(
        group_column="genotype",
        source_columns=["genotype:ch1"],
        level_names=mapping,
        level_counts={"Tg+": 3, "Tg-": 3, "WT": 3},
        contrasts=contrasts,
    )
    again = json.loads(json.dumps(record))
    assert again == record
    assert again["schema_version"] == 1
    assert again["group_column"]["r_name"] == R_GROUP_COLUMN
    assert [c["folder"] for c in again["contrasts"]] == ["Tgpos_vs_WT", "Tgneg_vs_WT"]
    # The two Tg contrasts no longer share a weight column (F2).
    assert again["contrasts"][0]["weights"] != again["contrasts"][1]["weights"]


def test_naming_record_refuses_invalid_design():
    mapping = assign_level_names(["A", "B"])
    with pytest.raises(NamingError, match="Invalid contrast design"):
        build_naming_record(
            group_column="g",
            source_columns=["g"],
            level_names=mapping,
            level_counts={},
            contrasts=[{"name": "1bad", "expression": "A - B"}],
        )


@pytest.mark.parametrize(
    "gse", sorted(p.name.split("_")[0] for p in FIXTURES.glob("GSE*_samples.csv"))
)
def test_every_fixture_dataset_gets_valid_unique_level_names(gse):
    """Replaces the data tests of the old test_contrast_sanitization.py, which
    read ../../UORCA_results and skipped silently when it was absent."""
    levels = _samples(gse)["uorca_group"].unique().tolist()
    mapping = assign_level_names(levels)
    _assert_valid_unique(mapping)
    assert len(mapping) == len(levels)


@pytest.mark.parametrize(
    "expression",
    ["KO -\nWT", "KO\t-\tWT", "KO\r\n- WT", "KO\u00a0-\u00a0WT", "  KO   -  WT \n"],
)
def test_whitespace_is_normalised_to_single_ascii_spaces(expression):
    """RNAseq.R accepts only ASCII spaces; Python must check and write that form."""
    assert normalise_expression(expression) == "KO - WT"
    mapping = assign_level_names(["KO", "WT"])
    rows = [{"name": "KO_vs_WT", "expression": expression, "description": "d"}]
    assert contrast_design_issues(rows, mapping) == []
    assert normalise_contrasts(rows) == [{"name": "KO_vs_WT", "expression": "KO - WT", "description": "d"}]
    record = build_naming_record(
        group_column="g", source_columns=["g"], level_names=mapping, level_counts={}, contrasts=rows
    )
    assert record["contrasts"][0]["expression"] == "KO - WT"


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA, "", "   ", "\t"])
def test_missing_values(value):
    assert is_missing_value(value)


@pytest.mark.parametrize("value", ["WT", "nan", "NA", 0, 0.0])
def test_present_values(value):
    assert not is_missing_value(value)


def test_missing_group_values_flag_or_stop():
    rows = [("WT", "d1"), ("", "d2"), ("KO", float("nan")), ("KO", "d3")]
    signals = missing_group_signals(rows, ["genotype", "donor"], policy="flag")
    assert [(s.signal, s.severity) for s in signals] == [("missing_group_values", "high")]
    assert "2 of 4 samples" in signals[0].reason
    with pytest.raises(NamingError, match="2 of 4 samples"):
        missing_group_signals(rows, ["genotype", "donor"], policy="stop")
    assert missing_group_signals([("WT",), ("KO",)], ["genotype"]) == []
