"""Tests for uorca.core.contrast_checks (no LLM, no R)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from uorca.core.contrast_checks import (
    check_contrast,
    check_contrasts,
    level_name_map,
    to_r_expression,
)
from uorca.core.design_validation import validate_design

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "contrast_design"
GSE295568_BATCH = "characteristics_ch1.4.batch"
GSE295568_PREFIX = "Placenta_CT27_Human trophoblast stem cells_"
TSC_SH_PPARG = (
    GSE295568_PREFIX
    + "TSCs were infected with PPARG KD1 shRNA lentivirus and maintained under self-renewal conditions."
)
TSC_SH_CTRL = (
    GSE295568_PREFIX
    + "TSCs were infected with non-targeting control shRNA lentivirus and maintained under self-renewal conditions."
)
RESCUE = (
    GSE295568_PREFIX
    + "TSCs were infected with PPARG KD1 shRNA lentivirus and post 72 hours shRNA-resistant "
    "PPARG was Dox-induced for 3 days, induced with {} ng/mL Dox."
)
EVT_SH_PPARG = (
    GSE295568_PREFIX
    + "PPARG KD1 shRNA lentiviral infection was performed 3–20 h after EVT induction to "
    "achieve sustained PPARG knockdown during differentiation."
)
EVT_DMSO = GSE295568_PREFIX + "Cells were differentiated into EVT for 8 days with DMSO (vehicle control)."

LEVELS = ["A", "B", "C", "D"]


def _names(levels=LEVELS):
    return level_name_map(levels)


# --- good expressions -------------------------------------------------------


@pytest.mark.parametrize(
    "expr,shape,weights",
    [
        ("A - B", "pairwise", {"A": 1.0, "B": -1.0}),
        ("-B + A", "pairwise", {"A": 1.0, "B": -1.0}),
        ("(A + B)/2 - C", "average", {"A": 0.5, "B": 0.5, "C": -1.0}),
        ("0.5*A + 0.5*B - C", "average", {"A": 0.5, "B": 0.5, "C": -1.0}),
        ("(A + B + C)/3 - D", "average", {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3, "D": -1.0}),
        ("(A - B) - (C - D)", "interaction", {"A": 1.0, "B": -1.0, "C": -1.0, "D": 1.0}),
        ("`A` - `B`", "pairwise", {"A": 1.0, "B": -1.0}),
    ],
)
def test_valid_expressions(expr, shape, weights):
    check = check_contrast("c1", expr, _names())
    assert check.is_valid, check.issues
    assert check.shape == shape
    assert check.weights == pytest.approx(weights)


# --- bad expressions --------------------------------------------------------


@pytest.mark.parametrize(
    "expr,code",
    [
        ("A + B - C - D", "unscaled"),
        ("2*A - 2*B", "unscaled"),
        ("A - E", "unknown_level"),
        ("A + B - C", "weights_not_zero_sum"),
        ("A", "weights_not_zero_sum"),
        ("A - A", "no_levels"),
        ("A * B - C", "parse_error"),
        ("A / B", "parse_error"),
        ("group == 'A'", "parse_error"),
        ("(A - B", "parse_error"),
        ("A - B + 1", "constant_term"),
        ("2A - B", "parse_error"),
    ],
)
def test_invalid_expressions(expr, code):
    check = check_contrast("c1", expr, _names())
    assert not check.is_valid
    assert check.shape == "invalid"
    assert code in {i.code for i in check.errors}


def test_unknown_level_suggests_a_close_level():
    check = check_contrast("c1", "DMS0 - Treated", _names(["DMSO", "Treated"]))
    messages = " ".join(i.message for i in check.errors)
    assert "'DMS0'" in messages and "Did you mean 'DMSO'" in messages


@pytest.mark.parametrize("name", ["1bad", "has space", "a-b", "_x", "TRUE"])
def test_invalid_contrast_names_are_rejected(name):
    check = check_contrast(name, "A - B", _names())
    assert "invalid_name" in {i.code for i in check.errors}


def test_duplicate_names_are_rejected():
    report = check_contrasts(
        [{"name": "x", "expression": "A - B"}, {"name": "x", "expression": "C - D"}], LEVELS
    )
    assert report.checks[0].is_valid
    assert "duplicate_name" in {i.code for i in report.checks[1].errors}
    assert not report.all_valid


def test_missing_expression_column_raises():
    with pytest.raises(ValueError, match="needs 'name' and 'expression'"):
        check_contrasts([{"name": "x"}], LEVELS)


# --- R naming ---------------------------------------------------------------


def test_raw_levels_are_renamed_like_rnaseq_r():
    names = level_name_map(["0.5 uM T0070907", "DMSO"])
    assert names == {"0.5 uM T0070907": "X0.5.uM.T0070907", "DMSO": "DMSO"}
    assert to_r_expression("0.5 uM T0070907 - DMSO", names) == "X0.5.uM.T0070907 - DMSO"
    check = check_contrast("T_vs_DMSO", "0.5 uM T0070907 - DMSO", names)
    assert check.is_valid
    assert check.weights == {"X0.5.uM.T0070907": 1.0, "DMSO": -1.0}


def test_already_sanitised_tokens_are_accepted():
    names = level_name_map(["Pim1-/-", "WT"])
    assert check_contrast("ko", "Pim1... - WT", names).is_valid
    assert check_contrast("ko", "Pim1-/- - WT", names).is_valid


def test_levels_that_collide_in_r_raise():
    with pytest.raises(ValueError, match="both become"):
        level_name_map(["a b", "a-b"])


@pytest.mark.parametrize("accession", ["GSE298581", "GSE166308", "GSE324244", "GSE132902"])
def test_on_disk_contrasts_of_simple_datasets_are_valid(accession):
    samples = pd.read_csv(FIXTURES / f"{accession}_samples.csv")
    contrasts = pd.read_csv(FIXTURES / f"{accession}_contrasts.csv")
    report = check_contrasts(contrasts, samples["uorca_group"].unique())
    assert report.all_valid, [c.to_dict() for c in report.checks if not c.is_valid]


# --- GSE295568: real sentence levels, nested batch --------------------------


@pytest.fixture
def gse295568():
    samples = pd.read_csv(FIXTURES / "GSE295568_samples.csv")
    contrasts = pd.read_csv(FIXTURES / "GSE295568_contrasts.csv")
    design = validate_design(samples, ["uorca_group"], [GSE295568_BATCH])
    return samples, contrasts, design


def test_gse295568_on_disk_contrasts_parse_and_stay_within_batch(gse295568):
    samples, contrasts, design = gse295568
    report = check_contrasts(
        contrasts, samples["uorca_group"].unique(), block_of_group=design.block_of_group()
    )
    assert report.all_valid, [c.to_dict() for c in report.checks if not c.is_valid]
    assert report.shape_counts() == {"pairwise": 16, "average": 1}
    assert not any(c.warnings for c in report.checks)


def test_gse295568_cross_batch_contrast_is_warned(gse295568):
    samples, _, design = gse295568
    names = level_name_map(samples["uorca_group"].unique())
    check = check_contrast(
        "evt_kd_vs_dmso", f"{EVT_SH_PPARG} - {EVT_DMSO}", names,
        block_of_group=design.block_of_group(),
    )
    assert check.is_valid
    assert [i.code for i in check.warnings] == ["crosses_nested_block"]
    assert "Batch2" in check.warnings[0].message and "Batch4" in check.warnings[0].message


def test_gse295568_on_disk_choice_has_dose_vs_dose_but_no_rescue_vs_knockdown(gse295568):
    """The canonical pairs show which comparisons the run produced.

    The run compared the two rescue doses with each other, and did not
    compare either rescue arm with shPPARG or shCtrl.
    """
    samples, contrasts, _ = gse295568
    report = check_contrasts(contrasts, samples["uorca_group"].unique())
    r = report.level_names
    pairs = report.pairs()
    assert ((r[RESCUE.format("12.5")],), (r[RESCUE.format("15")],)) in pairs
    assert ((r[TSC_SH_PPARG],), (r[TSC_SH_CTRL],)) in pairs
    for dose in ("12.5", "15"):
        assert ((r[RESCUE.format(dose)],), (r[TSC_SH_PPARG],)) not in pairs
        assert ((r[RESCUE.format(dose)],), (r[TSC_SH_CTRL],)) not in pairs
