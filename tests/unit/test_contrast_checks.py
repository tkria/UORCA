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

LEVELS = ["A", "B", "C", "D", "E", "F"]


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
        ("A - B - C + D", "interaction", {"A": 1.0, "B": -1.0, "C": -1.0, "D": 1.0}),
        ("(A + D) - (B + C)", "interaction", {"A": 1.0, "B": -1.0, "C": -1.0, "D": 1.0}),
        ("((A - B) - (C - D))", "interaction", {"A": 1.0, "B": -1.0, "C": -1.0, "D": 1.0}),
        # Weights alone cannot tell this from (A + C) - (B + D) = (A - B) - (D - C).
        ("(A - B) + (C - D)", "interaction", {"A": 1.0, "B": -1.0, "C": 1.0, "D": -1.0}),
        ("A + B - C - D", "interaction", {"A": 1.0, "B": 1.0, "C": -1.0, "D": -1.0}),
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
        ("2*A - 2*B", "unscaled"),
        ("(A - B) + (A - B)", "unscaled"),
        ("(A - B) - (C - D) - (E - F)", "unscaled"),
        ("(A + B) - C/2", "weights_not_zero_sum"),
        ("A + B + C - 3*D", "unscaled"),
        ("A - G", "unknown_level"),
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


# --- evaluation-set cases (tests/eval/contrast_design) ----------------------

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval" / "contrast_design"
EVAL_CASES = sorted(p.stem for p in EVAL_DIR.glob("GSE*.yaml"))
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


def _load_case(accession: str) -> dict:
    import yaml

    return yaml.safe_load((EVAL_DIR / f"{accession}.yaml").read_text(encoding="utf-8"))


def _group_members(samples: pd.DataFrame, selector: dict) -> pd.Series:
    mask = pd.Series(True, index=samples.index)
    for col, value in selector.items():
        mask &= samples[col].astype(str) == str(value)
    return mask


def _case_pair(item: dict, label_to_r: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(sorted(label_to_r[g] for g in item["target"])),
        tuple(sorted(label_to_r[g] for g in item["reference"])),
    )


def test_eval_set_has_the_five_cases():
    assert EVAL_CASES == sorted(PIPELINE_DESIGN_COLUMNS)


@pytest.mark.parametrize("accession", EVAL_CASES)
def test_eval_case_matches_fixture_and_current_run(accession):
    from uorca.core.review_flags import compute_review_flags

    case = _load_case(accession)
    assert case["accession"] == accession
    assert "owner to confirm" in case["status"]
    samples = pd.read_csv(FIXTURES / f"{accession}_samples.csv")

    # Every group selects samples, groups are disjoint, and each maps to one
    # level of the run's group column.
    claimed = pd.Series(0, index=samples.index)
    label_to_level: dict[str, str] = {}
    for label, selector in case["groups"].items():
        members = _group_members(samples, selector)
        assert members.sum() >= 2, f"{label} selects {members.sum()} samples"
        claimed += members.astype(int)
        levels = samples.loc[members, "uorca_group"].unique()
        assert len(levels) == 1, f"{label} spans run groups {levels}"
        label_to_level[label] = str(levels[0])
    assert claimed.max() == 1

    # The expected design is what validate_design produces.
    design_spec = case["design"]
    blocks = design_spec["required_blocks"] + [d["column"] for d in design_spec.get("must_drop", [])]
    design = validate_design(samples, design_spec["group_columns"], blocks)
    assert design.formula == design_spec["expected_formula"]
    assert design.residual_df == design_spec["expected_residual_df"]
    dropped = {d.column: d.reason_code for d in design.dropped_terms}
    for item in design_spec.get("must_drop", []):
        assert dropped[item["column"]] == item["reason_code"]

    # The review flag expectation holds for the run's grouping.
    flags = compute_review_flags(samples, "uorca_group", PIPELINE_DESIGN_COLUMNS[accession])
    assert flags.needs_review is design_spec["needs_review_expected"]

    # The "before" numbers in current_run match the stored contrasts.
    contrasts = pd.read_csv(FIXTURES / f"{accession}_contrasts.csv")
    report = check_contrasts(contrasts, samples["uorca_group"].unique())
    label_to_r = {lbl: report.level_names[lvl] for lbl, lvl in label_to_level.items()}
    required = [_case_pair(item, label_to_r) for item in case["contrasts"]["required"]]
    present = report.pairs()
    run = case["current_run"]
    assert run["contrasts"] == len(contrasts)
    assert sum(p in present for p in required) == run["required_present"]
    assert len(required) - run["required_present"] == len(run["required_missing"])


def test_identical_weight_vectors_get_the_same_shape():
    names = _names()
    forms = ["A - B - C + D", "(A + D) - (B + C)", "((A - B) - (C - D))", "(A - B) - (C - D)"]
    checks = [check_contrast("c", f, names) for f in forms]
    assert {c.shape for c in checks} == {"interaction"}
    assert len({c.canonical_pair() for c in checks}) == 1


# --- sanitisation faults (GSE210935) ---------------------------------------


def test_gse210935_prefix_levels_are_a_sanitisation_fault():
    from uorca.core.contrast_checks import sanitisation_faults

    samples = pd.read_csv(FIXTURES / "GSE210935_samples.csv")
    contrasts = pd.read_csv(FIXTURES / "GSE210935_contrasts.csv")
    names = level_name_map(samples["uorca_group"].unique())
    faults = sanitisation_faults(names)
    # What the R log of the 2026-10-07 run shows.
    assert faults["8h_GMTGW"][0] == "XXX8h_GMTGW"
    assert faults["8h_GMIL4"][0] == "XX8h_GMIL4"
    assert "8h_GM" in faults["8h_GMTGW"][1]
    assert "8h_GM" not in faults and not any(k.startswith("MoDC") for k in faults)

    report = check_contrasts(contrasts, samples["uorca_group"].unique())
    by_expr = {c.expression: c for c in report.checks}
    bad = by_expr["8h_GMTGW - 8h_GM"]
    codes = {i.code for i in bad.errors}
    assert "sanitisation_fault" in codes and "unknown_level" not in codes
    msg = next(i.message for i in bad.errors if i.code == "sanitisation_fault")
    assert "'8h_GM'" in msg and "'8h_GMTGW'" in msg and "XXX8h_GMTGW" in msg
    assert by_expr["MoDC_GMTGW - MoDC_GMT"].is_valid


def test_unknown_token_is_still_unknown_level_without_a_fault():
    check = check_contrast("c", "8h_GMX - 8h_GM", level_name_map(["8h_GM", "8h_GMT"]))
    assert {i.code for i in check.errors} >= {"unknown_level"}
    assert "sanitisation_fault" not in {i.code for i in check.errors}


def test_missing_block_mapping_raises():
    with pytest.raises(ValueError, match=r"no mapping for group levels \['C'\]"):
        check_contrast("c", "A - C", _names(), block_of_group={"batch": {"A": "b1", "B": "b1"}})
