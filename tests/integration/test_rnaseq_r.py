"""Run the real RNAseq.R on small synthetic Kallisto output.

Each case writes fake ``abundance.tsv`` files (fixed seed), a sample table
and a contrasts file into a temporary directory, runs
``Rscript uorca/analysis/scripts/RNAseq.R`` and checks that the run either
gives correct results or stops with a clear error. Group levels mirror the
real failures of GSE161060 (two levels that ``make.names`` merges) and
GSE210935 (levels that are prefixes of each other, digit-leading contrast
names), plus a level named ``2`` used inside ``(2 + 4)/2``.

Skipped, with the reason shown, when ``Rscript`` or the R packages are
missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RNASEQ_R = Path(__file__).resolve().parents[2] / "uorca" / "analysis" / "scripts" / "RNAseq.R"
R_PACKAGES = ("limma", "edgeR", "tximport", "ComplexHeatmap", "gplots", "circlize")


def _r_skip_reason() -> str | None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return "Rscript is not on PATH"
    check = "for (p in c(%s)) if (!requireNamespace(p, quietly = TRUE)) cat(p, '')" % ", ".join(
        f'"{p}"' for p in R_PACKAGES
    )
    out = subprocess.run([rscript, "-e", check], capture_output=True, text=True, timeout=120)
    missing = out.stdout.strip()
    return f"R packages missing: {missing}" if missing else None


_SKIP = _r_skip_reason()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.r,
    pytest.mark.slow,
    pytest.mark.skipif(_SKIP is not None, reason=_SKIP or ""),
]

N_GENES = 400
UP_GENE = "tx_up"
"""Planted gene: 8-fold higher in every level listed in ``up_levels``."""


def _write_inputs(
    tmp: Path,
    groups: list[str],
    *,
    group_col: str,
    contrasts: list[tuple[str, str]] | None,
    up_levels: dict[str, float] | None = None,
    reps: int = 3,
    extra_cols: dict[str, list[str]] | None = None,
) -> list[str]:
    """Write abundance files, the sample table and contrasts; return the R args."""
    rng = np.random.default_rng(20261028)
    base = rng.uniform(50, 400, N_GENES)
    targets = [UP_GENE] + [f"tx{i}" for i in range(1, N_GENES)]
    rows = []
    for g in groups:
        for r in range(reps):
            sample = f"S_{len(rows) + 1:02d}"
            mu = base.copy()
            mu[0] *= (up_levels or {}).get(g, 1.0)
            counts = rng.negative_binomial(20, 20 / (20 + mu)).astype(float)
            d = tmp / "abundance" / sample
            d.mkdir(parents=True)
            pd.DataFrame(
                {
                    "target_id": targets,
                    "length": 1500,
                    "eff_length": 1300.0,
                    "est_counts": counts,
                    "tpm": counts / counts.sum() * 1e6,
                }
            ).to_csv(d / "abundance.tsv", sep="\t", index=False)
            rows.append(
                {"abundance_file": str(d / "abundance.tsv"), "geo_accession": f"GSM{len(rows) + 1}", group_col: g}
            )
    samples = pd.DataFrame(rows, index=pd.Index([r["geo_accession"] for r in rows]))
    for col, values in (extra_cols or {}).items():
        samples[col] = [v for v in values for _ in range(reps)]
    (tmp / "metadata").mkdir()
    samples.to_csv(tmp / "metadata" / "edger_analysis_samples.csv")
    args = [str(tmp / "metadata" / "edger_analysis_samples.csv"), group_col, str(tmp), "NA"]
    if contrasts is not None:
        pd.DataFrame(
            {"name": [c[0] for c in contrasts], "expression": [c[1] for c in contrasts]}
        ).to_csv(
            tmp / "metadata" / "contrasts.csv", index=False
        )
        args.append(str(tmp / "metadata" / "contrasts.csv"))
    return args


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "LC_ALL": "en_US.UTF-8"}
    return subprocess.run(
        ["Rscript", str(RNASEQ_R), *args], capture_output=True, text=True, timeout=600, cwd=cwd, env=env
    )


def _deg(tmp: Path, folder: str) -> pd.DataFrame:
    return pd.read_csv(tmp / "RNAseqAnalysis" / folder / "DEG.csv").set_index("Gene")


def _contrast_folders(tmp: Path) -> set[str]:
    d = tmp / "RNAseqAnalysis"
    return {p.name for p in d.iterdir() if (p / "DEG.csv").is_file()} if d.is_dir() else set()


def _assert_stops(proc: subprocess.CompletedProcess[str], *fragments: str) -> None:
    assert proc.returncode != 0, proc.stdout[-2000:]
    for f in fragments:
        assert f in proc.stderr, proc.stderr[-3000:]


# --- GSE161060-like: two levels that make.names merges (F1) ---------------


def test_gse161060_raw_levels_stop_with_collision_error(tmp_path):
    pos, neg = "CD45+F4/80+CD11b+EGFP+", "CD45+F4/80+CD11b+EGFP-"
    args = _write_inputs(
        tmp_path, [neg, pos], group_col="cell_type",
        contrasts=[("EGFP_pos_vs_EGFP_neg", f"{pos} - {neg}")],
    )
    proc = _run(args, tmp_path)
    _assert_stops(proc, "same R name", pos, neg)
    assert _contrast_folders(tmp_path) == set()


def test_gse161060_assigned_names_give_correct_results(tmp_path):
    pos, neg = "g02_CD45posF4_80posCD11bposEGFPpos", "g01_CD45posF4_80posCD11bposEGFPneg"
    args = _write_inputs(
        tmp_path, [neg, pos], group_col="uorca_group",
        contrasts=[("EGFP_pos_vs_EGFP_neg", f"{pos} - {neg}")],
        up_levels={pos: 8.0},
    )
    proc = _run(args, tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "nothing renamed" in proc.stdout
    deg = _deg(tmp_path, "EGFP_pos_vs_EGFP_neg")
    assert deg.loc[UP_GENE, "logFC"] == pytest.approx(3.0, abs=0.5)
    assert deg.loc[UP_GENE, "adj.P.Val"] < 0.01


def test_partial_collision_stops(tmp_path):
    """F2: Tg+ and Tg- would be pooled while the run succeeds."""
    args = _write_inputs(
        tmp_path, ["Tg+", "Tg-", "WT"], group_col="genotype",
        contrasts=[("Tgpos_vs_WT", "Tg+ - WT"), ("Tgneg_vs_WT", "Tg- - WT")],
    )
    _assert_stops(_run(args, tmp_path), "same R name", "Tg+", "Tg-")


# --- GSE210935-like: prefix levels (F3) and digit-leading names (F4) ------

GSE210935_LEVELS = ["8h_GM", "8h_GMTGW", "8h_GMIL4", "MoDC_GMT"]


def test_gse210935_digit_leading_name_stops_clearly(tmp_path):
    args = _write_inputs(
        tmp_path, GSE210935_LEVELS, group_col="treatment",
        contrasts=[("8h_GMIL4_vs_8h_GM", "8h_GMIL4 - 8h_GM")],
    )
    _assert_stops(_run(args, tmp_path), "not a valid R name", "8h_GMIL4_vs_8h_GM")


def test_gse210935_raw_expressions_are_rewritten_without_corruption(tmp_path):
    """Legacy input (raw levels, as in a download bundle of an old run)."""
    args = _write_inputs(
        tmp_path, GSE210935_LEVELS, group_col="treatment",
        contrasts=[("GMTGW_vs_GM_8h", "8h_GMTGW - 8h_GM"), ("GMTGW_vs_GMIL4_8h", "8h_GMTGW - 8h_GMIL4")],
        up_levels={"8h_GMTGW": 8.0},
    )
    proc = _run(args, tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "XXX8h" not in proc.stdout
    assert _deg(tmp_path, "GMTGW_vs_GM_8h").loc[UP_GENE, "logFC"] == pytest.approx(3.0, abs=0.5)
    assert _deg(tmp_path, "GMTGW_vs_GMIL4_8h").loc[UP_GENE, "logFC"] == pytest.approx(3.0, abs=0.5)


def test_gse210935_assigned_names_give_correct_results(tmp_path):
    args = _write_inputs(
        tmp_path, ["X8h_GM", "X8h_GMTGW", "X8h_GMIL4", "MoDC_GMT"], group_col="uorca_group",
        contrasts=[("c_GMTGW_vs_GM_8h", "X8h_GMTGW - X8h_GM"), ("MoDC_GMT_vs_GM_8h", "MoDC_GMT - X8h_GM")],
        up_levels={"X8h_GMTGW": 8.0},
    )
    proc = _run(args, tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert _deg(tmp_path, "c_GMTGW_vs_GM_8h").loc[UP_GENE, "logFC"] == pytest.approx(3.0, abs=0.5)
    assert abs(_deg(tmp_path, "MoDC_GMT_vs_GM_8h").loc[UP_GENE, "logFC"]) < 0.5


# --- a level named "2" inside "(2 + 4)/2 - 0" (F5) -------------------------


def test_numeric_level_divisor_stops_instead_of_nan_weights(tmp_path):
    args = _write_inputs(
        tmp_path, ["0", "2", "4"], group_col="dose",
        contrasts=[("dose_avg_vs_zero", "(2 + 4)/2 - 0")],
    )
    _assert_stops(_run(args, tmp_path), "dose_avg_vs_zero")
    assert _contrast_folders(tmp_path) == set()


def test_numeric_level_with_assigned_names_averages_correctly(tmp_path):
    args = _write_inputs(
        tmp_path, ["X0", "X2", "X4"], group_col="uorca_group",
        contrasts=[("dose_avg_vs_zero", "(X2 + X4)/2 - X0")],
        up_levels={"X2": 4.0, "X4": 16.0},
    )
    proc = _run(args, tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    # mean(log2 4, log2 16) - 0 = 3
    assert _deg(tmp_path, "dose_avg_vs_zero").loc[UP_GENE, "logFC"] == pytest.approx(3.0, abs=0.5)


# --- names that collide as folders (F9-F11) and code injection (F7, F8) ---


@pytest.mark.parametrize(
    "names",
    [("KO.vs.WT", "KO_vs_WT"), ("KO_vs_WT", "ko_vs_wt"), ("KO_vs_WT", "KO_vs_WT")],
)
def test_contrast_folder_collisions_stop(tmp_path, names):
    args = _write_inputs(
        tmp_path, ["KO", "WT"], group_col="uorca_group",
        contrasts=[(names[0], "KO - WT"), (names[1], "WT - KO")],
    )
    _assert_stops(_run(args, tmp_path), "same folder")
    assert _contrast_folders(tmp_path) == set()


@pytest.mark.parametrize(
    "name, expression",
    [
        ('x = "KO - WT", y', "KO - WT"),
        ("KO_vs_WT", "KO - WT + 0*nchar(file.create('PWNED'))"),
        ("KO_vs_WT", 'KO - "WT"'),
        ("KO_vs_WT", "KO == WT"),
    ],
)
def test_injected_code_is_not_run(tmp_path, name, expression):
    args = _write_inputs(tmp_path, ["KO", "WT"], group_col="uorca_group", contrasts=[(name, expression)])
    proc = _run(args, tmp_path)
    _assert_stops(proc)
    assert not (tmp_path / "PWNED").exists()
    assert "Error" in proc.stderr


def test_malformed_contrasts_file_stops(tmp_path):
    args = _write_inputs(tmp_path, ["KO", "WT"], group_col="uorca_group", contrasts=[("KO_vs_WT", "KO - WT")])
    pd.DataFrame({"name": ["KO_vs_WT"], "expr": ["KO - WT"]}).to_csv(args[-1], index=False)
    _assert_stops(_run(args, tmp_path), "name", "expression")


def test_literal_na_level_is_a_group(tmp_path):
    """F15: a level spelled NA must not be read as a missing value."""
    args = _write_inputs(
        tmp_path, ["NA", "WT"], group_col="genotype",
        contrasts=[("NA_vs_WT", "NA - WT")], up_levels={"NA": 8.0},
    )
    proc = _run(args, tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert _deg(tmp_path, "NA_vs_WT").loc[UP_GENE, "logFC"] == pytest.approx(3.0, abs=0.5)


def test_two_group_run_without_contrasts_file(tmp_path):
    """F21: the two-group fallback used '+' on strings and failed."""
    args = _write_inputs(tmp_path, ["A", "B"], group_col="uorca_group", contrasts=None, up_levels={"B": 8.0})
    proc = _run(args, tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert _contrast_folders(tmp_path) == {"B_vs_A"}
    assert _deg(tmp_path, "B_vs_A").loc[UP_GENE, "logFC"] == pytest.approx(3.0, abs=0.5)


def test_manifest_and_session_info_are_written(tmp_path):
    args = _write_inputs(
        tmp_path, ["KO", "WT"], group_col="uorca_group",
        contrasts=[("KO_vs_WT", "KO - WT"), ("WT_vs_KO", "WT - KO")],
    )
    proc = _run(args, tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    manifest = pd.read_csv(tmp_path / "RNAseqAnalysis" / "contrast_manifest.csv")
    assert manifest["contrast_name"].tolist() == ["KO_vs_WT", "WT_vs_KO"]
    assert manifest["folder"].tolist() == ["KO_vs_WT", "WT_vs_KO"]
    assert (manifest["n_samples"] == 6).all()
    assert (manifest["n_genes"] > 0).all()
    assert "limma" in (tmp_path / "RNAseqAnalysis" / "sessionInfo.txt").read_text()


# --- download bundle (audit §5.5) -------------------------------------------


def _bundle_and_run(results_dir: Path, dataset_id: str, out: Path) -> subprocess.CompletedProcess[str]:
    """Build the R bundle as the explorer does, unzip it and run its RNAseq.R."""
    import json
    import zipfile

    from uorca.gui.components.helpers import add_rscript_bundle_to_zip

    zip_path = out / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        add_rscript_bundle_to_zip(zf, None, str(results_dir), dataset_id, dataset_id, "pkg")  # type: ignore[arg-type]
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    bundle = out / "pkg"
    assert json.loads((bundle / "metadata" / "analysis_info.json").read_text())["accession"] == dataset_id
    return subprocess.run(
        ["Rscript", "RNAseq.R"], capture_output=True, text=True, timeout=600, cwd=bundle,
        env={**os.environ, "LC_ALL": "en_US.UTF-8"},
    )


def _write_analysis_info(dataset: Path, merged_column: str, contrasts: list[tuple[str, str]]) -> None:
    import json

    (dataset / "metadata" / "analysis_info.json").write_text(json.dumps({
        "accession": dataset.name, "organism": "Unknown", "analysis_success": True,
        "merged_column": merged_column, "unique_groups": [],
        "contrasts": [{"name": n, "expression": e} for n, e in contrasts],
    }))


def test_bundle_of_new_run_uses_r_level_names(tmp_path):
    """GSE161060-like: raw levels collide under make.names; the bundle must use uorca_group."""
    import json

    from uorca.core.naming import assign_level_names, build_naming_record

    pos, neg = "CD45+F4/80+CD11b+EGFP+", "CD45+F4/80+CD11b+EGFP-"
    names = assign_level_names([neg, pos])
    contrasts = [("EGFP_pos_vs_EGFP_neg", f"{names[pos]} - {names[neg]}")]
    dataset = tmp_path / "results" / "GSE000001"
    dataset.mkdir(parents=True)
    _write_inputs(
        dataset, [names[neg], names[pos]], group_col="uorca_group", contrasts=contrasts,
        up_levels={names[pos]: 8.0}, extra_cols={"cell_type": [neg, pos]},
    )
    _write_analysis_info(dataset, "cell_type", contrasts)
    record = build_naming_record(
        group_column="cell_type", source_columns=["cell type"], level_names=names,
        level_counts={pos: 3, neg: 3}, contrasts=[{"name": n, "expression": e} for n, e in contrasts],
    )
    (dataset / "metadata" / "naming.json").write_text(json.dumps(record))

    out = tmp_path / "unzipped"
    out.mkdir()
    proc = _bundle_and_run(tmp_path / "results", "GSE000001", out)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert 'merged_group <- "uorca_group"' in (out / "pkg" / "RNAseq.R").read_text()
    deg = _deg(out / "pkg", "EGFP_pos_vs_EGFP_neg")
    assert deg.loc[UP_GENE, "logFC"] == pytest.approx(3.0, abs=0.5)


def test_bundle_of_old_run_keeps_merged_column(tmp_path):
    """No naming.json: the bundle still uses merged_column and raw expressions."""
    contrasts = [("SARS_vs_Mock", "SARS-CoV-2 - Mock")]
    dataset = tmp_path / "results" / "GSE000002"
    dataset.mkdir(parents=True)
    _write_inputs(dataset, ["Mock", "SARS-CoV-2"], group_col="infection", contrasts=contrasts,
                  up_levels={"SARS-CoV-2": 8.0})
    _write_analysis_info(dataset, "infection", contrasts)

    out = tmp_path / "unzipped"
    out.mkdir()
    proc = _bundle_and_run(tmp_path / "results", "GSE000002", out)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert 'merged_group <- "infection"' in (out / "pkg" / "RNAseq.R").read_text()
    assert _deg(out / "pkg", "SARS_vs_Mock").loc[UP_GENE, "logFC"] == pytest.approx(3.0, abs=0.5)


# --- legacy expressions with non-ASCII letters (GSE295568) ------------------


def test_gse295568_legacy_contrasts_with_micro_sign_run(tmp_path):
    """13 of GSE295568's 17 contrasts contain 'μ' and ran on the old script.

    Raw levels and the real contrasts.csv (name and expression columns, equal
    to results/20261007_PPARg/GSE295568/metadata/contrasts.csv) as a download
    bundle of that run would pass them.
    """
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "contrast_design"
    levels = sorted(pd.read_csv(fixtures / "GSE295568_samples.csv", dtype=str)["uorca_group"].unique())
    real = pd.read_csv(fixtures / "GSE295568_contrasts.csv", dtype=str)
    assert sum("μ" in e for e in real["expression"]) == 13
    args = _write_inputs(
        tmp_path, levels, group_col="experimental_group", reps=2,
        contrasts=list(zip(real["name"], real["expression"])),
    )
    proc = _run(args, tmp_path)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert _contrast_folders(tmp_path) == {str(n).replace(".", "_") for n in real["name"]}
