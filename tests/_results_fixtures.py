"""Build fake UORCA results folders for import tests (fake text only, no real data)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

FAKE_QUESTION = "Which genes change when fake compound Z is applied to imaginary cells?"

_GENES = [f"FAKEGENE{i}" for i in range(1, 13)]


# Mouse-style symbols (first letter capital) for the mouse fixture.
_MOUSE_GENES = [f"Fakegene{i}" for i in range(1, 13)]

# Long contrast names modelled on the shape of real ones (GSE295568): long, and sharing long
# prefixes, so that wrapped or truncated labels collide. Fake text only.
LONG_CONTRASTS: tuple[str, ...] = (
    "FAKECOMPOUND_Z_knockdown_clone1_vs_control_stem_cell_self_renewal",
    "FAKECOMPOUND_Z_knockdown_clone1_vs_control_stem_cell_differentiation",
    "FAKECOMPOUND_Z_knockdown_clone1_REGULATOR_overexpression_vs_control",
    "FAKECOMPOUND_Z_rescue_12p5_vs_15_doxycycline_trophoblast_lineage",
)


def _write_deg(path: Path, seed: int, genes: list[str] = _GENES, adj_p: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["Gene", "logFC", "AveExpr", "t", "P.Value"] + (["adj.P.Val"] if adj_p else []) + ["B"]
    lines = [",".join(cols)]
    for i, gene in enumerate(genes):
        lfc = ((i + seed) % 7 - 3) * 1.1
        p = 10 ** (-(i % 6) - 1)
        adj = [f"{min(1.0, p * 4):.2e}"] if adj_p else []
        lines.append(",".join([gene, f"{lfc:.3f}", f"{5 + i * 0.3:.2f}", f"{lfc * 3:.2f}",
                               f"{p:.2e}", *adj, f"{2.0 - i * 0.1:.2f}"]))
    path.write_text("\n".join(lines) + "\n")


def _write_cpm(path: Path, genes: list[str] = _GENES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Gene,Sample1,Sample2,Sample3,Sample4"]
    for i, gene in enumerate(genes):
        lines.append(f"{gene},{10 + i},{12 + i},{30 + 2 * i},{33 + 2 * i}")
    path.write_text("\n".join(lines) + "\n")


def make_dataset(
    parent: Path,
    accession: str,
    *,
    success: bool = True,
    cpm: bool = True,
    contrasts: tuple[str, ...] = ("Treated_vs_Control",),
    analysis_info: bool = True,
    null_groups: bool = False,
    organism: str = "Homo sapiens",
    adj_p: bool = True,
) -> Path:
    """Write one dataset folder <parent>/<accession>/ and return it.

    analysis_info.json follows the schema written by uorca/graph/nodes/evaluate.py, including
    the ``contrasts`` list. ``null_groups`` writes ``"unique_groups": null`` (older results),
    which the Explorer skips. ``organism="Mus musculus"`` uses mouse-style gene symbols.
    ``adj_p=False`` writes DEG tables without the ``adj.P.Val`` column.
    """
    genes = _MOUSE_GENES if organism == "Mus musculus" else _GENES
    ds = parent / accession
    rnaseq = ds / "RNAseqAnalysis"
    rnaseq.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(contrasts):
        _write_deg(rnaseq / name / "DEG.csv", seed=k, genes=genes, adj_p=adj_p)
    if cpm:
        _write_cpm(rnaseq / "CPM.csv", genes=genes)
    meta = ds / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    if analysis_info:
        (meta / "analysis_info.json").write_text(json.dumps({
            "accession": accession,
            "organism": organism,
            "analysis_success": success,
            "merged_column": "group",
            "unique_groups": None if null_groups else ["Control", "Treated"],
            "number_of_samples": 4,
            "dataset_information": f"Fake dataset {accession} for import tests.",
            "contrasts": [
                {
                    "name": c,
                    "expression": "Treated - Control",
                    "description": f"Fake contrast {c}",
                    "justification": f"Fake justification for {c}",
                }
                for c in contrasts
            ],
        }))
    (meta / "contrasts.csv").write_text(
        "name,expression,description\n"
        + "".join(f"{c},Treated - Control,Fake contrast {c}\n" for c in contrasts)
    )
    (meta / "edger_analysis_samples.csv").write_text(
        ",sample,group\n0,Sample1,Control\n1,Sample2,Control\n2,Sample3,Treated\n3,Sample4,Treated\n"
    )
    return ds


def make_analysis_folder(
    root: Path,
    *,
    complete: tuple[str, ...] = ("GSE900001", "GSE900002"),
    partial: tuple[str, ...] = ("GSE900003",),
    failed: tuple[str, ...] = (),
    status_only: tuple[str, ...] = (),
    research_question: Optional[str] = FAKE_QUESTION,
) -> Path:
    """Layout (a): <root>/GSE*/RNAseqAnalysis/ plus job_status and research_question.json.

    ``partial`` datasets model a real failure: three DEG tables written, then the R step failed
    (analysis_success false, no CPM.csv). ``failed`` datasets have metadata only.
    ``status_only`` accessions appear only in job_status/.
    """
    root.mkdir(parents=True, exist_ok=True)
    for acc in complete:
        make_dataset(root, acc)
    for acc in partial:
        make_dataset(root, acc, success=False, cpm=False, contrasts=("A_vs_B", "A_vs_C", "B_vs_C"))
    for acc in failed:
        meta = root / acc / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "analysis_info.json").write_text(json.dumps({"accession": acc, "analysis_success": False}))
    status_dir = root / "job_status"
    status_dir.mkdir(exist_ok=True)
    for acc in (*complete, *partial, *failed, *status_only):
        state = "completed" if acc in complete else "failed"
        (status_dir / f"{acc}_status.json").write_text(json.dumps({"state": state}))
    if research_question is not None:
        (root / "research_question.json").write_text(json.dumps({"research_question": research_question}))
    return root


def make_arm_folder(root: Path, *, identify: bool = True, **kwargs) -> Path:
    """Layout (b): <root>/analysis/ (as layout a) and optionally <root>/identify/."""
    make_analysis_folder(root / "analysis", **kwargs)
    if identify:
        ident = root / "identify"
        ident.mkdir(parents=True, exist_ok=True)
        (ident / "selected_datasets.csv").write_text("Accession\nGSE900001\nGSE900002\nGSE900003\n")
        (ident / "Dataset_identification_result.csv").write_text(
            "Accession,Title,Summary,RelevanceScore,Valid\n"
            "GSE900001,Fake title one,Fake summary one,8,Yes\n"
            "GSE900002,Fake title two,Fake summary two,7,Yes\n"
        )
        (ident / "identification_metadata.json").write_text(json.dumps({"note": "fake"}))
    return root


# ---------------------------------------------------------------------------
# Variants for the defects in plans 2-5 (plan 6, section 4.4). Each writes one complete
# dataset; combine them with make_analysis_folder(..., complete=()) or call them directly.
# ---------------------------------------------------------------------------


def make_long_contrast_dataset(parent: Path, accession: str = "GSE900010") -> Path:
    """A complete human dataset whose contrasts have long names with shared prefixes."""
    return make_dataset(parent, accession, contrasts=LONG_CONTRASTS)


def make_mouse_dataset(parent: Path, accession: str = "GSE900020") -> Path:
    """A complete Mus musculus dataset with mouse-style gene symbols."""
    return make_dataset(parent, accession, organism="Mus musculus")


def make_no_adj_p_dataset(parent: Path, accession: str = "GSE900030") -> Path:
    """A complete human dataset whose DEG.csv has no ``adj.P.Val`` column."""
    return make_dataset(parent, accession, adj_p=False)


def make_variants_folder(root: Path) -> Path:
    """Layout (a) with one normal human dataset plus each variant above (4 complete datasets)."""
    make_analysis_folder(root, complete=("GSE900001",), partial=())
    make_long_contrast_dataset(root)
    make_mouse_dataset(root)
    make_no_adj_p_dataset(root)
    status_dir = root / "job_status"
    for acc in ("GSE900010", "GSE900020", "GSE900030"):
        (status_dir / f"{acc}_status.json").write_text(json.dumps({"state": "completed"}))
    return root
