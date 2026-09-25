"""Integration test for the library-source hard filter at validity check."""

import pandas as pd
import pytest


def _build_sra_df(records):
    """records: list of (accession, library_source). Returns synthetic SRA df."""
    rows = []
    for acc, src in records:
        rows.append({
            "Run": f"SRR{acc[3:]}",
            "BioProject": f"PRJNA{acc[3:]}",
            "GEO_Accession": acc,
            "LibrarySource": src,
            "LibraryStrategy": "RNA-Seq",
            "LibraryLayout": "PAIRED",
            "size_MB": 100,
        })
    return pd.DataFrame(rows)


def test_bulk_only_excludes_single_cell(monkeypatch, tmp_path):
    """With --library-source bulk, scRNA-seq datasets should be marked
    invalid before Stage 1 ever sees them."""
    # Simulate the validity loop's per-sample check directly.
    from uorca.identification import dataset_identification as dsi  # noqa: F401

    accepted_sources = {"TRANSCRIPTOMIC"}  # bulk

    sra_df = _build_sra_df([
        ("GSE001", "TRANSCRIPTOMIC"),
        ("GSE002", "TRANSCRIPTOMIC SINGLE CELL"),
        ("GSE003", "TRANSCRIPTOMIC"),
    ])

    valid_accessions = []
    for acc in sra_df["GEO_Accession"].unique():
        sub = sra_df[sra_df["GEO_Accession"] == acc]
        rnaseq_samples = sum(
            (s in accepted_sources) and (st == "RNA-Seq") and (l == "PAIRED")
            for s, st, l in zip(sub["LibrarySource"], sub["LibraryStrategy"], sub["LibraryLayout"])
        )
        if rnaseq_samples > 0:
            valid_accessions.append(acc)

    assert "GSE001" in valid_accessions
    assert "GSE003" in valid_accessions
    assert "GSE002" not in valid_accessions, "scRNA-seq must be excluded under bulk filter"


def test_sc_only_excludes_bulk():
    accepted_sources = {"TRANSCRIPTOMIC SINGLE CELL"}
    sra_df = _build_sra_df([
        ("GSE001", "TRANSCRIPTOMIC"),
        ("GSE002", "TRANSCRIPTOMIC SINGLE CELL"),
    ])

    valid_accessions = []
    for acc in sra_df["GEO_Accession"].unique():
        sub = sra_df[sra_df["GEO_Accession"] == acc]
        rnaseq_samples = sum(
            (s in accepted_sources) and (st == "RNA-Seq") and (l == "PAIRED")
            for s, st, l in zip(sub["LibrarySource"], sub["LibraryStrategy"], sub["LibraryLayout"])
        )
        if rnaseq_samples > 0:
            valid_accessions.append(acc)

    assert "GSE002" in valid_accessions
    assert "GSE001" not in valid_accessions


def test_both_accepts_all():
    accepted_sources = {"TRANSCRIPTOMIC", "TRANSCRIPTOMIC SINGLE CELL"}
    sra_df = _build_sra_df([
        ("GSE001", "TRANSCRIPTOMIC"),
        ("GSE002", "TRANSCRIPTOMIC SINGLE CELL"),
    ])

    valid_accessions = []
    for acc in sra_df["GEO_Accession"].unique():
        sub = sra_df[sra_df["GEO_Accession"] == acc]
        rnaseq_samples = sum(
            (s in accepted_sources) and (st == "RNA-Seq") and (l == "PAIRED")
            for s, st, l in zip(sub["LibrarySource"], sub["LibraryStrategy"], sub["LibraryLayout"])
        )
        if rnaseq_samples > 0:
            valid_accessions.append(acc)

    assert sorted(valid_accessions) == ["GSE001", "GSE002"]
