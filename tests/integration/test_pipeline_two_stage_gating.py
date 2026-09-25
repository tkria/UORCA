"""End-to-end integration test for the new two-stage gating logic with a mocked LLM."""

import asyncio
import json

import pandas as pd
import pytest


def _make_mock_dataset_summary(n: int) -> pd.DataFrame:
    """50-row synthetic dataset summary mimicking the post-validity dataframe."""
    rows = []
    for i in range(n):
        rows.append({
            "ID": f"GSE{i:03d}",
            "Title": f"Title {i}",
            "Summary": f"Summary {i}",
            "Species": "Homo sapiens",
            "Date": "2025-01-01",
            "NumSamples": 8,
            "PrimaryPubMedID": None,
            "DatasetSizeBytes": 0,
            "DatasetSizeGB": 0,
            "valid_dataset": True,
            "validation_reason": "OK",
            "rnaseq_samples": 8,
            "total_samples": 8,
            "Accession": f"GSE{i:03d}",
        })
    return pd.DataFrame(rows)


def test_threshold_gate_promotes_correct_subset(monkeypatch, tmp_path):
    """With deterministic Stage 1 scores by accession, only datasets with
    BiologyScore >= threshold should reach Stage 2."""
    from uorca.identification import dataset_identification as dsi
    from uorca.identification.scoring import (
        BiologyAssessment,
        BiologyAssessments,
        RelevanceAssessment,
        RelevanceAssessments,
    )

    def fake_call_structured(prompt, user_input, output_type):
        # Parse the JSON list of dataset entries from the user_input prefix
        ids = [d["ID"] for d in json.loads(user_input.split("Datasets:\n", 1)[1])]
        if output_type is BiologyAssessments:
            # Stage 1: assign biology by accession prefix
            assessments = []
            for id_ in ids:
                num = int(id_.replace("GSE", ""))
                if num < 20:
                    score = 0
                elif num < 35:
                    score = 3
                else:
                    score = 7
                assessments.append(
                    BiologyAssessment(ID=id_, BiologyScore=score, BiologyJustification="stub")
                )
            return BiologyAssessments(assessments=assessments)
        elif output_type is RelevanceAssessments:
            # Stage 2: deterministic biology=8, design=6 for all
            return RelevanceAssessments(assessments=[
                RelevanceAssessment(
                    ID=id_, BiologyScore=8, DesignScore=6,
                    BiologyJustification="b", DesignJustification="d",
                )
                for id_ in ids
            ])
        raise AssertionError(f"Unexpected output_type: {output_type}")

    monkeypatch.setattr(dsi, "call_structured", fake_call_structured)
    # Bypass enrichment (no real NCBI calls)
    monkeypatch.setattr(
        dsi, "enrich_representatives",
        lambda df: df.assign(MetadataSnapshot="(stub)", PubMedAbstract=None),
    )

    df = _make_mock_dataset_summary(50)

    # Run Stage 1
    stage1_df = asyncio.run(dsi.repeated_relevance(
        df, "test query", repeats=1, batch_size=25, openai_api_jobs=2,
        stage="stage1", stage1_biology_threshold=3,
    ))
    # All 50 datasets get a Stage 1 score
    assert len(stage1_df) == 50
    # The threshold gate should select 30 (15 with score=3, 15 with score=7)
    promoted = stage1_df[stage1_df["BiologyScore"] >= 3]
    assert len(promoted) == 30

    # Run Stage 2 on the promoted subset (after enriching it)
    enriched = dsi.enrich_representatives(promoted)
    stage2_df = asyncio.run(dsi.repeated_relevance(
        enriched, "test query", repeats=3, batch_size=25, openai_api_jobs=2,
        stage="stage2", stage1_biology_threshold=3,
    ))
    assert len(stage2_df) == 30
    # All Stage 2 rows should have both axes
    assert "BiologyScore" in stage2_df.columns
    assert "DesignScore" in stage2_df.columns
    assert "Run1BiologyScore" in stage2_df.columns
    assert "Run3DesignScore" in stage2_df.columns
    # All Stage 2 BiologyScore should be 8.0 (deterministic mock)
    assert all(stage2_df["BiologyScore"] == 8.0)
    assert all(stage2_df["DesignScore"] == 6.0)
