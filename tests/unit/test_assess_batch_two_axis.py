"""Tests for the two-stage assess_subbatch."""

import asyncio

import pandas as pd
import pytest


@pytest.fixture
def stub_df():
    return pd.DataFrame([
        {"ID": "GSE001", "Title": "x", "Species": "Homo sapiens",
         "NumSamples": 6, "Summary": "test"},
        {"ID": "GSE002", "Title": "y", "Species": "Homo sapiens",
         "NumSamples": 6, "Summary": "test"},
    ])


def test_assess_subbatch_stage1_returns_biology_only(monkeypatch, stub_df):
    """Stage 1 dispatch should call call_structured with BiologyAssessments."""
    from uorca.identification import dataset_identification as dsi
    from uorca.identification.scoring import (
        BiologyAssessment,
        BiologyAssessments,
    )

    captured = {}

    def fake_call_structured(prompt, user_input, output_type):
        captured["output_type"] = output_type
        return BiologyAssessments(
            assessments=[
                BiologyAssessment(ID="GSE001", BiologyScore=7, BiologyJustification="ok"),
                BiologyAssessment(ID="GSE002", BiologyScore=2, BiologyJustification="weak"),
            ]
        )

    monkeypatch.setattr(dsi, "call_structured", fake_call_structured)
    # Return a stub prompt for any prompt-load call.
    monkeypatch.setattr(dsi, "load_prompt", lambda p: "{biology_examples}\n{design_examples}\n{threshold}")

    sem = asyncio.Semaphore(1)
    result = asyncio.run(dsi.assess_subbatch(
        df=stub_df, query="q", schema=None, key="assessments",
        rep=0, idx=0, total_batches=1, sem=sem,
        stage="stage1",
        stage1_biology_threshold=3,
    ))
    assert captured["output_type"] is BiologyAssessments
    assert len(result) == 2
    assert result[0].BiologyScore == 7
    assert result[1].BiologyScore == 2


def test_assess_subbatch_stage2_returns_two_axes(monkeypatch, stub_df):
    """Stage 2 dispatch should call call_structured with RelevanceAssessments."""
    from uorca.identification import dataset_identification as dsi
    from uorca.identification.scoring import (
        RelevanceAssessment,
        RelevanceAssessments,
    )

    captured = {}

    def fake_call_structured(prompt, user_input, output_type):
        captured["output_type"] = output_type
        return RelevanceAssessments(
            assessments=[
                RelevanceAssessment(
                    ID="GSE001", BiologyScore=8, DesignScore=6,
                    BiologyJustification="b1", DesignJustification="d1",
                ),
                RelevanceAssessment(
                    ID="GSE002", BiologyScore=3, DesignScore=2,
                    BiologyJustification="b2", DesignJustification="d2",
                ),
            ]
        )

    monkeypatch.setattr(dsi, "call_structured", fake_call_structured)
    monkeypatch.setattr(dsi, "load_prompt", lambda p: "{biology_examples}\n{design_examples}\n{threshold}")

    enriched = stub_df.copy()
    enriched["MetadataSnapshot"] = ["meta1", "meta2"]
    enriched["PubMedAbstract"] = [None, None]

    sem = asyncio.Semaphore(1)
    result = asyncio.run(dsi.assess_subbatch(
        df=enriched, query="q", schema=None, key="assessments",
        rep=0, idx=0, total_batches=1, sem=sem,
        stage="stage2",
        stage1_biology_threshold=3,
    ))
    assert captured["output_type"] is RelevanceAssessments
    assert len(result) == 2
    assert result[0].BiologyScore == 8
    assert result[0].DesignScore == 6


def test_assess_subbatch_fallback_on_exception(monkeypatch, stub_df):
    """When call_structured raises, return default fallback assessments
    (not silently drop the batch)."""
    from uorca.identification import dataset_identification as dsi

    def fake_call_structured(prompt, user_input, output_type):
        raise RuntimeError("simulated LLM failure")

    monkeypatch.setattr(dsi, "call_structured", fake_call_structured)
    monkeypatch.setattr(dsi, "load_prompt", lambda p: "stub")

    sem = asyncio.Semaphore(1)
    result = asyncio.run(dsi.assess_subbatch(
        df=stub_df, query="q", schema=None, key="assessments",
        rep=0, idx=0, total_batches=1, sem=sem,
        stage="stage1",
        stage1_biology_threshold=3,
    ))
    # Fallback returns one default assessment per row in df.
    assert len(result) == 2
    for assessment in result:
        assert assessment.BiologyScore == 5  # default fallback
        assert "fail" in assessment.BiologyJustification.lower()
