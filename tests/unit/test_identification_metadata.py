"""identification_metadata.json must record how a run found its datasets.

Search terms are LLM-generated and differ between runs on the same query, so
without them an output directory cannot explain why it retrieved the corpus it
did. The terms were previously only present in the run log, which lives outside
the output directory and is lost as soon as results are moved or archived.
"""

import datetime

import pytest

from uorca.identification.dataset_identification import build_identification_metadata


@pytest.mark.unit
def test_metadata_records_search_terms():
    """The generated search terms are persisted alongside the results."""
    metadata = build_identification_metadata(
        research_query="PPAR-g and immunotherapy",
        search_terms={"immunotherapy", "PPARG", "cancer"},
        start_time=datetime.datetime(2026, 8, 7, 14, 42, 15),
        end_time=datetime.datetime(2026, 8, 7, 15, 2, 48),
        total_datasets_assessed=476,
        datasets_deemed_relevant=9,
        threshold=7.0,
    )

    assert "search_terms" in metadata, "search terms are not recorded"
    assert set(metadata["search_terms"]) == {"immunotherapy", "PPARG", "cancer"}


@pytest.mark.unit
def test_metadata_search_terms_are_ordered():
    """Terms come from a set, so they are sorted for a stable, diffable record."""
    metadata = build_identification_metadata(
        research_query="q",
        search_terms={"zeta", "alpha", "mu"},
        start_time=datetime.datetime(2026, 8, 7, 14, 0, 0),
        end_time=datetime.datetime(2026, 8, 7, 14, 30, 0),
        total_datasets_assessed=1,
        datasets_deemed_relevant=0,
        threshold=7.0,
    )

    assert metadata["search_terms"] == ["alpha", "mu", "zeta"]


@pytest.mark.unit
def test_metadata_preserves_existing_fields():
    """Adding search terms must not disturb the fields consumers already read."""
    start = datetime.datetime(2026, 8, 7, 14, 42, 15)
    end = datetime.datetime(2026, 8, 7, 15, 2, 48)

    metadata = build_identification_metadata(
        research_query="PPAR-g and immunotherapy",
        search_terms={"immunotherapy"},
        start_time=start,
        end_time=end,
        total_datasets_assessed=476,
        datasets_deemed_relevant=9,
        threshold=7.0,
    )

    assert metadata["input_query"] == "PPAR-g and immunotherapy"
    assert metadata["start_time"] == start.isoformat()
    assert metadata["end_time"] == end.isoformat()
    assert metadata["total_datasets_assessed"] == 476
    assert metadata["datasets_deemed_relevant"] == 9
    assert metadata["threshold_used"] == 7.0


@pytest.mark.unit
def test_metadata_is_json_serialisable():
    """The result is written with json.dump, so it must contain no sets."""
    import json

    metadata = build_identification_metadata(
        research_query="q",
        search_terms={"a", "b"},
        start_time=datetime.datetime(2026, 8, 7, 14, 0, 0),
        end_time=datetime.datetime(2026, 8, 7, 14, 30, 0),
        total_datasets_assessed=1,
        datasets_deemed_relevant=1,
        threshold=7.0,
    )

    assert json.loads(json.dumps(metadata))["search_terms"] == ["a", "b"]
