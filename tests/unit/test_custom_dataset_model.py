"""Unit tests for the CustomDataset dataclass."""

from datetime import datetime


def test_custom_dataset_defaults():
    from uorca.gui.project.models import CustomDataset

    cd = CustomDataset()
    assert cd.id == ""
    assert cd.label == ""
    assert cd.fastq_dir == ""
    assert cd.metadata_filename == ""
    assert cd.organism is None
    assert cd.description is None
    assert cd.validation_status == "unvalidated"
    assert cd.validation_message is None
    assert cd.last_validated is None
    # `created` should be auto-populated to an ISO timestamp
    assert cd.created is not None
    datetime.fromisoformat(cd.created)


def test_custom_dataset_round_trip():
    from uorca.gui.project.models import CustomDataset

    original = CustomDataset(
        id="gata4-cohort-n-16",
        label="GATA4 cohort (n=16)",
        fastq_dir="/data/gata4/fastqs",
        metadata_filename="gata4-cohort-n-16.csv",
        organism="Homo sapiens",
        description="GATA4 KO vs WT, iPSC-derived cardiomyocytes",
        validation_status="valid",
        last_validated="2026-05-01T10:00:00",
    )

    data = original.to_dict()
    assert data["id"] == "gata4-cohort-n-16"
    assert data["validation_status"] == "valid"

    reconstructed = CustomDataset.from_dict(data)
    assert reconstructed == original


def test_custom_dataset_from_dict_missing_keys():
    """Old YAML data without new fields should load with defaults."""
    from uorca.gui.project.models import CustomDataset

    cd = CustomDataset.from_dict({"id": "x", "label": "X", "fastq_dir": "/x"})
    assert cd.id == "x"
    assert cd.organism is None
    assert cd.validation_status == "unvalidated"
    # Missing `created` key should fall back to the factory-generated timestamp,
    # not silently drop to None.
    assert cd.created is not None
