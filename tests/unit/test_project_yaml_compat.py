"""Tests that adding custom_datasets / custom_dataset_ids preserves backwards
compatibility with project YAMLs written by older UORCA versions."""

import yaml


OLD_PROJECT_YAML = """\
name: Legacy project
description: Pre-custom-dataset project
default_research_question: heart failure DE
hpc:
  host_alias: example_host
  container_image: ""
  resource_dir: /data/kallisto
  remote_working_dir: /data/uorca
  slurm:
    partition: example_partition
    cpus_per_task: 12
    memory: 16G
    time_limit: 6:00:00
    max_parallel: 6
runs:
  - id: identify_20260101_120000
    type: identification
    status: completed
    research_question: heart failure DE
    output_path: scratch/foo
    summary: {}
    timestamp: 2026-01-01T12:00:00
    params: {}
    datasets: [GSE12345, GSE23456]
created: 2026-01-01T12:00:00
updated: 2026-01-01T12:00:00
"""


def test_old_yaml_loads_with_empty_custom_datasets(tmp_path):
    from uorca.gui.project.models import Project

    p = tmp_path / "old.yaml"
    p.write_text(OLD_PROJECT_YAML)

    project = Project.load(p)
    assert project.name == "Legacy project"
    assert project.custom_datasets == []
    assert len(project.runs) == 1
    assert project.runs[0].custom_dataset_ids == []


def test_round_trip_preserves_data(tmp_path):
    from uorca.gui.project.models import (
        CustomDataset,
        Project,
        RunEntry,
    )

    project = Project(name="Round trip", description="", default_research_question="q")
    project.custom_datasets.append(
        CustomDataset(
            id="my-data",
            label="My data",
            fastq_dir="/data/my",
            metadata_filename="my-data.csv",
            organism="Homo sapiens",
            validation_status="valid",
        )
    )
    project.runs.append(
        RunEntry(
            id="pipeline_x",
            type="pipeline",
            status="running",
            output_path="/tmp",
            datasets=["GSE111"],
            custom_dataset_ids=["my-data"],
        )
    )

    p = tmp_path / "rt.yaml"
    project.save(p)
    reloaded = Project.load(p)

    assert len(reloaded.custom_datasets) == 1
    assert reloaded.custom_datasets[0].id == "my-data"
    assert reloaded.custom_datasets[0].validation_status == "valid"
    assert reloaded.runs[0].custom_dataset_ids == ["my-data"]


def test_no_custom_datasets_serialized_when_empty(tmp_path):
    """An empty list still serializes (as []), but the YAML stays compact."""
    from uorca.gui.project.models import Project

    project = Project(name="Empty", description="", default_research_question="q")
    p = tmp_path / "empty.yaml"
    project.save(p)

    raw = yaml.safe_load(p.read_text())
    assert raw["custom_datasets"] == []


def test_null_custom_datasets_in_yaml(tmp_path):
    """YAML with `custom_datasets: null` (or run with `custom_dataset_ids: null`)
    should load cleanly as empty lists, not TypeError."""
    from uorca.gui.project.models import Project

    yaml_content = """\
name: Null fields
description: ""
default_research_question: q
hpc:
  host_alias: ""
  container_image: ""
  resource_dir: ""
  remote_working_dir: ""
  slurm:
    partition: ""
    cpus_per_task: 12
    memory: 16G
    time_limit: 6:00:00
    max_parallel: 6
runs:
  - id: r1
    type: pipeline
    status: running
    research_question: q
    output_path: /tmp
    summary: {}
    timestamp: 2026-01-01T12:00:00
    params: {}
    datasets: []
    custom_dataset_ids: null
custom_datasets: null
created: 2026-01-01T12:00:00
updated: 2026-01-01T12:00:00
"""
    p = tmp_path / "null.yaml"
    p.write_text(yaml_content)

    project = Project.load(p)
    assert project.custom_datasets == []
    assert project.runs[0].custom_dataset_ids == []
