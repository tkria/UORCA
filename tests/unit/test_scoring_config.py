"""Unit tests for the ScoringConfig dataclass."""

import pytest


def test_scoring_config_defaults():
    from uorca.gui.project.models import ScoringConfig

    cfg = ScoringConfig()
    assert cfg.biology_weight == 0.8
    assert cfg.design_min == 0
    assert cfg.stage1_biology_threshold == 3


def test_scoring_config_round_trip():
    from uorca.gui.project.models import ScoringConfig

    original = ScoringConfig(
        biology_weight=0.6,
        design_min=3,
        stage1_biology_threshold=4,
    )
    data = original.to_dict()
    assert data == {
        "biology_weight": 0.6,
        "design_min": 3,
        "stage1_biology_threshold": 4,
    }
    reconstructed = ScoringConfig.from_dict(data)
    assert reconstructed == original


def test_scoring_config_from_dict_missing_keys():
    """Old YAML data without scoring_config keys should produce defaults."""
    from uorca.gui.project.models import ScoringConfig

    cfg = ScoringConfig.from_dict({})
    assert cfg.biology_weight == 0.8
    assert cfg.design_min == 0
    assert cfg.stage1_biology_threshold == 3


def test_scoring_config_from_dict_partial():
    from uorca.gui.project.models import ScoringConfig

    cfg = ScoringConfig.from_dict({"biology_weight": 0.5})
    assert cfg.biology_weight == 0.5
    assert cfg.design_min == 0  # default preserved
    assert cfg.stage1_biology_threshold == 3


def test_project_round_trips_scoring_config(tmp_path):
    from uorca.gui.project.models import Project, ScoringConfig

    project = Project(name="ScoringTest", description="", default_research_question="q")
    project.scoring_config = ScoringConfig(
        biology_weight=0.5,
        design_min=4,
        stage1_biology_threshold=2,
    )

    p = tmp_path / "scoring.yaml"
    project.save(p)

    reloaded = Project.load(p)
    assert reloaded.scoring_config.biology_weight == 0.5
    assert reloaded.scoring_config.design_min == 4
    assert reloaded.scoring_config.stage1_biology_threshold == 2


def test_old_project_yaml_loads_with_default_scoring_config(tmp_path):
    """Project YAMLs written before ScoringConfig existed should load
    with default values."""
    import yaml

    OLD_YAML = """\
name: Legacy
description: ""
default_research_question: q
hpc:
  host_alias: ""
  container_image: ""
  resource_dir: ""
  remote_working_dir: ""
  slurm:
    partition: ""
    constraint: ""
    cpus_per_task: 12
    memory: 16G
    time_limit: 6:00:00
    max_parallel: 6
runs: []
custom_datasets: []
created: 2026-01-01T12:00:00
updated: 2026-01-01T12:00:00
"""
    from uorca.gui.project.models import Project

    p = tmp_path / "legacy.yaml"
    p.write_text(OLD_YAML)

    project = Project.load(p)
    assert project.scoring_config.biology_weight == 0.8
    assert project.scoring_config.design_min == 0
    assert project.scoring_config.stage1_biology_threshold == 3
