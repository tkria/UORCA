"""Unit tests for Project data model and ProjectManager."""

import pytest
import yaml
from datetime import datetime
from pathlib import Path


def test_slurm_config_defaults():
    from uorca.gui.project.models import SlurmConfig
    cfg = SlurmConfig()
    assert cfg.partition == ""
    assert cfg.cpus_per_task == 12
    assert cfg.memory == "16G"
    assert cfg.time_limit == "6:00:00"
    assert cfg.max_parallel == 6


def test_hpc_config_defaults():
    from uorca.gui.project.models import HpcConfig
    cfg = HpcConfig()
    assert cfg.host_alias == ""
    assert cfg.slurm.cpus_per_task == 12


def test_run_entry_creation():
    from uorca.gui.project.models import RunEntry
    run = RunEntry(
        id="identify_2026-04-16",
        type="identification",
        status="completed",
        research_question="Test query",
        output_path="/tmp/test",
    )
    assert run.type == "identification"
    assert run.timestamp is not None


def test_project_creation():
    from uorca.gui.project.models import Project
    project = Project(name="Test Project")
    assert project.name == "Test Project"
    assert project.runs == []
    assert project.created is not None


def test_project_add_run():
    from uorca.gui.project.models import Project, RunEntry
    project = Project(name="Test")
    run = RunEntry(id="test_run", type="identification", status="completed")
    project.runs.append(run)
    assert len(project.runs) == 1


def test_project_get_runs_by_type():
    from uorca.gui.project.models import Project, RunEntry
    project = Project(name="Test")
    project.runs.append(RunEntry(id="id1", type="identification", status="completed"))
    project.runs.append(RunEntry(id="pipe1", type="pipeline", status="completed"))
    project.runs.append(RunEntry(id="id2", type="identification", status="completed"))
    ident_runs = project.get_runs_by_type("identification")
    assert len(ident_runs) == 2


def test_project_to_yaml_and_back(tmp_path):
    from uorca.gui.project.models import Project, RunEntry
    project = Project(
        name="Test Project",
        description="A test",
        default_research_question="What genes?",
    )
    project.hpc.remote_working_dir = "/scratch/user"
    project.runs.append(RunEntry(
        id="id1", type="identification", status="completed",
        research_question="What genes?",
        output_path="/tmp/results",
        summary={"total": 100, "valid": 50},
        params={"max_results": 10},
        input_from="upload",
        target="human",
        datasets=["GSE123", "GSE456"],
        note="test run",
    ))

    yaml_path = tmp_path / "project.yaml"
    project.save(yaml_path)

    loaded = Project.load(yaml_path)
    assert loaded.name == "Test Project"
    assert loaded.description == "A test"
    assert loaded.created is not None
    assert loaded.updated is not None
    assert len(loaded.runs) == 1
    run = loaded.runs[0]
    assert run.summary["total"] == 100
    assert run.params == {"max_results": 10}
    assert run.input_from == "upload"
    assert run.target == "human"
    assert run.datasets == ["GSE123", "GSE456"]
    assert run.note == "test run"
    assert loaded.hpc.remote_working_dir == "/scratch/user"


def test_manager_create_project(tmp_path):
    from uorca.gui.project.manager import ProjectManager
    mgr = ProjectManager(projects_dir=tmp_path)
    project = mgr.create_project("Test Project", description="A test")
    assert project.name == "Test Project"
    assert (tmp_path / "test-project" / "project.yaml").exists()


def test_manager_list_projects(tmp_path):
    from uorca.gui.project.manager import ProjectManager
    mgr = ProjectManager(projects_dir=tmp_path)
    mgr.create_project("Project A")
    mgr.create_project("Project B")
    projects = mgr.list_projects()
    assert len(projects) == 2
    names = {p.name for p in projects}
    assert names == {"Project A", "Project B"}


def test_manager_load_project(tmp_path):
    from uorca.gui.project.manager import ProjectManager
    mgr = ProjectManager(projects_dir=tmp_path)
    mgr.create_project("My Project", description="test")
    loaded = mgr.load_project("my-project")
    assert loaded.name == "My Project"
    assert loaded.description == "test"


def test_manager_save_project(tmp_path):
    from uorca.gui.project.manager import ProjectManager
    mgr = ProjectManager(projects_dir=tmp_path)
    project = mgr.create_project("Save Test")
    project.description = "Updated"
    mgr.save_project("save-test", project)
    reloaded = mgr.load_project("save-test")
    assert reloaded.description == "Updated"


def test_manager_load_user_defaults(tmp_path):
    from uorca.gui.project.manager import ProjectManager

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "hpc": {
            "host_alias": "example_host",
            "container_image": "./uorca_dev.sif",
            "resource_dir": "./data/kallisto_indices",
            "slurm": {"partition": "example_partition", "memory": "32G"},
        }
    }))

    mgr = ProjectManager(projects_dir=tmp_path / "projects", config_path=config_path)
    defaults = mgr.get_user_defaults()
    assert defaults.host_alias == "example_host"
    assert defaults.slurm.partition == "example_partition"
    assert defaults.slurm.memory == "32G"


def test_manager_create_project_with_defaults(tmp_path):
    from uorca.gui.project.manager import ProjectManager

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({
        "hpc": {
            "host_alias": "example_host",
            "container_image": "./uorca_dev.sif",
            "slurm": {"partition": "example_partition"},
        }
    }))

    mgr = ProjectManager(projects_dir=tmp_path / "projects", config_path=config_path)
    project = mgr.create_project("New Project")
    assert project.hpc.host_alias == "example_host"
    assert project.hpc.slurm.partition == "example_partition"


def test_manager_get_project_slug(tmp_path):
    from uorca.gui.project.manager import ProjectManager
    mgr = ProjectManager(projects_dir=tmp_path)
    assert mgr.get_project_slug("My Cool Project") == "my-cool-project"
    assert mgr.get_project_slug("RNA-seq Analysis 2026") == "rna-seq-analysis-2026"
    assert mgr.get_project_slug("  Spaces & Symbols! ") == "spaces-symbols"
