"""Integration tests for the My Datasets component.

Driving Streamlit forms with file_uploader via AppTest is unreliable
across versions, so we exercise the component via direct rendering and
its helper logic via direct calls."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def project_with_hpc(tmp_path, monkeypatch):
    """Configure ProjectManager to use a temp dir and seed a project."""
    from uorca.gui.project.manager import ProjectManager
    from uorca.gui.project.models import HpcConfig, Project

    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    project = Project(
        name="Integration test",
        description="",
        default_research_question="q",
    )
    project.hpc = HpcConfig()
    project.hpc.host_alias = "example_host"
    project.hpc.remote_working_dir = "/data/uorca"
    project.hpc.resource_dir = "/data/kallisto"

    mgr = ProjectManager()
    slug = mgr.get_project_slug(project.name)
    mgr.save_project(slug, project)
    return slug, project


def test_render_empty_state(project_with_hpc, monkeypatch):
    """Smoke: rendering a project with no custom_datasets should not raise."""
    from streamlit.testing.v1 import AppTest

    slug, _ = project_with_hpc
    driver_path = Path(__file__).parent / "_my_datasets_driver.py"
    driver_path.write_text(
        "import streamlit as st\n"
        "from uorca.gui.project.manager import ProjectManager\n"
        "from uorca.gui.components.custom_datasets import render_my_datasets\n"
        "mgr = ProjectManager()\n"
        f"st.session_state['active_project'] = mgr.load_project({slug!r})\n"
        "render_my_datasets(st.session_state['active_project'])\n"
    )
    try:
        at = AppTest.from_file(str(driver_path), default_timeout=10).run()
        # No exceptions means the empty-state branch rendered cleanly.
        assert at.exception == [], f"Unexpected exceptions: {at.exception}"
        # Empty state shows an info message.
        assert any("No custom datasets" in i.value for i in at.info)
    finally:
        driver_path.unlink(missing_ok=True)


def test_handle_add_or_edit_submit_rejects_missing_csv(
    monkeypatch, project_with_hpc, tmp_path
):
    """Direct call into the submit handler — the no-CSV case must surface
    a loud error and NOT touch SSH."""
    import streamlit as st
    from uorca.gui.components import custom_datasets
    from uorca.gui.project.manager import ProjectManager

    slug, _ = project_with_hpc
    project = ProjectManager().load_project(slug)

    # If SSH is reached, fail loudly — the validation error path must short-circuit.
    monkeypatch.setattr(
        custom_datasets, "ssh_connect",
        lambda *a, **kw: pytest.fail("SSH must not be invoked when CSV is missing"),
    )

    captured_errors: list[str] = []
    monkeypatch.setattr(st, "error", lambda msg, **kw: captured_errors.append(str(msg)))
    monkeypatch.setattr(st, "rerun", lambda: None)

    custom_datasets._handle_add_or_edit_submit(
        project=project,
        editing=None,
        label="Test cohort",
        fastq_dir="/data/test/fastqs",
        uploaded=None,           # the trigger
        organism=None,
        description=None,
        password="pw",
    )

    assert any("Metadata CSV" in m for m in captured_errors), captured_errors


def test_handle_add_or_edit_submit_happy_path(
    monkeypatch, project_with_hpc, tmp_path
):
    """Successful registration appends a CustomDataset with status=valid."""
    import streamlit as st
    from uorca.gui.components import custom_datasets
    from uorca.gui.hpc import custom_dataset_validator
    from uorca.gui.project.manager import ProjectManager

    slug, _ = project_with_hpc
    project = ProjectManager().load_project(slug)

    fake_client = MagicMock()
    monkeypatch.setattr(custom_datasets, "ssh_connect", lambda *a, **kw: fake_client)
    monkeypatch.setattr(custom_datasets, "ssh_close", lambda c: None)
    monkeypatch.setattr(
        custom_datasets, "validate_fastq_dir",
        lambda client, path: ("valid", None),
    )
    monkeypatch.setattr(
        custom_datasets, "stage_metadata_csv",
        lambda client, csv_io, remote_path: None,
    )

    monkeypatch.setattr(st, "error", lambda *a, **kw: pytest.fail(f"unexpected error: {a}"))
    monkeypatch.setattr(st, "success", lambda *a, **kw: None)
    monkeypatch.setattr(st, "rerun", lambda: None)
    # session_state writes happen — a plain dict-like substitute is enough.
    monkeypatch.setattr(st, "session_state", {})

    fake_uploaded = MagicMock()
    fake_uploaded.getvalue = lambda: b"sample_id\nS1\n"

    custom_datasets._handle_add_or_edit_submit(
        project=project,
        editing=None,
        label="Test cohort",
        fastq_dir="/data/test/fastqs",
        uploaded=fake_uploaded,
        organism="Homo sapiens",
        description="desc",
        password="pw",
    )

    # Reload from disk — the project YAML should reflect the new entry.
    reloaded = ProjectManager().load_project(slug)
    assert len(reloaded.custom_datasets) == 1
    cd = reloaded.custom_datasets[0]
    assert cd.id == "test-cohort"
    assert cd.fastq_dir == "/data/test/fastqs"
    assert cd.organism == "Homo sapiens"
    assert cd.validation_status == "valid"
