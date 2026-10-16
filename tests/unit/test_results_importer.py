"""Tests for importing results produced outside the GUI into a project."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
import yaml

from tests._results_fixtures import (
    FAKE_QUESTION,
    make_analysis_folder,
    make_arm_folder,
    make_dataset,
)
from uorca.gui.project.importer import (
    DuplicateImportError,
    ImportResultsError,
    default_project_name,
    ensure_imported,
    import_into_named_project,
    import_results,
    inspect_results,
)
from uorca.gui.project.manager import ProjectManager
from uorca.gui.project.models import Project

pytestmark = pytest.mark.unit


@pytest.fixture
def mgr(tmp_path):
    return ProjectManager(projects_dir=tmp_path / "home" / "projects",
                          config_path=tmp_path / "home" / "config.yaml")


# --- inspect_results: layouts ---------------------------------------------------------


def test_analysis_folder_counts_complete_partial_failed(tmp_path):
    root = make_analysis_folder(tmp_path / "res", failed=("GSE900004",), status_only=("GSE900005",))
    c = inspect_results(root)
    assert c.layout == "analysis_folder"
    assert c.analysis_dir == root.resolve()
    assert c.datasets == ["GSE900001", "GSE900002", "GSE900003", "GSE900004", "GSE900005"]
    assert c.complete == ["GSE900001", "GSE900002"]
    assert c.partial == ["GSE900003"]
    assert c.failed == ["GSE900004", "GSE900005"]
    assert c.research_question == FAKE_QUESTION
    assert c.identify_dir is None
    assert c.importable


def test_partial_dataset_with_deg_tables_but_failed_r_step_is_not_complete(tmp_path):
    # Modelled on a real case: three DEG tables written, then the R step failed.
    root = make_analysis_folder(tmp_path / "res", complete=(), partial=("GSE900003",))
    c = inspect_results(root)
    assert c.complete == []
    assert c.partial == ["GSE900003"]
    assert not c.importable


def test_success_without_cpm_is_not_complete_and_warns(tmp_path):
    root = tmp_path / "res"
    make_dataset(root, "GSE900001")
    make_dataset(root, "GSE900002", cpm=False)
    c = inspect_results(root)
    assert c.complete == ["GSE900001"]
    assert c.partial == ["GSE900002"]
    assert any("CPM.csv is missing" in w for w in c.warnings)


def test_unreadable_analysis_info_is_reported(tmp_path):
    root = tmp_path / "res"
    make_dataset(root, "GSE900001")
    ds = make_dataset(root, "GSE900002")
    (ds / "metadata" / "analysis_info.json").write_text("{not json")
    c = inspect_results(root)
    assert c.complete == ["GSE900001"]
    assert any("GSE900002" in w and "unreadable" in w for w in c.warnings)


def test_arm_folder_finds_analysis_and_identify(tmp_path):
    arm = make_arm_folder(tmp_path / "arm_openai")
    c = inspect_results(arm)
    assert c.layout == "arm_folder"
    assert c.analysis_dir == (arm / "analysis").resolve()
    assert c.identify_dir == (arm / "identify").resolve()
    assert c.complete == ["GSE900001", "GSE900002"]
    assert default_project_name(c) == "arm_openai"


def test_analysis_subfolder_of_arm_finds_sibling_identify(tmp_path):
    arm = make_arm_folder(tmp_path / "arm_x")
    c = inspect_results(arm / "analysis")
    assert c.layout == "analysis_folder"
    assert c.identify_dir == (arm / "identify").resolve()
    assert default_project_name(c) == "arm_x"


def test_arm_folder_without_identify(tmp_path):
    arm = make_arm_folder(tmp_path / "arm_y", identify=False)
    c = inspect_results(arm)
    assert c.layout == "arm_folder"
    assert c.identify_dir is None


def test_single_dataset_folder(tmp_path):
    ds = make_dataset(tmp_path, "GSE900009")
    c = inspect_results(ds)
    assert c.layout == "dataset_folder"
    assert c.analysis_dir == ds.resolve()
    assert c.datasets == ["GSE900009"]
    assert c.complete == ["GSE900009"]


def test_research_question_absent(tmp_path):
    root = make_analysis_folder(tmp_path / "res", research_question=None)
    c = inspect_results(root)
    assert c.research_question is None
    assert c.warnings == []


def test_research_question_without_key_warns(tmp_path):
    root = make_analysis_folder(tmp_path / "res", research_question=None)
    (root / "research_question.json").write_text(json.dumps({"other": 1}))
    c = inspect_results(root)
    assert c.research_question is None
    assert any("research_question" in w for w in c.warnings)


# --- inspect_results: errors ----------------------------------------------------------


def test_missing_path_raises(tmp_path):
    with pytest.raises(ImportResultsError, match="does not exist"):
        inspect_results(tmp_path / "nope")


def test_file_path_raises(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(ImportResultsError, match="not a folder"):
        inspect_results(f)


def test_empty_path_raises():
    with pytest.raises(ImportResultsError, match="No path"):
        inspect_results("  ")


def test_folder_without_analysis_raises(tmp_path):
    (tmp_path / "empty" / "something").mkdir(parents=True)
    with pytest.raises(ImportResultsError, match="No UORCA analysis found"):
        inspect_results(tmp_path / "empty")


# --- import_results -------------------------------------------------------------------


def test_import_appends_completed_pipeline_run(tmp_path):
    root = make_analysis_folder(tmp_path / "res")
    project = Project(name="P")
    when = datetime(2026, 10, 6, 12, 0, 0)
    entries = import_results(project, inspect_results(root), note="from HPC", now=when)
    assert len(entries) == 1
    run = entries[0]
    assert project.runs == [run]
    assert run.type == "pipeline"
    assert run.status == "completed"
    assert run.output_path == str(root.resolve())
    assert run.datasets == ["GSE900001", "GSE900002", "GSE900003"]
    assert run.research_question == FAKE_QUESTION
    assert run.summary == {"submitted": 3, "succeeded": 2, "failed": 0, "partial": 1}
    assert run.params["imported"] is True
    assert run.params["source"] == str(root.resolve())
    assert run.params["imported_at"] == "2026-10-06T12:00:00"
    assert run.note == "from HPC"
    assert run.input_from is None
    assert run.id == "pipeline_20261006_120000_imported_res"
    assert project.updated == when.isoformat()


def test_import_links_identification_run(tmp_path):
    arm = make_arm_folder(tmp_path / "arm_openai")
    project = Project(name="P")
    entries = import_results(project, inspect_results(arm))
    assert [e.type for e in entries] == ["identification", "pipeline"]
    ident, pipe = entries
    assert ident.status == "completed"
    assert ident.output_path == str((arm / "identify").resolve())
    assert pipe.input_from == ident.id
    assert pipe.output_path == str((arm / "analysis").resolve())


def test_duplicate_import_refused(tmp_path):
    root = make_analysis_folder(tmp_path / "res")
    project = Project(name="P")
    import_results(project, inspect_results(root))
    with pytest.raises(DuplicateImportError, match="already in project 'P'"):
        import_results(project, inspect_results(root))
    assert len(project.runs) == 1


def test_duplicate_detected_through_arm_and_analysis_paths(tmp_path):
    arm = make_arm_folder(tmp_path / "arm")
    project = Project(name="P")
    import_results(project, inspect_results(arm))
    with pytest.raises(DuplicateImportError):
        import_results(project, inspect_results(arm / "analysis"))


def test_nothing_complete_refused(tmp_path):
    root = make_analysis_folder(tmp_path / "res", complete=(), partial=("GSE900003",))
    project = Project(name="P")
    with pytest.raises(ImportResultsError, match="No complete dataset"):
        import_results(project, inspect_results(root))
    assert project.runs == []


def test_imported_project_round_trips_through_yaml(tmp_path):
    arm = make_arm_folder(tmp_path / "arm")
    project = Project(name="P")
    import_results(project, inspect_results(arm), note="n")
    path = tmp_path / "p.yaml"
    project.save(path)
    loaded = Project.load(path)
    assert [r.to_dict() for r in loaded.runs] == [r.to_dict() for r in project.runs]
    completed = [r for r in loaded.get_runs_by_type("pipeline") if r.status == "completed" and r.output_path]
    assert len(completed) == 1


def test_old_yaml_without_import_params_still_loads(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump({
        "name": "Old",
        "runs": [{"id": "pipeline_1", "type": "pipeline", "status": "completed", "output_path": "results/x"}],
    }))
    loaded = Project.load(path)
    assert loaded.runs[0].params == {}
    assert loaded.runs[0].params.get("imported") is None


# --- manager helpers ------------------------------------------------------------------


def test_project_exists(mgr):
    assert not mgr.project_exists("My Project")
    mgr.create_project("My Project")
    assert mgr.project_exists("My Project")
    assert not mgr.project_exists("!!!")


def test_import_into_named_project_requires_create(tmp_path, mgr):
    c = inspect_results(make_analysis_folder(tmp_path / "res"))
    with pytest.raises(ImportResultsError, match="does not exist"):
        import_into_named_project(mgr, c, "New")
    assert not mgr.project_exists("New")
    project, entries = import_into_named_project(mgr, c, "New", create=True)
    assert mgr.load_project("new").runs[0].id == entries[0].id


def test_import_into_existing_project_keeps_runs(tmp_path, mgr):
    p = mgr.create_project("Existing", description="keep me")
    c = inspect_results(make_analysis_folder(tmp_path / "res"))
    import_into_named_project(mgr, c, "Existing")
    loaded = mgr.load_project("existing")
    assert loaded.description == "keep me"
    assert len(loaded.runs) == 1
    assert p.name == loaded.name


def test_import_into_named_project_slug_clash(tmp_path, mgr):
    mgr.create_project("Foo Bar")
    c = inspect_results(make_analysis_folder(tmp_path / "res"))
    with pytest.raises(ImportResultsError, match="already uses the file name"):
        import_into_named_project(mgr, c, "foo-bar", create=True)


def test_ensure_imported_is_idempotent(tmp_path, mgr):
    root = make_analysis_folder(tmp_path / "hpc_arm")
    project, entries = ensure_imported(mgr, root)
    assert project.name == "hpc_arm"
    assert entries is not None and len(entries) == 1
    project2, entries2 = ensure_imported(mgr, root)
    assert entries2 is None
    assert len(mgr.load_project("hpc-arm").runs) == 1
    assert len(mgr.list_projects()) == 1


# --- completeness matches the Explorer (unique_groups) -------------------------------


def test_null_unique_groups_is_not_complete(tmp_path):
    root = tmp_path / "res"
    make_dataset(root, "GSE900001", null_groups=True)
    make_dataset(root, "GSE900002")
    c = inspect_results(root)
    assert c.complete == ["GSE900002"]
    assert c.partial == ["GSE900001"]  # DEG tables exist, but the Explorer skips it
    assert any("unique_groups null" in w for w in c.warnings)


def test_all_null_unique_groups_is_not_importable(tmp_path, mgr):
    root = tmp_path / "res"
    make_dataset(root, "GSE900001", null_groups=True)
    c = inspect_results(root)
    assert not c.importable
    with pytest.raises(ImportResultsError, match="unique_groups"):
        import_into_named_project(mgr, c, "p", create=True)


def test_importer_complete_matches_results_integrator(tmp_path):
    from uorca.gui.results_integration import ResultsIntegrator

    root = tmp_path / "res"
    make_dataset(root, "GSE900001", null_groups=True)
    make_dataset(root, "GSE900002")
    make_dataset(root, "GSE900003", success=False, cpm=False)
    c = inspect_results(root)
    ri = ResultsIntegrator(results_dir=str(root))
    ri.load_data()
    assert sorted(ri.cpm_data) == c.complete == ["GSE900002"]


# --- ensure_imported never merges into an existing project ---------------------------


def test_ensure_imported_same_basename_gets_separate_projects(tmp_path, mgr):
    a = make_arm_folder(tmp_path / "q1" / "openai")
    b = make_arm_folder(tmp_path / "q2" / "openai")
    p1, e1 = ensure_imported(mgr, a)
    p2, e2 = ensure_imported(mgr, b)
    assert p1.name == "openai"
    assert p2.name == "q2/openai"
    assert e1 and e2
    assert len(mgr.load_project("openai").runs) == 2
    assert len(mgr.load_project("q2-openai").runs) == 2
    # Re-running either finds its own project and imports nothing.
    again, none = ensure_imported(mgr, a)
    assert none is None and again.name == "openai"
    again, none = ensure_imported(mgr, b)
    assert none is None and again.name == "q2/openai"
    assert len(mgr.list_projects()) == 2


def test_ensure_imported_does_not_touch_gui_project_with_same_name(tmp_path, mgr):
    mgr.create_project("results", description="my own GUI project")
    root = make_analysis_folder(tmp_path / "data" / "results")
    project, entries = ensure_imported(mgr, root)
    assert project.name == "data/results"
    assert entries is not None
    gui = mgr.load_project("results")
    assert gui.runs == [] and gui.description == "my own GUI project"


def test_ensure_imported_numeric_suffix_when_both_names_taken(tmp_path, mgr):
    mgr.create_project("openai")
    mgr.create_project("q1/openai")
    project, _ = ensure_imported(mgr, make_arm_folder(tmp_path / "q1" / "openai"))
    assert project.name == "q1/openai (2)"
    assert mgr.load_project("openai").runs == []
    assert mgr.load_project("q1-openai").runs == []


def test_ensure_imported_reuses_any_project_recording_the_path(tmp_path, mgr):
    root = make_analysis_folder(tmp_path / "res")
    import_into_named_project(mgr, inspect_results(root), "Hand made", create=True)
    project, entries = ensure_imported(mgr, root)
    assert entries is None and project.name == "Hand made"
    assert len(mgr.list_projects()) == 1
