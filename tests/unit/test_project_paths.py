"""Relative run paths resolve against one data root, not the app's working directory.

Projects live in ~/.uorca/projects and are shared by every checkout and worktree. A
run records ``scratch/<run>`` or ``results/<run>`` relative to the directory the app
was started from, so an app started from a worktree looked in the wrong place and
reported "No results file found".
"""

from pathlib import Path

import pytest

from uorca.gui.project.paths import DATA_ROOT_ENV, data_root, resolve_local_path


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    return tmp_path / "config.yaml"


@pytest.mark.unit
def test_absolute_path_is_unchanged(config, tmp_path):
    target = tmp_path / "elsewhere" / "run"
    assert resolve_local_path(str(target), config_path=config) == target


@pytest.mark.unit
def test_env_var_sets_the_root(config, tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert resolve_local_path("scratch/run", config_path=config) == tmp_path / "scratch" / "run"


@pytest.mark.unit
def test_config_data_root_sets_the_root(config, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    config.write_text(f"data_root: {root}\nhpc: {{}}\n")
    assert data_root(config_path=config) == root
    assert resolve_local_path("results/run", config_path=config) == root / "results" / "run"


@pytest.mark.unit
def test_env_var_takes_precedence_over_config(config, tmp_path, monkeypatch):
    (tmp_path / "from_config").mkdir()
    (tmp_path / "from_env").mkdir()
    config.write_text(f"data_root: {tmp_path / 'from_config'}\n")
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "from_env"))
    assert data_root(config_path=config) == tmp_path / "from_env"


@pytest.mark.unit
def test_falls_back_to_cwd_without_a_configured_root(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert data_root(config_path=config) == Path.cwd()


@pytest.mark.unit
def test_missing_configured_root_raises(config, tmp_path):
    config.write_text(f"data_root: {tmp_path / 'does_not_exist'}\n")
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        data_root(config_path=config)


@pytest.mark.unit
def test_importer_matches_relative_run_paths_under_the_root(tmp_path, monkeypatch):
    from uorca.gui.project.importer import _same_path

    analysis = tmp_path / "results" / "run"
    analysis.mkdir(parents=True)
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)  # the app's working directory is not the root
    assert _same_path("results/run", analysis)
