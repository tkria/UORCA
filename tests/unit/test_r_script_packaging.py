"""Tests for R script packaging path resolution.

Regression tests for a defect introduced by the flat -> uorca/ package
restructure: _repo_paths_for_r() resolved paths one directory too shallow,
so RNAseq.R was never found and was silently omitted from download packages
while the generated README still instructed the user to run it.
"""
import pytest
from pathlib import Path

from uorca.gui.components.helpers import (
    _repo_paths_for_r,
    _patch_rnaseq_template,
)


@pytest.mark.unit
def test_rnaseq_template_path_resolves_to_real_file():
    """The bundled RNAseq.R must actually exist at the resolved path."""
    rnaseq_r = _repo_paths_for_r()["rnaseq_r"]
    assert rnaseq_r.name == "RNAseq.R"
    assert rnaseq_r.exists(), (
        f"RNAseq.R not found at resolved path {rnaseq_r}. "
        "The R script ships with the package; a miss here means the path "
        "arithmetic in _repo_paths_for_r() is wrong again."
    )


@pytest.mark.unit
def test_project_root_is_the_directory_containing_pyproject():
    """project_root must be the repo root, not the uorca/ package directory.

    Asserted structurally (pyproject.toml) rather than by checking for data/,
    so the test does not depend on optional downloaded reference data.
    """
    project_root = _repo_paths_for_r()["project_root"]
    assert (project_root / "pyproject.toml").exists(), (
        f"project_root resolved to {project_root}, which contains no "
        "pyproject.toml. The kallisto t2g.txt lookup depends on this being "
        "the repository root."
    )
    assert project_root.name != "uorca", (
        "project_root points at the uorca/ package, not the repo root"
    )


@pytest.mark.unit
def test_patch_rnaseq_template_returns_a_patched_script():
    """The patcher must return real script text, not None."""
    patched = _patch_rnaseq_template(group_col="Group", use_contrasts=False)
    assert isinstance(patched, str)
    assert patched.strip(), "patched script is empty"
    assert 'merged_group <- "Group"' in patched


@pytest.mark.unit
def test_patch_rnaseq_template_raises_when_template_missing(monkeypatch):
    """A missing template must fail loudly, never return None silently.

    Returning None caused the caller to skip writing RNAseq.R while the
    README still promised it - a silent, user-visible data loss.
    """
    import uorca.gui.components.helpers as helpers

    def fake_paths():
        real = _repo_paths_for_r()
        return {**real, "rnaseq_r": Path("/nonexistent/RNAseq.R")}

    monkeypatch.setattr(helpers, "_repo_paths_for_r", fake_paths)

    with pytest.raises(FileNotFoundError, match="RNAseq.R"):
        helpers._patch_rnaseq_template(group_col="Group", use_contrasts=False)
