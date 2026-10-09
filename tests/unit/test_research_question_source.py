"""The explorer must show the research question of the results it is showing.

Regression: identification wrote the last query to ``uorca/config/dataset_query.json``
inside the package, and the explorer fell back to that file. Exploring unrelated
results then showed the question of whatever identification ran last.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit


def test_identification_no_longer_writes_a_global_query_file() -> None:
    from uorca.identification import dataset_identification

    assert not hasattr(dataset_identification, "save_query_config")
    assert "\"dataset_query.json\"" not in inspect.getsource(dataset_identification)


@pytest.mark.parametrize(
    ("module", "func"),
    [
        ("uorca.gui.components.uorca_summary_tab", "_load_research_query"),
        ("uorca.gui.components.ai_assistant_tab", "load_query_config"),
    ],
)
def test_explorer_reads_only_the_results_directory(
    module: str, func: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    mod = importlib.import_module(module)
    loader = getattr(mod, func)
    assert "\"dataset_query.json\"" not in inspect.getsource(loader)

    assert loader(str(tmp_path)) is None

    (tmp_path / "research_question.json").write_text(
        json.dumps({"research_question": "MYBPC3 cardiomyopathy"})
    )
    assert loader(str(tmp_path)) == "MYBPC3 cardiomyopathy"

    fake_st = MagicMock()
    monkeypatch.setattr(mod, "st", fake_st)
    (tmp_path / "research_question.json").write_text("{not json")
    assert loader(str(tmp_path)) is None
    fake_st.warning.assert_called_once()
