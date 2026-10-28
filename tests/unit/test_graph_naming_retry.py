"""Graph-level tests: names are fixed before Kallisto, and retries reuse Kallisto.

The LLM agents are replaced by scripted ``FunctionModel`` responses; no API is
called. Real metadata comes from ``tests/fixtures/contrast_design``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_graph import GraphRunContext

from uorca.core.design_validation import r_make_names
from uorca.core.naming import R_GROUP_COLUMN
from uorca.graph.agents import contrast_design_agent, reflection_agent
from uorca.graph.deps import WorkflowDeps
from uorca.graph.models import MetadataAnalysisOut, ReflectionOut
from uorca.graph.nodes.contrasts import ContrastDesignNode
from uorca.graph.nodes.edger import PrepareEdgeRNode
from uorca.graph.nodes.kallisto import KallistoQuantifyNode
from uorca.graph.nodes.metadata import MetadataAnalysisNode
from uorca.graph.nodes.reflect import ReflectNode
from uorca.graph.state import CheckpointStatus, WorkflowState

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "contrast_design"


def _metadata(gse: str) -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / f"{gse}_samples.csv", dtype=str)
    return df.drop(columns=["uorca_group"])  # the node must build its own


def _ctx(state: WorkflowState, tmp_path: Path) -> GraphRunContext[WorkflowState, WorkflowDeps]:
    deps = WorkflowDeps(output_dir=tmp_path, resource_dir=tmp_path, entrez_email="x@example.org")
    return GraphRunContext(state=state, deps=deps)


def _contrast_model(responses: list[list[dict[str, str]]]) -> tuple[FunctionModel, list[int]]:
    """A model that returns each scripted contrast list in turn (last one repeats)."""
    calls = [0]

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        contrasts = responses[min(calls[0], len(responses) - 1)]
        calls[0] += 1
        args: dict[str, Any] = {
            "contrasts": [
                {**c, "description": "d", "justification": "j"} for c in contrasts
            ],
            "analysis_summary": "s",
        }
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(fn), calls


async def _run_metadata(gse: str, column: str, tmp_path: Path) -> tuple[WorkflowState, Any]:
    state = WorkflowState(dataset_id=gse)
    state.metadata_df = _metadata(gse)
    levels = state.metadata_df[column].unique().tolist()
    out = MetadataAnalysisOut(
        selected_columns=[column],
        merged_column_name="cell type",
        unique_groups=levels,
        reasoning="r",
    )
    ctx = _ctx(state, tmp_path)
    with patch(
        "uorca.graph.nodes.metadata.metadata_analysis_agent.run",
        new=AsyncMock(return_value=type("R", (), {"output": out})()),
    ):
        nxt = await MetadataAnalysisNode().run(ctx)
    return state, nxt


async def _test_gse161060_metadata_node_keeps_both_cell_types(tmp_path):
    state, nxt = await _run_metadata("GSE161060", "characteristics_ch1.1.cell type", tmp_path)

    assert isinstance(nxt, ContrastDesignNode)
    assert state.merged_column == "cell_type"  # R-safe header, raw values
    names = state.group_level_names
    assert names is not None and len(set(names.values())) == 2
    df = state.metadata_df
    assert df is not None
    r_col = df[R_GROUP_COLUMN]
    assert r_col.nunique() == 2
    assert all(r_make_names(v) == v for v in r_col)
    # Raw values are untouched for the explorer.
    assert set(df["cell_type"]) == set(names)


async def _test_gse210935_contrast_node_retries_bad_design_before_kallisto(tmp_path):
    state, _ = await _run_metadata("GSE210935", "characteristics_ch1.3.treatment", tmp_path)
    names = state.group_level_names
    assert names is not None
    original = pd.read_csv(FIXTURES / "GSE210935_contrasts.csv").to_dict(orient="records")
    fixed = [
        {"name": "GMIL4_vs_GM_8h", "expression": f"{names['8h_GMIL4']} - {names['8h_GM']}"},
        {"name": "MoDC_GMT_vs_MoDC_GMIL4", "expression": "MoDC_GMT - MoDC_GMIL4"},
    ]
    model, calls = _contrast_model([original, fixed])

    with contrast_design_agent.override(model=model):
        nxt = await ContrastDesignNode().run(_ctx(state, tmp_path))

    assert isinstance(nxt, KallistoQuantifyNode)
    assert calls[0] == 2  # the original design was sent back once
    assert [c["name"] for c in state.contrasts or []] == ["GMIL4_vs_GM_8h", "MoDC_GMT_vs_MoDC_GMIL4"]
    record = json.loads((tmp_path / "metadata" / "naming.json").read_text())
    assert record["contrasts"][0]["weights"] == {"X8h_GMIL4": 1.0, "X8h_GM": -1.0}
    assert {lv["r_name"] for lv in record["levels"]} == set(names.values())
    saved = pd.read_csv(tmp_path / "metadata" / "contrasts.csv")
    assert saved["expression"].tolist()[0] == "X8h_GMIL4 - X8h_GM"


async def _test_persistently_bad_design_fails_early_with_clear_message(tmp_path):
    state, _ = await _run_metadata("GSE210935", "characteristics_ch1.3.treatment", tmp_path)
    original = pd.read_csv(FIXTURES / "GSE210935_contrasts.csv").to_dict(orient="records")
    model, _ = _contrast_model([original])

    with contrast_design_agent.override(model=model):
        nxt = await ContrastDesignNode().run(_ctx(state, tmp_path))

    assert isinstance(nxt, ReflectNode)
    assert nxt.failed_checkpoint == "contrast_design"
    last = state.analysis_history[-1]
    assert last["error_type"] == "naming"
    assert "'8h_GMIL4_vs_8h_GM' is not allowed" in last["error"]
    assert state.checkpoints["kallisto_quant"] == CheckpointStatus.NOT_STARTED


def _quantified_state(tmp_path: Path) -> WorkflowState:
    """State after Kallisto finished, with abundance files on disk."""
    state = WorkflowState(dataset_id="GSE210935")
    fastq_dir = tmp_path / "fastq"
    fastq_dir.mkdir()
    files = []
    for s in ("S1", "S2"):
        f = tmp_path / "abundance" / s / "abundance.tsv"
        f.parent.mkdir(parents=True)
        f.write_text("target_id\tlength\teff_length\test_counts\ttpm\n")
        files.append(str(f))
    state.fastq_dir = str(fastq_dir)
    state.kallisto_fastq_dir = str(fastq_dir)
    state.abundance_files = files
    state.tx2gene_file_used = str(tmp_path / "t2g.txt")
    state.update_checkpoint("kallisto_quant", CheckpointStatus.COMPLETED)
    return state


async def _test_naming_retry_does_not_rerun_kallisto(tmp_path):
    """Reflection after a naming error -> contrast design -> Kallisto is reused.

    The reflection model asks to retry from 'kallisto_quant' on purpose; the
    guard must route the retry to contrast design instead.
    """
    state = _quantified_state(tmp_path)
    state.metadata_df = _metadata("GSE210935")
    state.merged_column = "characteristics_ch1.3.treatment"
    from uorca.core.naming import assign_level_names

    state.group_level_names = assign_level_names(
        state.metadata_df[state.merged_column].unique().tolist()
    )
    state.r_group_column = R_GROUP_COLUMN
    state.analysis_history.append(
        {"node": "RunDEAnalysisNode", "error": "bad name", "checkpoint": "de_analysis",
         "error_type": "naming"}
    )
    ctx = _ctx(state, tmp_path)

    reflection = ReflectionOut(
        root_cause="names",
        failed_checkpoint="de_analysis",
        retry_strategy="retry_from_checkpoint",
        retry_from="kallisto_quant",
        reasoning="r",
    )
    with patch.object(
        reflection_agent, "run", new=AsyncMock(return_value=type("R", (), {"output": reflection})())
    ):
        retry_node = await ReflectNode(failed_checkpoint="de_analysis").run(ctx)
    assert isinstance(retry_node, ContrastDesignNode)
    assert state.checkpoints["kallisto_quant"] == CheckpointStatus.COMPLETED

    names = state.group_level_names
    model, _ = _contrast_model(
        [[{"name": "GMIL4_vs_GM_8h", "expression": f"{names['8h_GMIL4']} - {names['8h_GM']}"}]]
    )
    with contrast_design_agent.override(model=model):
        kallisto_node = await retry_node.run(ctx)
    assert isinstance(kallisto_node, KallistoQuantifyNode)

    with patch(
        "uorca.analysis.agents.analysis.run_kallisto_quantification_core",
        new=AsyncMock(side_effect=AssertionError("Kallisto must not run again")),
    ) as quant:
        nxt = await kallisto_node.run(ctx)
    assert isinstance(nxt, PrepareEdgeRNode)
    quant.assert_not_called()


@pytest.mark.parametrize(
    "change",
    ["checkpoint", "fastq_dir", "missing_file"],
)
def test_kallisto_reuse_is_refused_when_output_is_not_current(tmp_path, change):
    from uorca.graph.nodes.kallisto import _can_reuse_quantification

    state = _quantified_state(tmp_path)
    assert _can_reuse_quantification(state)
    if change == "checkpoint":
        state.checkpoints["kallisto_quant"] = CheckpointStatus.NOT_STARTED
    elif change == "fastq_dir":
        state.fastq_dir = str(tmp_path / "other")
    else:
        Path(state.abundance_files[0]).unlink()  # type: ignore[index]
    assert not _can_reuse_quantification(state)


async def _test_non_naming_error_keeps_reflection_choice(tmp_path):
    state = _quantified_state(tmp_path)
    state.analysis_history.append(
        {"node": "KallistoQuantifyNode", "error": "index", "checkpoint": "kallisto_quant",
         "error_type": "other"}
    )
    reflection = ReflectionOut(
        root_cause="index",
        failed_checkpoint="kallisto_quant",
        retry_strategy="retry_from_checkpoint",
        retry_from="kallisto_quant",
        reasoning="r",
    )
    with patch.object(
        reflection_agent, "run", new=AsyncMock(return_value=type("R", (), {"output": reflection})())
    ):
        node = await ReflectNode(failed_checkpoint="kallisto_quant").run(_ctx(state, tmp_path))
    assert isinstance(node, KallistoQuantifyNode)
    assert state.checkpoints["kallisto_quant"] == CheckpointStatus.NOT_STARTED


def test_gse161060_metadata_node_keeps_both_cell_types(tmp_path):
    asyncio.run(_test_gse161060_metadata_node_keeps_both_cell_types(tmp_path))


def test_gse210935_contrast_node_retries_bad_design_before_kallisto(tmp_path):
    asyncio.run(_test_gse210935_contrast_node_retries_bad_design_before_kallisto(tmp_path))


def test_persistently_bad_design_fails_early_with_clear_message(tmp_path):
    asyncio.run(_test_persistently_bad_design_fails_early_with_clear_message(tmp_path))


def test_naming_retry_does_not_rerun_kallisto(tmp_path):
    asyncio.run(_test_naming_retry_does_not_rerun_kallisto(tmp_path))


def test_non_naming_error_keeps_reflection_choice(tmp_path):
    asyncio.run(_test_non_naming_error_keeps_reflection_choice(tmp_path))


def test_contrast_manifest_check(tmp_path):
    """Python raises when RNAseq.R did not write a requested contrast."""
    from uorca.analysis.agents.analysis import _check_contrast_manifest

    (tmp_path / "KO_vs_WT").mkdir()
    (tmp_path / "KO_vs_WT" / "DEG.csv").write_text("Gene\n")
    pd.DataFrame(
        {"contrast_name": ["KO_vs_WT"], "folder": ["KO_vs_WT"], "n_genes": [1], "n_samples": [6]}
    ).to_csv(tmp_path / "contrast_manifest.csv", index=False)

    _check_contrast_manifest(str(tmp_path), ["KO_vs_WT"])
    with pytest.raises(RuntimeError, match="Missing from the manifest: \\['WT_vs_KO'\\]"):
        _check_contrast_manifest(str(tmp_path), ["KO_vs_WT", "WT_vs_KO"])
    (tmp_path / "KO_vs_WT" / "DEG.csv").unlink()
    with pytest.raises(RuntimeError, match="without DEG.csv"):
        _check_contrast_manifest(str(tmp_path), ["KO_vs_WT"])
    (tmp_path / "contrast_manifest.csv").unlink()
    with pytest.raises(RuntimeError, match="did not write"):
        _check_contrast_manifest(str(tmp_path), ["KO_vs_WT"])
