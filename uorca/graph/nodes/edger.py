"""edgeR preparation and analysis nodes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from pydantic_graph import BaseNode, GraphRunContext

from ..deps import WorkflowDeps
from ..state import CheckpointStatus, WorkflowState

if TYPE_CHECKING:
    from .evaluate import EvaluateNode
    from .reflect import ReflectNode

logger = logging.getLogger(__name__)


@dataclass
class PrepareEdgeRNode(BaseNode[WorkflowState, WorkflowDeps, str]):
    """Prepare sample mapping for edgeR analysis."""

    async def run(
        self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]
    ) -> Union["RunDEAnalysisNode", "ReflectNode"]:
        from uorca.analysis.agents.analysis import prepare_edgeR_analysis_core

        from .reflect import ReflectNode

        ctx.state.update_checkpoint("edger_prep", CheckpointStatus.IN_PROGRESS)

        try:
            prep_result = await prepare_edgeR_analysis_core(
                abundance_files=ctx.state.abundance_files,
                metadata_df=ctx.state.metadata_df,
                merged_column=ctx.state.merged_column,
                output_dir=ctx.deps.output_dir,
                deps=ctx.deps,  # For LLM agent fallback
            )

            ctx.state.sample_mapping_path = prep_result.sample_mapping_path

            logger.info(
                f"edgeR preparation complete: {prep_result.samples_matched} samples mapped"
            )

            ctx.state.update_checkpoint("edger_prep", CheckpointStatus.COMPLETED)
            return RunDEAnalysisNode()

        except Exception as e:
            logger.error(f"edgeR preparation failed: {e}")
            ctx.state.update_checkpoint("edger_prep", CheckpointStatus.FAILED)
            ctx.state.analysis_history.append(
                {
                    "node": "PrepareEdgeRNode",
                    "error": str(e),
                    "checkpoint": "edger_prep",
                }
            )
            return ReflectNode(failed_checkpoint="edger_prep")


@dataclass
class RunDEAnalysisNode(BaseNode[WorkflowState, WorkflowDeps, str]):
    """Run edgeR/limma differential expression analysis."""

    async def run(
        self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]
    ) -> Union["EvaluateNode", "ReflectNode"]:
        from uorca.analysis.agents.analysis import run_edger_limma_analysis_core

        from .evaluate import EvaluateNode
        from .reflect import ReflectNode

        ctx.state.update_checkpoint("de_analysis", CheckpointStatus.IN_PROGRESS)

        try:
            de_result = await run_edger_limma_analysis_core(
                sample_mapping_path=ctx.state.sample_mapping_path,
                merged_column=ctx.state.merged_column,
                contrasts=ctx.state.contrasts,
                tx2gene_path=ctx.state.tx2gene_file_used,
                output_dir=ctx.deps.output_dir,
            )

            ctx.state.deg_results_path = de_result.results_dir

            logger.info(f"DE analysis complete: results at {ctx.state.deg_results_path}")

            ctx.state.update_checkpoint("de_analysis", CheckpointStatus.COMPLETED)
            return EvaluateNode()

        except Exception as e:
            logger.error(f"DE analysis failed: {e}")
            ctx.state.update_checkpoint("de_analysis", CheckpointStatus.FAILED)
            ctx.state.analysis_history.append(
                {
                    "node": "RunDEAnalysisNode",
                    "error": str(e),
                    "checkpoint": "de_analysis",
                }
            )
            return ReflectNode(failed_checkpoint="de_analysis")
