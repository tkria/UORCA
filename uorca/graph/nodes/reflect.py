"""Reflection node - AI-powered failure diagnosis and retry routing."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from pydantic_graph import BaseNode, End, GraphRunContext

from ..agents import reflection_agent
from ..deps import WorkflowDeps
from ..state import CheckpointStatus, WorkflowState

if TYPE_CHECKING:
    from .contrasts import ContrastDesignNode
    from .edger import PrepareEdgeRNode, RunDEAnalysisNode
    from .geo_extract import GEOExtractNode
    from .kallisto import KallistoQuantifyNode
    from .metadata import MetadataAnalysisNode
    from .user_extract import UserDataExtractNode

logger = logging.getLogger(__name__)


@dataclass
class ReflectNode(BaseNode[WorkflowState, WorkflowDeps, str]):
    """AI-powered failure diagnosis and retry routing."""

    failed_checkpoint: str

    async def run(
        self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]
    ) -> Union[
        "GEOExtractNode",
        "UserDataExtractNode",
        "MetadataAnalysisNode",
        "ContrastDesignNode",
        "KallistoQuantifyNode",
        "PrepareEdgeRNode",
        "RunDEAnalysisNode",
        End[str],
    ]:
        # Check retry limit
        ctx.state.reflection_iterations += 1
        if ctx.state.reflection_iterations > ctx.deps.max_retries:
            ctx.state.analysis_success = False
            ctx.state.analysis_diagnostics = (
                f"Analysis failed after {ctx.deps.max_retries} retry attempts. "
                f"Last failure: {self.failed_checkpoint}"
            )
            return End(
                f"Analysis failed: max retries exceeded at {self.failed_checkpoint}"
            )

        logger.info(
            f"Reflection iteration {ctx.state.reflection_iterations} "
            f"for {self.failed_checkpoint}"
        )

        # Build reflection prompt
        recent_history = ctx.state.analysis_history[-5:]

        prompt = f"""The RNA-seq analysis failed at checkpoint: {self.failed_checkpoint}

Current checkpoint statuses:
{json.dumps({k: v.value for k, v in ctx.state.checkpoints.items()}, indent=2)}

Last successful checkpoint: {ctx.state.last_successful_checkpoint or 'None'}

Recent analysis history:
{json.dumps(recent_history, indent=2, default=str)}

Previous reflections:
{ctx.state.reflections[-3:] if ctx.state.reflections else 'None'}

Dataset: {ctx.state.dataset_id}
Organism: {ctx.state.organism}

Diagnose the failure and recommend a retry strategy.
"""

        result = await reflection_agent.run(
            prompt,
            deps=ctx.deps,
        )

        # Store reflection
        ctx.state.reflections.append(result.output.reasoning)

        logger.info(f"Reflection diagnosis: {result.output.root_cause}")
        logger.info(f"Recommended strategy: {result.output.retry_strategy}")

        # Handle abort
        if result.output.retry_strategy == "abort":
            ctx.state.analysis_success = False
            ctx.state.analysis_diagnostics = (
                f"Analysis aborted: {result.output.root_cause}"
            )
            return End(f"Analysis aborted: {result.output.root_cause}")

        # Apply parameter changes if suggested
        for key, value in result.output.parameter_changes.items():
            if hasattr(ctx.state, key):
                setattr(ctx.state, key, value)
                logger.info(f"Applied parameter change: {key} = {value}")

        # Determine retry point
        retry_from = result.output.retry_from or self.failed_checkpoint

        # Reset failed checkpoint to allow retry
        ctx.state.checkpoints[retry_from] = CheckpointStatus.NOT_STARTED

        # Route to appropriate node
        return self._get_retry_node(retry_from, ctx.state)

    def _get_retry_node(
        self, checkpoint: str, state: WorkflowState
    ) -> Union[
        "GEOExtractNode",
        "UserDataExtractNode",
        "MetadataAnalysisNode",
        "ContrastDesignNode",
        "KallistoQuantifyNode",
        "PrepareEdgeRNode",
        "RunDEAnalysisNode",
    ]:
        """Map checkpoint name to node class."""
        from .contrasts import ContrastDesignNode
        from .edger import PrepareEdgeRNode, RunDEAnalysisNode
        from .geo_extract import GEOExtractNode
        from .kallisto import KallistoQuantifyNode
        from .metadata import MetadataAnalysisNode
        from .user_extract import UserDataExtractNode

        # For extraction checkpoint, route based on data source
        if checkpoint == "extraction":
            if state.data_source == "user":
                # Recreate UserDataExtractNode with stored config
                if state.user_fastq_dir and state.user_metadata_path:
                    return UserDataExtractNode(
                        fastq_dir=state.user_fastq_dir,
                        metadata_path=state.user_metadata_path,
                        organism=state.user_organism,
                        description=state.user_description,
                    )
                else:
                    logger.error(
                        "Cannot retry user data extraction - missing config in state"
                    )
                    # Fall through to GEOExtractNode which will fail gracefully
            return GEOExtractNode()

        checkpoint_to_node = {
            "metadata_analysis": MetadataAnalysisNode,
            "contrast_design": ContrastDesignNode,
            "kallisto_quant": KallistoQuantifyNode,
            "edger_prep": PrepareEdgeRNode,
            "de_analysis": RunDEAnalysisNode,
        }

        node_class = checkpoint_to_node.get(checkpoint, MetadataAnalysisNode)
        return node_class()
