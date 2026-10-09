"""GEO extraction node - wraps existing extraction tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from pydantic_graph import BaseNode, GraphRunContext

from ..deps import WorkflowDeps
from ..state import CheckpointStatus, WorkflowState

if TYPE_CHECKING:
    from .metadata import MetadataAnalysisNode
    from .reflect import ReflectNode

logger = logging.getLogger(__name__)


@dataclass
class GEOExtractNode(BaseNode[WorkflowState, WorkflowDeps, str]):
    """Extract metadata and FASTQ files from GEO/SRA."""

    async def run(
        self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]
    ) -> Union["MetadataAnalysisNode", "ReflectNode"]:
        from uorca.analysis.agents.extraction import (
            download_fastqs_core,
            fetch_geo_metadata_core,
        )

        from .metadata import MetadataAnalysisNode
        from .reflect import ReflectNode

        ctx.state.update_checkpoint("extraction", CheckpointStatus.IN_PROGRESS)

        try:
            # Fetch GEO metadata (reuse existing logic)
            metadata_result = await fetch_geo_metadata_core(
                accession=ctx.state.dataset_id,
                output_dir=ctx.deps.output_dir,
                entrez_email=ctx.deps.entrez_email,
                entrez_api_key=ctx.deps.entrez_api_key,
            )

            ctx.state.metadata_df = metadata_result.metadata_df
            ctx.state.metadata_path = metadata_result.metadata_path
            ctx.state.organism = metadata_result.organism
            ctx.state.dataset_information = metadata_result.dataset_info

            # Download FASTQ files
            fastq_result = await download_fastqs_core(
                metadata_df=ctx.state.metadata_df,
                output_dir=ctx.deps.output_dir,
                http_client=ctx.deps.http_client,
            )

            ctx.state.fastq_dir = fastq_result.fastq_dir
            ctx.state.update_checkpoint("extraction", CheckpointStatus.COMPLETED)

            return MetadataAnalysisNode()

        except Exception as e:
            logger.error(f"GEO extraction failed: {e}")
            ctx.state.update_checkpoint("extraction", CheckpointStatus.FAILED)
            ctx.state.analysis_history.append(
                {
                    "node": "GEOExtractNode",
                    "error": str(e),
                    "checkpoint": "extraction",
                }
            )
            return ReflectNode(failed_checkpoint="extraction")
