"""Kallisto quantification node - wraps existing Kallisto tool."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import kallisto_index_agent
from ..deps import WorkflowDeps
from ..state import CheckpointStatus, WorkflowState

if TYPE_CHECKING:
    from .edger import PrepareEdgeRNode
    from .reflect import ReflectNode

logger = logging.getLogger(__name__)


def _discover_files(resource_dir: Path, pattern: str) -> List[str]:
    """Discover files matching pattern in resource directory."""
    matches = list(resource_dir.rglob(pattern))
    return [str(m) for m in matches]


@dataclass
class KallistoQuantifyNode(BaseNode[WorkflowState, WorkflowDeps, str]):
    """Run Kallisto quantification on FASTQ files."""

    async def run(
        self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]
    ) -> Union["PrepareEdgeRNode", "ReflectNode"]:
        from uorca.analysis.agents.analysis import run_kallisto_quantification_core

        from .edger import PrepareEdgeRNode
        from .reflect import ReflectNode

        ctx.state.update_checkpoint("kallisto_quant", CheckpointStatus.IN_PROGRESS)

        try:
            # Validate fastq_dir is set
            if not ctx.state.fastq_dir:
                raise ValueError("FASTQ directory not set - extraction may have failed")

            # Step 1: Discover available index and tx2gene files
            logger.info(f"Discovering Kallisto indices in {ctx.deps.resource_dir}")
            available_indices = _discover_files(ctx.deps.resource_dir, "*.idx")
            available_tx2gene = _discover_files(ctx.deps.resource_dir, "*t2g.txt")

            logger.info(f"Found {len(available_indices)} index files")
            logger.info(f"Found {len(available_tx2gene)} tx2gene files")

            if not available_indices:
                raise FileNotFoundError(
                    f"No Kallisto index files (*.idx) found in {ctx.deps.resource_dir}"
                )

            # Step 2: Let AI agent select appropriate index based on organism
            selection_prompt = f"""Select the appropriate Kallisto index and tx2gene file.

Organism detected: {ctx.state.organism}
Dataset: {ctx.state.dataset_id}
Dataset information: {ctx.state.dataset_information or 'Not available'}

Available Kallisto index files:
{chr(10).join(f'- {idx}' for idx in available_indices)}

Available tx2gene files:
{chr(10).join(f'- {t2g}' for t2g in available_tx2gene)}

Select the index and tx2gene file that best match the organism "{ctx.state.organism}".
"""

            logger.info("Calling kallisto_index_agent for intelligent selection")
            selection_result = await kallisto_index_agent.run(
                selection_prompt,
                deps=ctx.deps,
            )

            index_path = selection_result.output.selected_index
            tx2gene_path = selection_result.output.tx2gene_file

            logger.info(f"Agent selected index: {index_path}")
            logger.info(f"Agent selected tx2gene: {tx2gene_path}")
            logger.info(f"Agent reasoning: {selection_result.output.reasoning}")
            logger.info(f"Agent confidence: {selection_result.output.confidence}")

            # Step 3: Run Kallisto quantification with selected paths
            kallisto_result = await run_kallisto_quantification_core(
                fastq_dir=ctx.state.fastq_dir,
                output_dir=ctx.deps.output_dir,
                index_path=index_path,
                tx2gene_path=tx2gene_path,
                sample_mapping=ctx.state.fastq_sample_mapping,  # For clean output naming
            )

            ctx.state.abundance_files = kallisto_result.abundance_files
            ctx.state.kallisto_index_used = kallisto_result.index_used
            ctx.state.tx2gene_file_used = kallisto_result.tx2gene_file

            if not ctx.state.abundance_files:
                raise ValueError("Kallisto produced no abundance files")

            logger.info(
                f"Kallisto completed: {len(ctx.state.abundance_files)} samples quantified"
            )

            ctx.state.update_checkpoint("kallisto_quant", CheckpointStatus.COMPLETED)
            return PrepareEdgeRNode()

        except Exception as e:
            logger.error(f"Kallisto quantification failed: {e}")
            ctx.state.update_checkpoint("kallisto_quant", CheckpointStatus.FAILED)
            ctx.state.analysis_history.append(
                {
                    "node": "KallistoQuantifyNode",
                    "error": str(e),
                    "checkpoint": "kallisto_quant",
                    "organism": ctx.state.organism,
                    "fastq_dir": ctx.state.fastq_dir,
                }
            )
            return ReflectNode(failed_checkpoint="kallisto_quant")
