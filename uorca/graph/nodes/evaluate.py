"""Evaluation node - check if analysis succeeded."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union

from pydantic_graph import BaseNode, End, GraphRunContext

from ..deps import WorkflowDeps
from ..state import CheckpointStatus, WorkflowState

if TYPE_CHECKING:
    from .reflect import ReflectNode

logger = logging.getLogger(__name__)


def cleanup_fastq_and_sra(output_dir: Path) -> None:
    """Remove FASTQ and SRA files from the output directory to free disk space.

    Called on both success and failure to prevent disk space accumulation.
    """
    total_freed = 0

    for subdir_name in ("fastq", "sra"):
        subdir = output_dir / subdir_name
        if not subdir.exists():
            continue

        # Calculate size before removal
        dir_size = sum(
            f.stat().st_size for f in subdir.rglob("*") if f.is_file()
        )
        total_freed += dir_size

        try:
            shutil.rmtree(subdir)
            logger.info(
                "Removed %s/ (%.2f GB freed)",
                subdir_name,
                dir_size / (1024**3),
            )
        except Exception as e:
            logger.warning("Failed to remove %s/: %s", subdir_name, e)

    if total_freed > 0:
        logger.info(
            "Cleanup complete: %.2f GB freed", total_freed / (1024**3)
        )


def write_analysis_info(state: WorkflowState, deps: WorkflowDeps) -> None:
    """Write analysis_info.json to the metadata directory for the results explorer.

    This file is required by ResultsIntegrator to load and display results.
    """
    metadata_dir = deps.output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    info_path = metadata_dir / "analysis_info.json"

    # Build checkpoint history
    checkpoints = {}
    for cp_name, cp_status in state.checkpoints.items():
        checkpoints[cp_name] = {
            "status": cp_status.value if hasattr(cp_status, "value") else str(cp_status),
        }

    # Build contrasts list with full detail
    contrasts = []
    if state.contrasts:
        for c in state.contrasts:
            if isinstance(c, dict):
                contrasts.append(c)
            else:
                contrasts.append({"name": str(c)})

    info = {
        "accession": state.dataset_id,
        "organism": state.organism or "Unknown",
        "dataset_information": state.dataset_information or "",
        "analysis_success": state.analysis_success,
        "reflection_iterations": state.reflection_iterations,
        "merged_column": state.merged_column,
        "unique_groups": state.unique_groups or [],
        "kallisto_index_used": state.kallisto_index_used,
        "tx2gene_file_used": state.tx2gene_file_used,
        "checkpoints": checkpoints,
        "contrasts": contrasts,
    }

    try:
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2, default=str)
        logger.info("Wrote analysis_info.json to %s", info_path)
    except Exception as e:
        logger.warning("Failed to write analysis_info.json: %s", e)


@dataclass
class EvaluateNode(BaseNode[WorkflowState, WorkflowDeps, str]):
    """Evaluate analysis success and determine next action."""

    async def run(
        self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]
    ) -> Union[End[str], "ReflectNode"]:
        from .reflect import ReflectNode

        # Check all critical checkpoints
        critical_checkpoints = [
            "extraction",
            "metadata_analysis",
            "contrast_design",
            "kallisto_quant",
            "edger_prep",
            "de_analysis",
        ]

        failed = [
            cp
            for cp in critical_checkpoints
            if ctx.state.checkpoints.get(cp) == CheckpointStatus.FAILED
        ]

        if failed:
            logger.warning(f"Failed checkpoints: {failed}")
            return ReflectNode(failed_checkpoint=failed[0])

        # All passed - success!
        ctx.state.analysis_success = True
        ctx.state.analysis_diagnostics = (
            f"Analysis completed successfully. "
            f"Results at: {ctx.state.deg_results_path}"
        )

        logger.info("Analysis completed successfully!")

        # Write analysis_info.json for the results explorer
        write_analysis_info(ctx.state, ctx.deps)

        # Cleanup FASTQ/SRA files on success
        if ctx.deps.cleanup_after_success:
            cleanup_fastq_and_sra(ctx.deps.output_dir)

        return End(ctx.state.deg_results_path or "Analysis complete")
