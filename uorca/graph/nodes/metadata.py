"""Metadata analysis node - AI-powered column selection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from pydantic_graph import BaseNode, GraphRunContext

from uorca.core.design_validation import r_make_names
from uorca.core.naming import (
    R_GROUP_COLUMN,
    NamingError,
    assign_level_names,
    check_joined_levels,
    is_missing_value,
    legacy_r_naming_hazards,
    missing_group_signals,
)

from ..agents import metadata_analysis_agent
from ..deps import WorkflowDeps
from ..state import CheckpointStatus, WorkflowState

if TYPE_CHECKING:
    from .contrasts import ContrastDesignNode
    from .reflect import ReflectNode

logger = logging.getLogger(__name__)


@dataclass
class MetadataAnalysisNode(BaseNode[WorkflowState, WorkflowDeps, str]):
    """AI-powered metadata column selection for DE analysis."""

    async def run(
        self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]
    ) -> Union["ContrastDesignNode", "ReflectNode"]:
        from .contrasts import ContrastDesignNode
        from .reflect import ReflectNode

        ctx.state.update_checkpoint("metadata_analysis", CheckpointStatus.IN_PROGRESS)

        try:
            df = ctx.state.metadata_df
            if df is None:
                raise ValueError("metadata_df is None - extraction may have failed")

            # AI analysis for GEO data - identify grouping columns
            # Build reflection context
            reflection_context = ""
            if ctx.state.reflections:
                reflection_context = "\nPrevious analysis attempts and feedback:\n"
                reflection_context += "\n".join(ctx.state.reflections[-2:])

            # Prepare analysis columns (remove GEO ID columns and the R name
            # column a previous attempt may have added)
            id_cols = ["GSM", "SRX", "SRR", "geo_accession", R_GROUP_COLUMN]
            analysis_cols = [c for c in df.columns if c not in id_cols]

            prompt = f"""Analyse this RNA-seq metadata for differential expression grouping.

Available columns for analysis: {analysis_cols}

Sample data:
{df[analysis_cols].head(10).to_string() if analysis_cols else df.head(10).to_string()}

Dataset information: {ctx.state.dataset_information or 'Not available'}
Research question: {ctx.state.research_question or 'General exploratory analysis'}
{reflection_context}
"""

            result = await metadata_analysis_agent.run(
                prompt,
                deps=ctx.deps,
            )

            ctx.state.merged_column = result.output.merged_column_name
            ctx.state.unique_groups = result.output.unique_groups

            # Validate selected columns exist in dataframe
            missing_cols = [c for c in result.output.selected_columns if c not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Selected columns not found in metadata: {missing_cols}. "
                    f"Available columns: {list(df.columns)}"
                )

            if R_GROUP_COLUMN in (ctx.state.merged_column, *result.output.selected_columns):
                raise ValueError(
                    f"{R_GROUP_COLUMN!r} is reserved for the R level names; choose "
                    "metadata columns and a merged column name other than it."
                )

            # Apply column creation/renaming
            if len(result.output.selected_columns) > 1:
                # Joining with "_" can merge different groups (a_b + c and
                # a + b_c); refuse before the merge hides it.
                check_joined_levels(
                    df[result.output.selected_columns].astype(str).itertuples(index=False)
                )
                # Multiple columns: merge them into a new column
                df[ctx.state.merged_column] = (
                    df[result.output.selected_columns].astype(str).agg("_".join, axis=1)
                )
            elif result.output.selected_columns[0] != ctx.state.merged_column:
                # Single column with different name: copy to new column name
                df[ctx.state.merged_column] = df[result.output.selected_columns[0]]
            # else: single column with same name - column already exists, no action needed

            # Missing or empty group values: warn and flag for review, or stop
            # (one switch, naming.MISSING_GROUP_POLICY; owner question Q13).
            selected = list(result.output.selected_columns)
            ctx.state.review_signals = missing_group_signals(
                df[selected].itertuples(index=False), selected
            )

            # Normalise merged_column to an R-safe identifier. R's read.csv with
            # default check.names=TRUE rewrites a header like "disease state" to
            # "disease.state", which then no longer matches the merged_column
            # string passed to RNAseq.R as a CLI arg.
            # r_make_names also covers a leading "_" or "." + digit and
            # reserved words, which read.csv renames too (F20).
            safe_name = r_make_names(re.sub(r"[^A-Za-z0-9_.]", "_", ctx.state.merged_column))
            if safe_name != ctx.state.merged_column:
                logger.info(
                    "Sanitised merged_column for R: %r → %r",
                    ctx.state.merged_column,
                    safe_name,
                )
                df.rename(columns={ctx.state.merged_column: safe_name}, inplace=True)
                ctx.state.merged_column = safe_name

            # An empty or blank label is a missing value, not a group named ""
            # (which R would call "X").
            merged = df[ctx.state.merged_column]
            df[ctx.state.merged_column] = merged.mask(merged.map(is_missing_value))

            # CRITICAL: Save modified dataframe back to state
            ctx.state.metadata_df = df

            # CRITICAL: Use actual unique values from the merged column, not LLM's predicted values
            # This ensures contrasts use the real factor levels present in the data
            actual_groups = df[ctx.state.merged_column].unique().tolist()
            logger.info(f"Selected grouping column: {ctx.state.merged_column}")
            logger.info(f"LLM expected groups: {ctx.state.unique_groups}")
            logger.info(f"Actual groups in data: {actual_groups}")

            # Override with actual groups to ensure downstream consistency
            ctx.state.unique_groups = actual_groups

            # Validate we have enough groups
            if len(ctx.state.unique_groups) < 2:
                raise ValueError(
                    f"Need at least 2 groups for DE analysis, found: {ctx.state.unique_groups}"
                )

            # Give every level a valid, unique R name now, before Kallisto, so
            # RNAseq.R renames nothing (renaming audit 2026-10-28, Q4).
            raw_levels = [str(g) for g in actual_groups]
            for hazard in legacy_r_naming_hazards(raw_levels):
                logger.info("R make.names hazard avoided: %s", hazard)
            level_names = assign_level_names(raw_levels)
            df[R_GROUP_COLUMN] = df[ctx.state.merged_column].map(
                lambda v: level_names[str(v)]
            )
            ctx.state.metadata_df = df
            ctx.state.r_group_column = R_GROUP_COLUMN
            ctx.state.group_level_names = level_names
            ctx.state.group_source_columns = list(result.output.selected_columns)
            logger.info("R level names: %s", level_names)

            ctx.state.update_checkpoint("metadata_analysis", CheckpointStatus.COMPLETED)
            return ContrastDesignNode()

        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")
            ctx.state.update_checkpoint("metadata_analysis", CheckpointStatus.FAILED)
            ctx.state.analysis_history.append(
                {
                    "node": "MetadataAnalysisNode",
                    "error": str(e),
                    "checkpoint": "metadata_analysis",
                    "error_type": "naming" if isinstance(e, NamingError) else "other",
                }
            )
            return ReflectNode(failed_checkpoint="metadata_analysis")
