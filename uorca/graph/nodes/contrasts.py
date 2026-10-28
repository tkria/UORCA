"""Contrast design node - AI-powered contrast generation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Union

import pandas as pd
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_graph import BaseNode, GraphRunContext

from uorca.core.naming import NamingError, build_naming_record, normalise_contrasts

from ..agents import contrast_design_agent
from ..deps import WorkflowDeps
from ..state import CheckpointStatus, WorkflowState

if TYPE_CHECKING:
    from .kallisto import KallistoQuantifyNode
    from .reflect import ReflectNode

logger = logging.getLogger(__name__)


@dataclass
class ContrastDesignNode(BaseNode[WorkflowState, WorkflowDeps, str]):
    """AI-powered contrast design for differential expression."""

    async def run(
        self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]
    ) -> Union["KallistoQuantifyNode", "ReflectNode"]:
        from .kallisto import KallistoQuantifyNode
        from .reflect import ReflectNode

        ctx.state.update_checkpoint("contrast_design", CheckpointStatus.IN_PROGRESS)

        try:
            level_names = ctx.state.group_level_names
            if not level_names or not ctx.state.r_group_column:
                raise ValueError(
                    "Group level R names are missing; MetadataAnalysisNode must run first."
                )

            # Render the groups as a closed enumeration of R names so the LLM
            # reuses them verbatim. The raw value is shown for meaning only:
            # raw values can contain '+', '-', '/' and spaces, which R would
            # read as operators.
            groups_lines = "\n".join(
                f"- {r}" + (f"    (raw value: {raw})" if raw != r else "")
                for raw, r in level_names.items()
            )

            prompt = f"""Design contrasts for differential expression analysis.

Analysis grouping column: {ctx.state.merged_column}

Group level names (use these tokens VERBATIM in every expression — never the
raw value in brackets, no quotes, no `{ctx.state.merged_column}` prefix, no `==`):
{groups_lines}

Dataset information: {ctx.state.dataset_information or 'Not available'}
Research question: {ctx.state.research_question or 'Not specified'}

Output rules for each contrast:
- `expression` is the level tokens combined with ` - ` (and optionally ` + ` /
  parentheses for averages or interactions). Examples (substitute LEVEL_X with
  one of the tokens listed above):
      LEVEL_A - LEVEL_B
      (LEVEL_A + LEVEL_B)/2 - LEVEL_C
      (LEVEL_A - LEVEL_B) - (LEVEL_C - LEVEL_D)
  Wrong patterns: `{ctx.state.merged_column} == '...'`, `{ctx.state.merged_column}LEVEL_A`, `'LEVEL_A'`.
- `name` must start with a letter and contain only letters, digits and
  underscores (no dots, hyphens or spaces). Names must be unique, ignoring case.

Design biologically meaningful contrasts that:
1. Address the research question if provided
2. Compare relevant experimental conditions
"""

            call_deps = replace(
                ctx.deps, group_level_names=dict(level_names), contrast_issues=None
            )
            try:
                result = await contrast_design_agent.run(prompt, deps=call_deps)
            except UnexpectedModelBehavior as exc:
                if call_deps.contrast_issues:
                    raise NamingError(
                        "The contrast design still failed the deterministic checks "
                        "after the model's retries:\n- "
                        + "\n- ".join(call_deps.contrast_issues)
                    ) from exc
                raise

            # Same whitespace as was validated, so R reads what Python checked.
            contrasts = normalise_contrasts([c.model_dump() for c in result.output.contrasts])

            # Deterministic gate before Kallisto: the output validator has
            # already asked the model to fix these, so any issue left is a
            # failure of this design (renaming audit 2026-10-28, Q1/Q2).
            counts = (
                ctx.state.metadata_df[ctx.state.merged_column].astype(str).value_counts().to_dict()
                if ctx.state.metadata_df is not None
                else {}
            )
            naming_record = build_naming_record(
                group_column=ctx.state.merged_column or "",
                source_columns=ctx.state.group_source_columns or [],
                level_names=level_names,
                level_counts=counts,
                contrasts=contrasts,
            )
            ctx.state.contrasts = contrasts

            metadata_dir = ctx.deps.output_dir / "metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            (metadata_dir / "naming.json").write_text(json.dumps(naming_record, indent=2))

            # Save contrasts to file
            contrasts_df = pd.DataFrame(ctx.state.contrasts)
            contrast_path = metadata_dir / "contrasts.csv"
            contrasts_df.to_csv(contrast_path, index=False)
            ctx.state.contrast_path = str(contrast_path)

            logger.info(f"Designed {len(ctx.state.contrasts)} contrasts")
            for c in ctx.state.contrasts:
                logger.info(f"  - {c['name']}: {c['expression']}")

            ctx.state.update_checkpoint("contrast_design", CheckpointStatus.COMPLETED)
            return KallistoQuantifyNode()

        except Exception as e:
            logger.error(f"Contrast design failed: {e}")
            ctx.state.update_checkpoint("contrast_design", CheckpointStatus.FAILED)
            ctx.state.analysis_history.append(
                {
                    "node": "ContrastDesignNode",
                    "error": str(e),
                    "checkpoint": "contrast_design",
                    "error_type": "naming" if isinstance(e, NamingError) else "other",
                }
            )
            return ReflectNode(failed_checkpoint="contrast_design")
