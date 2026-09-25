"""Contrast design node - AI-powered contrast generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import pandas as pd
from pydantic_graph import BaseNode, GraphRunContext

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
            # Render the groups as a closed enumeration so the LLM reuses them
            # verbatim instead of inventing operators ('==') or column-name
            # prefixes ('group<level>') in the expression string.
            groups_lines = "\n".join(f"- {g}" for g in ctx.state.unique_groups)

            prompt = f"""Design contrasts for differential expression analysis.

Analysis grouping column: {ctx.state.merged_column}

Group levels (use these tokens VERBATIM in every expression — no quotes,
no `{ctx.state.merged_column}` prefix, no `==`):
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
- `name` must be a valid R identifier: only letters/digits/underscores,
  starting with a letter; use underscores in place of any hyphens or spaces.

Design biologically meaningful contrasts that:
1. Address the research question if provided
2. Compare relevant experimental conditions
"""

            result = await contrast_design_agent.run(
                prompt,
                deps=ctx.deps,
            )

            ctx.state.contrasts = [c.model_dump() for c in result.output.contrasts]

            # Save contrasts to file
            contrasts_df = pd.DataFrame(ctx.state.contrasts)
            contrast_path = ctx.deps.output_dir / "metadata" / "contrasts.csv"
            contrast_path.parent.mkdir(parents=True, exist_ok=True)
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
                }
            )
            return ReflectNode(failed_checkpoint="contrast_design")
