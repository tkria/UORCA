"""User data extraction node - handles user-provided FASTQ files and metadata."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import pandas as pd
from pydantic_graph import BaseNode, End, GraphRunContext

from ..agents import fastq_metadata_linker_agent, fastq_pair_discovery_agent
from ..deps import WorkflowDeps
from ..models import FASTQMetadataLinkResult, FASTQPairDiscoveryResult
from ..state import CheckpointStatus, WorkflowState

if TYPE_CHECKING:
    from .metadata import MetadataAnalysisNode
    from .reflect import ReflectNode

logger = logging.getLogger(__name__)


@dataclass
class UserDataExtractNode(BaseNode[WorkflowState, WorkflowDeps, str]):
    """Extract and validate user-provided FASTQ files and metadata."""

    fastq_dir: str
    metadata_path: str
    organism: Optional[str] = None
    description: Optional[str] = None

    async def run(
        self, ctx: GraphRunContext[WorkflowState, WorkflowDeps]
    ) -> Union["MetadataAnalysisNode", "ReflectNode", End[str]]:
        from .metadata import MetadataAnalysisNode
        from .reflect import ReflectNode

        ctx.state.update_checkpoint("extraction", CheckpointStatus.IN_PROGRESS)

        try:
            # Step 1: Discover FASTQ pairs (try standard patterns first, then agent)
            fastq_dir = Path(self.fastq_dir)
            fastq_pairs = self._discover_fastq_pairs_standard(fastq_dir)

            if not fastq_pairs:
                # Standard patterns didn't match - use AI agent to discover pairs
                logger.info("Standard FASTQ naming patterns not detected, using AI agent")
                fastq_pairs = await self._discover_fastq_pairs_with_agent(
                    fastq_dir, ctx.deps
                )

            if not fastq_pairs:
                raise ValueError(
                    f"No paired FASTQ files found in {self.fastq_dir}. "
                    f"Could not detect pairs using standard patterns or AI analysis."
                )

            logger.info(f"Found {len(fastq_pairs)} FASTQ pairs")

            # Step 2: Load metadata
            metadata_df = pd.read_csv(self.metadata_path)
            logger.info(
                f"Loaded metadata: {len(metadata_df)} rows, columns: {list(metadata_df.columns)}"
            )

            # Step 3: Use LLM to match FASTQ files to metadata
            fastq_prefixes = [p[0] for p in fastq_pairs]

            link_prompt = self._build_linking_prompt(
                fastq_prefixes=fastq_prefixes,
                metadata_df=metadata_df,
                description=self.description or ctx.state.research_question,
            )

            link_result = await fastq_metadata_linker_agent.run(
                link_prompt,
                deps=ctx.deps,
            )

            logger.info(f"Linker result: can_match={link_result.output.can_match}")
            logger.info(f"Linker reasoning: {link_result.output.reasoning}")

            # Step 4: Handle matching failure
            if not link_result.output.can_match:
                return self._handle_matching_failure(ctx, link_result.output)

            # Step 5: Validate and apply mappings
            sample_id_col = link_result.output.sample_id_column
            if sample_id_col and sample_id_col not in metadata_df.columns:
                raise ValueError(
                    f"LLM selected column '{sample_id_col}' not found in metadata. "
                    f"Available: {list(metadata_df.columns)}"
                )

            # Create sample mapping (FASTQ prefix -> metadata row)
            mapping_dict = {
                m.fastq_prefix: m.metadata_id for m in link_result.output.mappings
            }

            # Validate coverage
            unmapped = set(fastq_prefixes) - set(mapping_dict.keys())
            if unmapped:
                logger.warning(f"Unmapped FASTQ files: {unmapped}")
                if len(unmapped) == len(fastq_prefixes):
                    raise ValueError("No FASTQ files could be mapped to metadata")

            # Add sample_id column for downstream compatibility
            if sample_id_col:
                metadata_df["sample_id"] = metadata_df[sample_id_col]
            else:
                metadata_df["sample_id"] = metadata_df.index.astype(str)

            # Step 6: Handle organism
            organism = self._resolve_organism(link_result.output)

            # Step 7: Save sample mapping
            sample_mapping_path = ctx.deps.output_dir / "metadata" / "sample_mapping.csv"
            sample_mapping_path.parent.mkdir(parents=True, exist_ok=True)
            mapping_df = pd.DataFrame(
                [{"fastq_prefix": k, "sample_id": v} for k, v in mapping_dict.items()]
            )
            mapping_df.to_csv(sample_mapping_path, index=False)
            logger.info(f"Sample mapping saved to {sample_mapping_path}")

            # Step 8: Save metadata copy to output dir
            metadata_output_path = ctx.deps.output_dir / "metadata" / "user_metadata.csv"
            metadata_df.to_csv(metadata_output_path, index=False)

            # Step 9: Populate state
            ctx.state.metadata_df = metadata_df
            ctx.state.metadata_path = str(metadata_output_path)
            ctx.state.organism = organism
            ctx.state.fastq_dir = self.fastq_dir
            ctx.state.dataset_information = self.description or "User-provided dataset"
            ctx.state.sample_mapping_path = str(sample_mapping_path)
            ctx.state.fastq_sample_mapping = mapping_dict  # Store mapping for Kallisto

            ctx.state.update_checkpoint("extraction", CheckpointStatus.COMPLETED)
            return MetadataAnalysisNode()

        except Exception as e:
            logger.error(f"User data extraction failed: {e}")
            ctx.state.update_checkpoint("extraction", CheckpointStatus.FAILED)
            ctx.state.analysis_history.append(
                {
                    "node": "UserDataExtractNode",
                    "error": str(e),
                    "checkpoint": "extraction",
                }
            )
            return ReflectNode(failed_checkpoint="extraction")

    def _discover_fastq_pairs_standard(
        self, fastq_dir: Path
    ) -> List[Tuple[str, Path, Path]]:
        """
        Discover paired FASTQ files using standard naming conventions.

        Handles two common naming conventions:
        1. Illumina: {prefix}_S*_L*_R1_*.fastq.gz and *_R2_*.fastq.gz
        2. Simple: {prefix}_1.fastq.gz and {prefix}_2.fastq.gz

        Returns:
            List of (prefix, read1_path, read2_path) tuples
        """
        pairs: List[Tuple[str, Path, Path]] = []

        # Pattern 1: Illumina style (*_R1_*.fastq.gz with S index)
        illumina_r1_pattern = re.compile(r"(.+?)_S\d+_L\d+_R1_\d+\.fastq\.gz$")
        for r1 in fastq_dir.glob("*_R1_*.fastq.gz"):
            match = illumina_r1_pattern.match(r1.name)
            if match:
                prefix = match.group(1)
                # Find matching R2
                r2_pattern = f"{prefix}_S*_L*_R2_*.fastq.gz"
                r2_files = list(fastq_dir.glob(r2_pattern))
                if r2_files:
                    pairs.append((prefix, r1, r2_files[0]))
                else:
                    logger.warning(f"Missing R2 for Illumina file: {r1.name}")

        # Pattern 2: Simple style (*_1.fastq.gz)
        if not pairs:  # Only try simple pattern if no Illumina files found
            for r1 in fastq_dir.glob("*_1.fastq.gz"):
                prefix = r1.name.replace("_1.fastq.gz", "")
                r2 = fastq_dir / f"{prefix}_2.fastq.gz"
                if r2.exists():
                    pairs.append((prefix, r1, r2))
                else:
                    logger.warning(f"Missing read 2 for {r1.name}")

        return pairs

    async def _discover_fastq_pairs_with_agent(
        self, fastq_dir: Path, deps: WorkflowDeps
    ) -> List[Tuple[str, Path, Path]]:
        """
        Use AI agent to discover paired FASTQ files when standard patterns don't match.

        Returns:
            List of (prefix, read1_path, read2_path) tuples
        """
        # Get all FASTQ files
        fastq_files = sorted([f.name for f in fastq_dir.glob("*.fastq.gz")])

        if not fastq_files:
            return []

        # Build prompt for agent
        file_list = "\n".join(f"- {f}" for f in fastq_files[:100])  # Limit to 100 files
        if len(fastq_files) > 100:
            file_list += f"\n... ({len(fastq_files) - 100} more files)"

        prompt = f"""Analyze these FASTQ files and identify read pairs (R1/R2).

FASTQ Files in directory:
{file_list}

Total files: {len(fastq_files)}

For each pair, provide:
1. sample_prefix: A meaningful sample identifier that can be matched to metadata
2. r1_file: The exact filename of the R1 file
3. r2_file: The exact filename of the matching R2 file

Important: The sample_prefix should capture the biological sample identity, not technical details like flowcell or lane.
"""

        logger.info("Using AI agent to discover FASTQ pairs (non-standard naming detected)")
        result = await fastq_pair_discovery_agent.run(prompt, deps=deps)

        logger.info(f"Agent detected pattern: {result.output.naming_pattern}")
        logger.info(f"Agent reasoning: {result.output.reasoning}")

        if result.output.unpaired_files:
            logger.warning(f"Unpaired files: {result.output.unpaired_files}")

        # Convert agent output to standard format
        pairs: List[Tuple[str, Path, Path]] = []
        for pair in result.output.pairs:
            r1_path = fastq_dir / pair.r1_file
            r2_path = fastq_dir / pair.r2_file

            # Validate files exist
            if not r1_path.exists():
                logger.warning(f"R1 file not found: {pair.r1_file}")
                continue
            if not r2_path.exists():
                logger.warning(f"R2 file not found: {pair.r2_file}")
                continue

            pairs.append((pair.sample_prefix, r1_path, r2_path))

        return pairs

    def _build_linking_prompt(
        self,
        fastq_prefixes: List[str],
        metadata_df: pd.DataFrame,
        description: Optional[str],
    ) -> str:
        """Build prompt for FASTQ-metadata linking."""
        # Send EVERY prefix — earlier truncation to 20 caused the linker
        # agent to refuse to commit to mappings whenever len(prefixes)>20,
        # because the trailing "...(N more)" marker was interpreted as
        # hidden data the agent couldn't verify against. Prefixes are short
        # strings; even hundreds are well within context.
        prefix_str = "\n".join(f"- {p}" for p in fastq_prefixes)

        # Send the full metadata for small datasets so the agent can see
        # every sample row when deciding the matching column. Truncate only
        # when the table would be unwieldy in the prompt (>50 rows).
        if len(metadata_df) <= 50:
            metadata_preview = metadata_df.to_string()
            metadata_label = f"Metadata (all {len(metadata_df)} rows)"
        else:
            metadata_preview = (
                metadata_df.head(50).to_string()
                + f"\n... ({len(metadata_df) - 50} more rows; same schema)"
            )
            metadata_label = f"Metadata (first 50 of {len(metadata_df)} rows)"

        return f"""Match these FASTQ files to metadata rows.

FASTQ File Prefixes (extracted from filenames):
{prefix_str}
Total: {len(fastq_prefixes)} FASTQ pairs (all listed above)

Metadata Columns: {list(metadata_df.columns)}

{metadata_label}:
{metadata_preview}

Dataset Description: {description or 'Not provided'}

Determine:
1. Which metadata column contains sample identifiers that match FASTQ prefixes?
2. Can each FASTQ file be matched to exactly one metadata row?
3. What organism is this data from (if determinable from metadata)?

Note: FASTQ prefixes may use dashes (-) while metadata may use underscores (_), or vice versa.
Every FASTQ prefix is listed above — produce mappings for ALL of them when the matching column is unambiguous.
"""

    def _handle_matching_failure(
        self,
        ctx: GraphRunContext[WorkflowState, WorkflowDeps],
        link_output: FASTQMetadataLinkResult,
    ) -> End[str]:
        """Handle case where FASTQ-metadata matching fails."""

        error_message = f"""
FASTQ-Metadata Matching Failed
==============================

{link_output.clarification_needed}

Suggestions:
{chr(10).join(f'- {s}' for s in link_output.suggestions)}

Reasoning: {link_output.reasoning}

Please update your metadata CSV or FASTQ file names and re-run.
"""
        logger.error(error_message)

        ctx.state.analysis_success = False
        ctx.state.analysis_diagnostics = error_message
        ctx.state.update_checkpoint("extraction", CheckpointStatus.FAILED)

        return End(error_message)

    def _resolve_organism(
        self,
        link_output: FASTQMetadataLinkResult,
    ) -> str:
        """Resolve organism from config, inference, or default."""

        # Priority 1: Provided in config
        if self.organism:
            logger.info(f"Using organism from config: {self.organism}")
            return self.organism

        # Priority 2: Inferred with high confidence
        if link_output.inferred_organism and link_output.organism_confidence:
            if link_output.organism_confidence >= 0.7:
                logger.info(
                    f"Using inferred organism: {link_output.inferred_organism} "
                    f"(confidence: {link_output.organism_confidence})"
                )
                return link_output.inferred_organism

        # Priority 3: Low confidence inference
        if link_output.inferred_organism:
            logger.warning(
                f"Low confidence organism inference: {link_output.inferred_organism} "
                f"(confidence: {link_output.organism_confidence}). "
                f"Consider specifying organism in config."
            )
            return link_output.inferred_organism

        # Priority 4: Default fallback
        logger.warning("Could not determine organism. Defaulting to 'Homo sapiens'.")
        return "Homo sapiens"
