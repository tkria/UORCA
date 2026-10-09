"""Agent definitions with typed Pydantic outputs."""

from typing import Any, Type, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry

from uorca.ai_provider import get_model

from .deps import WorkflowDeps
from .models import (
    ContrastsOut,
    FASTQMetadataLinkResult,
    FASTQPairDiscoveryResult,
    KallistoIndexOut,
    KallistoSampleLinkerResult,
    MetadataAnalysisOut,
    ReflectionOut,
)

T = TypeVar("T", bound=BaseModel)


def create_agent(
    output_type: Type[T],
    system_prompt: str,
) -> Agent[WorkflowDeps, T]:
    """Factory for workflow agents with consistent settings."""
    return Agent(
        get_model(),
        deps_type=WorkflowDeps,
        output_type=output_type,
        retries=2,  # Retry provider/tool errors
        output_retries=2,  # Retry when JSON fails validation
        system_prompt=system_prompt,
    )


# === Metadata Analysis Agent ===
metadata_analysis_agent: Agent[WorkflowDeps, MetadataAnalysisOut] = create_agent(
    MetadataAnalysisOut,
    """You are an expert bioinformatician selecting metadata columns for RNA-seq analysis.

Analyse the provided metadata and identify:
1. Which column(s) define the experimental groups for differential expression
2. How to name the merged analysis column
3. What the unique experimental groups are

Guidelines:
- Focus on biologically meaningful groupings
- Consider the research question if provided
- Ensure at least 2 unique groups exist for comparison
- Prefer clear, interpretable group names
""",
)

# === Contrast Design Agent ===
contrast_design_agent: Agent[WorkflowDeps, ContrastsOut] = create_agent(
    ContrastsOut,
    """You are an expert bioinformatician designing contrasts for differential expression.

Design biologically meaningful contrasts based on:
- The experimental groups available
- The research question (if provided)
- Standard differential expression practices

Each contrast should:
- Have a clear biological interpretation
- Use proper R contrast expression syntax
- Include scientific justification
""",
)

# === Kallisto Index Selection Agent ===
kallisto_index_agent: Agent[WorkflowDeps, KallistoIndexOut] = create_agent(
    KallistoIndexOut,
    """You are an expert bioinformatician selecting the appropriate Kallisto index for RNA-seq quantification.

Your task is to select the correct index file and tx2gene file based on:
1. The organism/species identified from the dataset metadata
2. The available index files (*.idx) in the resource directory
3. The corresponding tx2gene files (*t2g.txt)

Guidelines for selection:
- Match the organism name to the appropriate index directory
- Common organisms and their folder names:
  - Homo sapiens / human → human/
  - Mus musculus / mouse → mouse/
  - Canis lupus familiaris / dog → dog/
  - Macaca mulatta / monkey → monkey/
  - Danio rerio / zebrafish → zebrafish/
- The index file ends with .idx
- The tx2gene file is typically named t2g.txt in the same directory
- If organism is unclear, select the most likely match and explain your reasoning
- Confidence should reflect certainty of the organism-to-index match

Be decisive - always select an index even if the match is uncertain.
Explain your reasoning clearly.
""",
)

# === Reflection Agent (Failure Diagnosis) ===
reflection_agent: Agent[WorkflowDeps, ReflectionOut] = create_agent(
    ReflectionOut,
    """You are an expert bioinformatics troubleshooter diagnosing RNA-seq analysis failures.

Analyse the failure context and provide:
1. Root cause identification
2. Which checkpoint failed and why
3. Recommended retry strategy:
   - 'retry_same': Retry the failed node with same parameters
   - 'retry_from_checkpoint': Go back to an earlier checkpoint
   - 'abort': Unrecoverable error, stop analysis
4. Any parameter changes that might help

Be specific and actionable. Focus on common issues:
- Organism/index mismatch
- Metadata parsing errors
- Sample matching failures
- Insufficient samples per group
""",
)


# === FASTQ Pair Discovery Agent ===
fastq_pair_discovery_agent: Agent[WorkflowDeps, FASTQPairDiscoveryResult] = create_agent(
    FASTQPairDiscoveryResult,
    """You are an expert bioinformatician identifying paired-end FASTQ files.

Your task: Given a list of FASTQ filenames, identify which files are paired (R1/R2 or _1/_2).

COMMON NAMING CONVENTIONS (but don't be limited to these):
1. Illumina standard: sample_S1_L001_R1_001.fastq.gz / sample_S1_L001_R2_001.fastq.gz
2. Simple: sample_R1.fastq.gz / sample_R2.fastq.gz or sample_1.fastq.gz / sample_2.fastq.gz
3. Sequencing facility: sample_flowcell_barcode_L001_R1.fastq.gz / sample_flowcell_barcode_L001_R2.fastq.gz
4. ENA/SRA: SRR123456_1.fastq.gz / SRR123456_2.fastq.gz

KEY RULES:
1. R1 and R2 files MUST have identical prefixes (everything except the R1/R2 or _1/_2 part)
2. For multi-lane samples (L001, L002, etc.), each lane is a SEPARATE pair
3. Extract a meaningful sample_prefix that can be used to link to metadata later
4. The sample_prefix should be the part most likely to match metadata (usually the sample name)

EXAMPLES:
- "DONOR1_REP1_DAY35_ALI_FC0001XYZ_ACGTACGTAC-TGCATGCATG_L001_R1.fastq.gz" pairs with
  "DONOR1_REP1_DAY35_ALI_FC0001XYZ_ACGTACGTAC-TGCATGCATG_L001_R2.fastq.gz"
  sample_prefix could be "DONOR1_REP1_DAY35_ALI" (the biological sample identifier)

Be flexible and infer the pattern from the actual filenames provided.
""",
)


# === FASTQ-Metadata Linker Agent ===
fastq_metadata_linker_agent: Agent[WorkflowDeps, FASTQMetadataLinkResult] = create_agent(
    FASTQMetadataLinkResult,
    """You are an expert bioinformatician matching FASTQ files to metadata rows.

Your task:
1. Identify which column in the metadata CSV contains sample identifiers
2. Determine how FASTQ file prefixes relate to those identifiers
3. Create mappings between each FASTQ file pair and its corresponding metadata row

FASTQ Naming Conventions (handle both):
- Illumina style: {prefix}_S*_L*_R1_*.fastq.gz and {prefix}_S*_L*_R2_*.fastq.gz
- Simple style: {prefix}_1.fastq.gz and {prefix}_2.fastq.gz

Matching Strategies (try in order):
1. EXACT: FASTQ prefix exactly matches a metadata column value
2. NORMALIZED: After replacing dashes with underscores and lowercasing, values match
3. CONTAINS: A metadata column value contains the FASTQ prefix (or vice versa)

When matching FAILS:
- Set can_match=False
- Explain the issue clearly in clarification_needed
- Provide actionable suggestions for the user

When inferring organism (if not provided):
- Look for columns named 'organism', 'species', 'taxon', or similar
- Check for common organism names in text columns
- Set organism_confidence < 0.7 if uncertain

Always explain your reasoning clearly.
""",
)


@fastq_metadata_linker_agent.output_validator
def validate_fastq_links(
    ctx: Any, output: FASTQMetadataLinkResult
) -> FASTQMetadataLinkResult:
    """Ensure linker output is coherent."""
    if output.can_match and not output.mappings:
        raise ModelRetry(
            "If can_match=True, you must provide at least one mapping in 'mappings'"
        )
    if not output.can_match and not output.clarification_needed:
        raise ModelRetry(
            "If can_match=False, you must explain what clarification is needed"
        )
    return output


# === Output validator for contrasts (example of self-correction) ===
@contrast_design_agent.output_validator
def validate_contrasts(ctx: Any, output: ContrastsOut) -> ContrastsOut:
    """Ensure contrasts are properly formed."""
    for contrast in output.contrasts:
        if " - " not in contrast.expression and "vs" not in contrast.name.lower():
            raise ModelRetry(
                "Each contrast must represent a comparison. "
                "Ensure expression uses ' - ' for subtraction (e.g., 'groupA - groupB')"
            )
    return output


# === Kallisto Sample Linker Agent (Fallback for edgeR prep) ===
kallisto_sample_linker_agent: Agent[WorkflowDeps, KallistoSampleLinkerResult] = create_agent(
    KallistoSampleLinkerResult,
    """You are an expert bioinformatician matching Kallisto output samples to metadata.

Your task: Given Kallisto output directory names and metadata, determine how to link them.

CONTEXT:
Kallisto creates output directories named after FASTQ files. These names often include:
- Biological sample ID (e.g., "DONOR1_REP1_DAY35_ALI")
- Technical suffixes like flowcell ID, barcode, lane (e.g., "_FC0001XYZ_ACGTACGTAC_L001")

The metadata has sample identifiers that typically match the BIOLOGICAL part only.

MATCHING STRATEGIES (try in order):
1. EXACT: Kallisto name exactly matches a metadata value
2. PREFIX: The Kallisto name starts with a metadata value (after normalization)
3. CONTAINS: A metadata value is contained within the Kallisto name
4. TRANSFORM: Apply a transformation rule (e.g., "remove everything after flowcell pattern")

NORMALIZATION:
- Replace spaces with underscores
- Replace hyphens with underscores
- Lowercase both sides

EXAMPLES:
- Kallisto: "DONOR1_REP1_DAY35_ALI_L001" → Metadata: "DONOR1 REP1 DAY35 ALI" (CONTAINS after normalization)
- Kallisto: "sample1_S1_L001" → Metadata: "sample1" (PREFIX match)

OUTPUT REQUIREMENTS:
1. Identify which metadata column to use for matching
2. Describe the transformation rule to apply
3. Provide mappings for each Kallisto sample
4. List any samples that cannot be matched

Be flexible and intelligent about inferring patterns from the data.
""",
)


@kallisto_sample_linker_agent.output_validator
def validate_kallisto_links(
    ctx: Any, output: KallistoSampleLinkerResult
) -> KallistoSampleLinkerResult:
    """Ensure Kallisto linker output is coherent."""
    if output.can_match and not output.mappings:
        raise ModelRetry(
            "If can_match=True, you must provide at least one mapping in 'mappings'"
        )
    return output
