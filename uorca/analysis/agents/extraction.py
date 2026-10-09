"""
Data‑Extraction Agent
─────────────────────
* Tool 1  fetch_geo_metadata  – GEO → GSM ↔ SRX ↔ SRR long table (.csv)
* Tool 2  download_fastqs     – prefetch + fasterq‑dump into FASTQ.gz

Both tools populate ctx.deps so downstream agents (metadata / analysis)
see the files immediately.
"""
from __future__ import annotations
import asyncio, subprocess, re, pathlib, shutil, sys, os, logging
from typing import List

import pandas as pd
from dotenv import load_dotenv
import GEOparse as gp
from Bio import Entrez
from pydantic_ai import Agent, RunContext
from uorca.shared import ExtractionContext, RNAseqCoreContext, CheckpointStatus
from uorca.shared.workflow_logging import log_tool, log_agent_tool
from uorca.shared.entrez_utils import fetch_taxonomy_info, configure_entrez
from uorca.ai_provider import get_model
import datetime

logger = logging.getLogger(__name__)

# ── env + agent definition ──────────────────────────────────────────────────
load_dotenv()

_prompts_dir = pathlib.Path(__file__).parent.parent / "prompts"
extract_agent = Agent(
    get_model(),
    deps_type=RNAseqCoreContext,
    system_prompt=(
        (_prompts_dir / "extraction.txt").read_text()
        if (_prompts_dir / "extraction_system.txt").exists()
        else "You handle data extraction tasks."
    ),
    model_settings={"temperature": 1}
)

# ──────────────────────────────────────────────────────────────────────────
@extract_agent.tool
@log_tool
async def fetch_geo_metadata(ctx: RunContext[RNAseqCoreContext], accession: str) -> str:
    """Retrieve sample‑level metadata and SRR run IDs for a GEO series."""
    logger.info("fetch_geo_metadata() called for %s", accession)

    # Update checkpoint: metadata extraction started
    if hasattr(ctx.deps, 'checkpoints') and ctx.deps.checkpoints:
        cp = ctx.deps.checkpoints.metadata_extraction
        cp.status = CheckpointStatus.IN_PROGRESS
        cp.error_message = None
        cp.timestamp = datetime.datetime.now().isoformat()
        logger.info("Checkpoint: Metadata extraction started")

    out_root = pathlib.Path(ctx.deps.output_dir or ".").resolve()
    meta_dir = out_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving metadata under %s", meta_dir)

    # Try wget download first as a precautionary measure
    number_part = accession.replace('GSE', '')
    if len(number_part) > 3:
        base_part = number_part[:-3] + 'nnn'
    else:
        base_part = 'nnn'

    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{base_part}/{accession}/soft/{accession}_family.soft.gz"
    filename = f"{accession}_family.soft.gz"

    # Attempt wget download with timeout (but continue even if it fails)
    try:
        logger.info("Pre-downloading %s from %s", filename, url)
        wget_result = subprocess.run([
            "wget", "-O", str(meta_dir / filename), url
        ], capture_output=True, text=True, timeout=60)

        if wget_result.returncode == 0:
            logger.info("Successfully pre-downloaded %s via wget", filename)
        else:
            logger.warning("wget failed for %s (return code %d), proceeding anyway: %s",
                         accession, wget_result.returncode, wget_result.stderr)

    except subprocess.TimeoutExpired:
        logger.warning("wget timed out after 60 seconds for %s, proceeding anyway", accession)
    except Exception as wget_e:
        logger.warning("wget subprocess failed for %s, proceeding anyway: %s", accession, str(wget_e))

    # Now attempt regular GEO parsing
    try:
        gse = gp.get_GEO(accession, destdir=str(meta_dir), silent=True)
        logger.info("Successfully fetched GEO metadata for %s", accession)
    except Exception as e:
        logger.error("Failed to fetch GEO metadata for %s: %s", accession, str(e))
        raise Exception(f"GEO metadata fetch failed for {accession}: {str(e)}")
    gsms = gse.gsms
    logger.info("Fetched %d GSM samples", len(gsms))

    # Capture dataset information
    dataset_info_parts = []

    # Get title
    if 'title' in gse.metadata and gse.metadata['title']:
        title = gse.metadata['title'][0] if isinstance(gse.metadata['title'], list) else gse.metadata['title']
        dataset_info_parts.append(f"Title: {title}")

    # Get summary
    if 'summary' in gse.metadata and gse.metadata['summary']:
        summary = gse.metadata['summary'][0] if isinstance(gse.metadata['summary'], list) else gse.metadata['summary']
        dataset_info_parts.append(f"Summary: {summary}")

    # Get overall design
    if 'overall_design' in gse.metadata and gse.metadata['overall_design']:
        design = gse.metadata['overall_design'][0] if isinstance(gse.metadata['overall_design'], list) else gse.metadata['overall_design']
        dataset_info_parts.append(f"Overall Design: {design}")

    # Combine all information
    dataset_information = "\n\n".join(dataset_info_parts)

    # Store in context
    ctx.deps.dataset_information = dataset_information
    logger.info("Captured dataset information from GEO metadata")

    logger.info("Dataset information captured in context for downstream use")

    # Extract species information from taxonomic ID
    species_name = "Unknown"
    try:
        if 'sample_taxid' in gse.metadata and gse.metadata['sample_taxid']:
            taxid = gse.metadata['sample_taxid'][0] if isinstance(gse.metadata['sample_taxid'], list) else gse.metadata['sample_taxid']
            logger.info("Found taxonomic ID: %s", taxid)

            # Convert taxonomic ID to species name using rate-limited Entrez
            taxonomy_info = fetch_taxonomy_info(str(taxid))

            if taxonomy_info:
                species_name = taxonomy_info['scientific_name']
                logger.info("Identified species: %s", species_name)
                if taxonomy_info.get('common_name'):
                    logger.info("Common name: %s", taxonomy_info['common_name'])
            else:
                logger.warning("No species information found for taxonomic ID: %s", taxid)
        else:
            logger.warning("No taxonomic ID found in GEO metadata")
    except Exception as e:
        logger.warning("Error extracting species information: %s", str(e))

    # Store species in context
    ctx.deps.organism = species_name
    logger.info("Set organism to: %s", species_name)

    re_srx, re_srr = re.compile(r"(SR[XP]\d+)"), re.compile(r"(SRR\d+)")

    def srx_from_rel(rel: List[str] | None):
        if not rel:
            return None
        for line in rel:
            hit = re_srx.search(line)
            if hit:
                return hit.group(1)
        return None

    # ── wide sample table (now uses GEOparse phenotype_data) ────────────────
    meta_wide = (
        gse.phenotype_data
          .reset_index()              # GSM ID becomes its own column
          .rename(columns={"index": "GSM"})
    )
    logger.info("meta_wide.csv prepared (%d rows, %d cols)", len(meta_wide), meta_wide.shape[1])


    # ── long GSM↔SRX↔SRR table ──────────────────────────────────────────────
    def srrs_from_srx(srx: str):
        cmd = (
            f"esearch -db sra -query {srx} | "
            "efetch -format runinfo | cut -d',' -f1 | grep ^SRR"
        )
        try:
            out = subprocess.run(
                cmd, shell=True, check=True,
                stdout=subprocess.PIPE, text=True
            ).stdout.splitlines()
            return [s for s in out if re_srr.match(s)]
        except subprocess.CalledProcessError:
            logger.warning("Entrez Direct failed for %s – skipping", srx)
            return []

    long_rows = []
    for gsm, g in gsms.items():
        srx = srx_from_rel(g.metadata.get("relation"))
        if not srx:
            continue
        for srr in srrs_from_srx(srx):
            long_rows.append({"GSM": gsm, "SRX": srx, "SRR": srr})

    meta_long = pd.DataFrame(long_rows)
    logger.info("Constructed long table: %d SRR rows", len(meta_long))

    # ── merge & save ─────────────────────────────────────────────────────────
    out_df = meta_long.merge(
        meta_wide,
        on="GSM",
        how="left",
        validate="many_to_one"
    )

    metadata_filename = f"{accession}_metadata.csv"
    out_df.to_csv(meta_dir / metadata_filename, index=False)

    logger.info("%s written (%d rows)", metadata_filename, len(out_df))

    # Filter for RNA-seq specific samples
    logger.info("Filtering for RNA-seq specific samples")
    initial_rows = len(out_df)

    # Apply RNA-seq filters - exact matches required
    rna_seq_filter = (
        (out_df['library_source'] == "transcriptomic") &
        (out_df['library_strategy'] == "RNA-Seq")
    )
    out_df = out_df[rna_seq_filter]

    logger.info("RNA-seq filtering: %d -> %d rows", initial_rows, len(out_df))

    # Check if any samples remain after filtering
    if len(out_df) == 0:
        error_msg = (
            f"No RNA-seq samples found after filtering. Dataset {accession} contains no samples with library_source='transcriptomic', and library_strategy='RNA-Seq'. "
            f"Dataset will be terminated."
        )
        logger.error("%s", error_msg)

        # Mark in context that analysis should not proceed
        ctx.deps.analysis_should_proceed = False
        ctx.deps.analysis_skip_reason = error_msg
        ctx.deps.analysis_success = False

        # Update checkpoint: metadata extraction failed
        if hasattr(ctx.deps, 'checkpoints') and ctx.deps.checkpoints:
            cp = ctx.deps.checkpoints.metadata_extraction
            cp.status = CheckpointStatus.FAILED
            cp.error_message = error_msg
            cp.timestamp = datetime.datetime.now().isoformat()

        return error_msg

    # Check organism consistency
    unique_organisms = out_df['organism_ch1'].nunique()
    organisms_list = out_df['organism_ch1'].unique().tolist()
    logger.info("Found %d unique organisms: %s", unique_organisms, organisms_list)

    if unique_organisms > 1:
        error_msg = (
            f"Multiple organisms detected in dataset {accession}: {organisms_list}. "
            f"Analysis requires samples from a single organism. Dataset will be terminated."
        )
        logger.error("%s", error_msg)

        # Mark in context that analysis should not proceed
        ctx.deps.analysis_should_proceed = False
        ctx.deps.analysis_skip_reason = error_msg
        ctx.deps.analysis_success = False

        # Update checkpoint: metadata extraction failed
        if hasattr(ctx.deps, 'checkpoints') and ctx.deps.checkpoints:
            cp = ctx.deps.checkpoints.metadata_extraction
            cp.status = CheckpointStatus.FAILED
            cp.error_message = error_msg
            cp.timestamp = datetime.datetime.now().isoformat()

        return error_msg

    # Validate sample count - need more than 2 samples for meaningful analysis
    unique_samples = out_df['GSM'].nunique()
    logger.info("Found %d unique samples (GSM entries)", unique_samples)

    if unique_samples <= 2:
        error_msg = (
            f"Insufficient samples for analysis: only {unique_samples} unique "
            f"samples found. Need at least 3 samples for meaningful differential "
            f"expression analysis. Dataset {accession} will be terminated."
        )
        logger.error("%s", error_msg)

        # Mark in context that analysis should not proceed
        ctx.deps.analysis_should_proceed = False
        ctx.deps.analysis_skip_reason = error_msg
        ctx.deps.analysis_success = False

        # Update checkpoint to reflect termination but don't raise
        if hasattr(ctx.deps, 'checkpoints') and ctx.deps.checkpoints:
            cp = ctx.deps.checkpoints.metadata_extraction
            cp.status = CheckpointStatus.FAILED
            cp.error_message = error_msg
            cp.timestamp = datetime.datetime.now().isoformat()

        # Return early with informative message
        return error_msg

    # expose merged table to downstream agents
    ctx.deps.metadata_df = out_df
    ctx.deps.metadata_path = str(meta_dir / metadata_filename)

    # Clean up temporary files created by GEOparse
    logger.info("Cleaning up temporary files")
    cleanup_count = 0
    for temp_file in meta_dir.glob("*.soft.gz"):
        try:
            temp_file.unlink()
            cleanup_count += 1
            logger.debug(f"Removed temporary file: {temp_file}")
        except Exception as e:
            logger.warning(f"Could not remove temporary file {temp_file}: {e}")

    if cleanup_count > 0:
        logger.info(f"Cleaned up {cleanup_count} temporary files")

    # Update checkpoint: metadata extraction completed successfully
    if hasattr(ctx.deps, 'checkpoints') and ctx.deps.checkpoints:
        cp = ctx.deps.checkpoints.metadata_extraction
        cp.status = CheckpointStatus.COMPLETED
        cp.details = f"Extracted metadata for {unique_samples} unique samples"
        cp.error_message = None
        cp.timestamp = datetime.datetime.now().isoformat()
        logger.info("Checkpoint: Metadata extraction completed successfully")

    return (
        f"Fetched {len(gsms)} GSM samples ({unique_samples} unique).\n"
        f"Identified {len(meta_long)} SRR runs "
        f"across {meta_long['SRX'].nunique()} SRX experiments."
    )

# ─────────────────────────────────────────────────────────────────────────────
@extract_agent.tool
@log_tool
async def download_fastqs(
    ctx: RunContext[RNAseqCoreContext],
    threads: int = 6,
    max_spots: int | None = None
) -> str:
    """Convert SRR accessions → **paired FASTQ.gz** using SRA‑Toolkit."""

    logger.info("📥 download_fastqs() called – %d threads, max_spots=%s",
                threads, max_spots)

    # Update checkpoint: FASTQ extraction started
    if hasattr(ctx.deps, 'checkpoints') and ctx.deps.checkpoints:
        cp = ctx.deps.checkpoints.fastq_extraction
        cp.status = CheckpointStatus.IN_PROGRESS
        cp.error_message = None
        cp.timestamp = datetime.datetime.now().isoformat()
        logger.info("📥 Checkpoint: FASTQ extraction started")

    if ctx.deps.metadata_df is None or "SRR" not in ctx.deps.metadata_df.columns:
        raise ValueError("metadata_df with SRR column required. Run fetch_geo_metadata first.")

    # Determine available CPUs from SLURM or system
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    threads = max(1, min(slurm_cpus - 2, threads))  # Use provided threads but cap at available CPUs minus 2
    logger.info("⚙️ Using %d threads based on system resources (SLURM_CPUS_PER_TASK=%s)",
                threads, os.environ.get("SLURM_CPUS_PER_TASK", "not set"))

    out_root = pathlib.Path(ctx.deps.output_dir or ".").resolve()
    prefetch_dir = out_root / "sra"
    fastq_dir = out_root / "fastq"
    for d in (prefetch_dir, fastq_dir):
        d.mkdir(parents=True, exist_ok=True)

    def sra_path(srr: str) -> pathlib.Path:
        # First try the standard .sra extension
        p = prefetch_dir / srr / f"{srr}.sra"
        if p.exists():
            return p

        # Next try .sralite extension
        p_lite = prefetch_dir / srr / f"{srr}.sralite"
        if p_lite.exists():
            return p_lite

        # If neither exists at standard location, do recursive search for both
        hits = list(prefetch_dir.rglob(f"{srr}.sra")) + list(prefetch_dir.rglob(f"{srr}.sralite"))
        return hits[0] if hits else p  # Return first hit or original path if nothing found


    def fastq_ready(srr: str):
        # Check if the compressed FASTQ file exists
        # Using "{srr}*.fastq.gz" to match any variation
        fastq_files = list(fastq_dir.glob(f"{srr}*.fastq.gz"))
        return len(fastq_files) > 0 and all(f.stat().st_size > 0 for f in fastq_files)


    srrs = ctx.deps.metadata_df["SRR"].dropna().astype(str).tolist()
    logger.info("Total SRRs listed: %d", len(srrs))

    # First check which SRRs already have FASTQ files - no need to download SRA for these
    ready_count = sum(1 for srr in srrs if fastq_ready(srr))
    logger.info("SRRs with ready FASTQ files: %d", ready_count)

    # Only prefetch SRAs for SRRs that don't have FASTQ files and don't have SRA files
    srrs_needing_fastq = [s for s in srrs if not fastq_ready(s)]
    need_prefetch = [s for s in srrs_needing_fastq if not sra_path(s).exists()]
    logger.info("SRRs needing SRA prefetch (no FASTQ, no SRA): %d", len(need_prefetch))

    # Helper function for prefetching a single SRR
    async def prefetch_single(srr: str):
        cmd = f"prefetch {srr} -O {prefetch_dir} -t https"
        logger.debug("▶ Running command: %s", cmd)
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("Prefetch failed for %s: %s", srr, stderr.decode().strip())
        elif stderr:
            logger.debug("Prefetch stderr for %s: %s", srr, stderr.decode().strip())
        return {
            "srr": srr,
            "returncode": proc.returncode,
            "stdout": stdout.decode() if stdout else "",
            "stderr": stderr.decode() if stderr else ""
        }

    if need_prefetch:
        # Use parallel prefetch for all cases to avoid creating temporary files
        if len(need_prefetch) > 0:
            logger.info("🔄 Prefetching %d SRRs in parallel batches", len(need_prefetch))
            # Create batches of SRRs to avoid overwhelming the system
            batch_size = min(10, max(1, len(need_prefetch) // 2))
            batches = [need_prefetch[i:i+batch_size] for i in range(0, len(need_prefetch), batch_size)]

            successful_prefetch = 0
            for batch_num, batch in enumerate(batches, 1):
                logger.info("🔄 Processing prefetch batch %d/%d with %d SRRs",
                            batch_num, len(batches), len(batch))
                tasks = [prefetch_single(srr) for srr in batch]
                results = await asyncio.gather(*tasks)

                # Log results
                for res in results:
                    if res["returncode"] == 0:
                        successful_prefetch += 1
                        logger.info("Successfully prefetched %s", res["srr"])
                    else:
                        logger.error("Failed to prefetch %s", res["srr"])

            logger.info("Prefetch completed - %d/%d successful", successful_prefetch, len(need_prefetch))

    # Check which SRRs need conversion to FASTQ (have SRA but no FASTQ)
    need_conversion = [srr for srr in srrs_needing_fastq if sra_path(srr).exists()]
    logger.info("SRRs needing conversion to FASTQ (have SRA, no FASTQ): %d", len(need_conversion))

    # Log SRRs that have neither FASTQ nor SRA files after prefetch attempt
    missing_both = [srr for srr in srrs_needing_fastq if not sra_path(srr).exists()]
    if missing_both:
        logger.warning("SRRs missing both FASTQ and SRA files after prefetch: %d (%s)",
                      len(missing_both), ", ".join(missing_both[:5]) + ("..." if len(missing_both) > 5 else ""))

    # Find pigz for faster compression
    pigz = shutil.which("pigz")
    if pigz:
        logger.info("Found pigz for faster parallel compression")
    else:
        logger.info("pigz not found, falling back to standard gzip")

    # Process conversions (fasterq-dump + compression)
    converted = 0
    for srr in need_conversion:
        sra = sra_path(srr)
        if not sra.exists():
            logger.warning("%s: .sra missing after prefetch – skip.", srr)
            continue

        # Calculate threads for this conversion based on total available
        conversion_threads = max(1, min(threads, 16))  # Cap at 16 threads per conversion

        # Run fasterq-dump with proper logging
        cmd = [
            "fasterq-dump", str(sra),
            "--threads", str(conversion_threads),
            "--split-files",
            "-v",
            "-O", str(fastq_dir)
        ]
        if max_spots:
            cmd += ["-X", str(max_spots)]

        logger.info("▶ Running fasterq-dump on %s with %d threads (SRA size: %.1f MB)",
                   srr, conversion_threads, sra.stat().st_size / (1024*1024))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        # Log the stdout/stderr for debugging
        if stdout:
            logger.debug("Fasterq-dump stdout for %s: %s", srr, stdout.decode().strip())
        if stderr:
            logger.info("Fasterq-dump stderr for %s: %s", srr, stderr.decode().strip())

        if proc.returncode != 0:
            logger.warning("fasterq-dump failed on %s with code %d. Check if SRA file is valid.", srr, proc.returncode)
            continue

        # Compress the FASTQ files one by one (simpler approach)
        fastqs = list(fastq_dir.glob(f"{srr}*.fastq"))
        if fastqs:
            logger.info("🔄 Compressing %d FASTQ files for %s", len(fastqs), srr)
            for fq in fastqs:
                if pigz:
                    # Use pigz with multiple threads if available
                    threads_per_file = max(1, slurm_cpus // 2)  # Use half of available CPUs for compression
                    gz_cmd = [pigz, "-fv", "-p", str(threads_per_file), str(fq)]
                else:
                    # Fall back to standard gzip
                    gz_cmd = ["gzip", "-fv", str(fq)]

                logger.info("▶ Running compression: %s", " ".join(gz_cmd))
                proc = await asyncio.create_subprocess_exec(
                    *gz_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode == 0:
                    logger.info("Compressed %s successfully", fq.name)
                else:
                    logger.warning("Compression failed for %s: %s", fq, stderr.decode().strip())

            # After successful compression, clean up the SRA file to save space
            if fastq_ready(srr):
                try:
                    sra_size_mb = sra.stat().st_size / (1024*1024)
                    sra.unlink()
                    logger.info("🗑️  Cleaned up SRA file: %s (freed %.1f MB)", sra.name, sra_size_mb)
                except Exception as e:
                    logger.warning("Failed to remove SRA file %s: %s", sra, e)
            else:
                logger.warning("FASTQ files not ready after compression for %s, keeping SRA file", srr)

        converted += 1
        logger.info("Completed processing for %s (%d/%d)",
                    srr, converted, len(need_conversion))

    # Update context for downstream agents
    ctx.deps.fastq_dir = str(fastq_dir)
    logger.info("FASTQ conversion finished: %d new SRRs converted", converted)

    # Update checkpoint: FASTQ extraction completed successfully
    if hasattr(ctx.deps, 'checkpoints') and ctx.deps.checkpoints:
        cp = ctx.deps.checkpoints.fastq_extraction
        cp.status = CheckpointStatus.COMPLETED
        cp.details = f"Downloaded and converted {converted} SRRs to FASTQ files"
        cp.error_message = None
        cp.timestamp = datetime.datetime.now().isoformat()
        logger.info("Checkpoint: FASTQ extraction completed successfully")

    return (
        "FASTQ download complete.\n"
        f"Total SRRs: {len(srrs)}   Newly converted: {converted}\n"
        f"FASTQ files saved to: {fastq_dir}"
    )


# ─────────────────────────────────────────────────────────────────────────────
@log_agent_tool
async def run_agent_async(prompt: str, deps: ExtractionContext, usage=None):
    """Thin wrapper used by master.py."""
    logger.info("🛠️ Extraction agent invoked by master – prompt: %s", prompt)
    result = await extract_agent.run(prompt, deps=deps, usage=usage)

    # Log the agent's output
    logger.info("📄 Extraction agent output: %s", result.output)

    # If you want to log usage statistics
    if hasattr(result, 'usage') and result.usage:
        try:
            usage_stats = result.usage()
            logger.info("Extraction agent usage: %s", usage_stats)
        except Exception as e:
            logger.debug("Could not get usage stats: %s", e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Core functions for pydantic-graph integration
# These are independent of RunContext and can be called from graph nodes
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import httpx


@dataclass
class GEOMetadataResult:
    """Result from GEO metadata fetch."""
    metadata_df: pd.DataFrame
    metadata_path: str
    organism: str
    dataset_info: str


@dataclass
class FASTQDownloadResult:
    """Result from FASTQ download."""
    fastq_dir: str


async def fetch_geo_metadata_core(
    accession: str,
    output_dir: Path,
    entrez_email: str,
    entrez_api_key: Optional[str] = None,
) -> GEOMetadataResult:
    """
    Fetch GEO metadata - core logic extracted for graph node use.

    Args:
        accession: GEO accession (e.g., "GSE12345")
        output_dir: Directory for output files
        entrez_email: Email for NCBI API
        entrez_api_key: Optional NCBI API key

    Returns:
        GEOMetadataResult with metadata DataFrame, path, organism, and dataset info
    """
    logger.info("fetch_geo_metadata_core() called for %s", accession)

    # Set environment variables from parameters (if provided) for configure_entrez()
    if entrez_email:
        os.environ["ENTREZ_EMAIL"] = entrez_email
    if entrez_api_key:
        os.environ["ENTREZ_API_KEY"] = entrez_api_key

    # Configure Entrez (reads from environment variables)
    configure_entrez()

    out_root = Path(output_dir).resolve()
    meta_dir = out_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving metadata under %s", meta_dir)

    # Try wget download first as a precautionary measure
    number_part = accession.replace('GSE', '')
    if len(number_part) > 3:
        base_part = number_part[:-3] + 'nnn'
    else:
        base_part = 'nnn'

    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{base_part}/{accession}/soft/{accession}_family.soft.gz"
    filename = f"{accession}_family.soft.gz"

    # Attempt wget download with timeout (but continue even if it fails)
    try:
        logger.info("Pre-downloading %s from %s", filename, url)
        wget_result = subprocess.run([
            "wget", "-O", str(meta_dir / filename), url
        ], capture_output=True, text=True, timeout=60)

        if wget_result.returncode == 0:
            logger.info("Successfully pre-downloaded %s via wget", filename)
        else:
            logger.warning("wget failed for %s (return code %d), proceeding anyway",
                         accession, wget_result.returncode)

    except subprocess.TimeoutExpired:
        logger.warning("wget timed out after 60 seconds for %s, proceeding anyway", accession)
    except Exception as wget_e:
        logger.warning("wget subprocess failed for %s, proceeding anyway: %s", accession, str(wget_e))

    # Now attempt regular GEO parsing
    try:
        gse = gp.get_GEO(accession, destdir=str(meta_dir), silent=True)
        logger.info("Successfully fetched GEO metadata for %s", accession)
    except Exception as e:
        logger.error("Failed to fetch GEO metadata for %s: %s", accession, str(e))
        raise Exception(f"GEO metadata fetch failed for {accession}: {str(e)}")

    gsms = gse.gsms
    logger.info("Fetched %d GSM samples", len(gsms))

    # Capture dataset information
    dataset_info_parts = []

    if 'title' in gse.metadata and gse.metadata['title']:
        title = gse.metadata['title'][0] if isinstance(gse.metadata['title'], list) else gse.metadata['title']
        dataset_info_parts.append(f"Title: {title}")

    if 'summary' in gse.metadata and gse.metadata['summary']:
        summary = gse.metadata['summary'][0] if isinstance(gse.metadata['summary'], list) else gse.metadata['summary']
        dataset_info_parts.append(f"Summary: {summary}")

    if 'overall_design' in gse.metadata and gse.metadata['overall_design']:
        design = gse.metadata['overall_design'][0] if isinstance(gse.metadata['overall_design'], list) else gse.metadata['overall_design']
        dataset_info_parts.append(f"Overall Design: {design}")

    dataset_information = "\n\n".join(dataset_info_parts)
    logger.info("Captured dataset information from GEO metadata")

    # Extract species information from taxonomic ID
    species_name = "Unknown"
    try:
        if 'sample_taxid' in gse.metadata and gse.metadata['sample_taxid']:
            taxid = gse.metadata['sample_taxid'][0] if isinstance(gse.metadata['sample_taxid'], list) else gse.metadata['sample_taxid']
            logger.info("Found taxonomic ID: %s", taxid)

            taxonomy_info = fetch_taxonomy_info(str(taxid))

            if taxonomy_info:
                species_name = taxonomy_info['scientific_name']
                logger.info("Identified species: %s", species_name)
            else:
                logger.warning("No species information found for taxonomic ID: %s", taxid)
        else:
            logger.warning("No taxonomic ID found in GEO metadata")
    except Exception as e:
        logger.warning("Error extracting species information: %s", str(e))

    re_srx, re_srr = re.compile(r"(SR[XP]\d+)"), re.compile(r"(SRR\d+)")

    def srx_from_rel(rel: List[str] | None):
        if not rel:
            return None
        for line in rel:
            hit = re_srx.search(line)
            if hit:
                return hit.group(1)
        return None

    # Wide sample table (uses GEOparse phenotype_data)
    meta_wide = (
        gse.phenotype_data
          .reset_index()
          .rename(columns={"index": "GSM"})
    )
    logger.info("meta_wide.csv prepared (%d rows, %d cols)", len(meta_wide), meta_wide.shape[1])

    # Long GSM↔SRX↔SRR table
    def srrs_from_srx(srx: str):
        cmd = (
            f"esearch -db sra -query {srx} | "
            "efetch -format runinfo | cut -d',' -f1 | grep ^SRR"
        )
        try:
            out = subprocess.run(
                cmd, shell=True, check=True,
                stdout=subprocess.PIPE, text=True
            ).stdout.splitlines()
            return [s for s in out if re_srr.match(s)]
        except subprocess.CalledProcessError:
            logger.warning("Entrez Direct failed for %s – skipping", srx)
            return []

    long_rows = []
    for gsm, g in gsms.items():
        srx = srx_from_rel(g.metadata.get("relation"))
        if not srx:
            continue
        for srr in srrs_from_srx(srx):
            long_rows.append({"GSM": gsm, "SRX": srx, "SRR": srr})

    meta_long = pd.DataFrame(long_rows)
    logger.info("Constructed long table: %d SRR rows", len(meta_long))

    # Merge & save
    out_df = meta_long.merge(
        meta_wide,
        on="GSM",
        how="left",
        validate="many_to_one"
    )

    metadata_filename = f"{accession}_metadata.csv"
    out_df.to_csv(meta_dir / metadata_filename, index=False)
    logger.info("%s written (%d rows)", metadata_filename, len(out_df))

    # Filter for RNA-seq specific samples
    logger.info("Filtering for RNA-seq specific samples")
    initial_rows = len(out_df)

    rna_seq_filter = (
        (out_df['library_source'] == "transcriptomic") &
        (out_df['library_strategy'] == "RNA-Seq")
    )
    out_df = out_df[rna_seq_filter]
    logger.info("RNA-seq filtering: %d -> %d rows", initial_rows, len(out_df))

    if len(out_df) == 0:
        raise ValueError(
            f"No RNA-seq samples found after filtering. Dataset {accession} contains no samples "
            f"with library_source='transcriptomic' and library_strategy='RNA-Seq'."
        )

    # Check organism consistency
    unique_organisms = out_df['organism_ch1'].nunique()
    organisms_list = out_df['organism_ch1'].unique().tolist()
    logger.info("Found %d unique organisms: %s", unique_organisms, organisms_list)

    if unique_organisms > 1:
        raise ValueError(
            f"Multiple organisms detected in dataset {accession}: {organisms_list}. "
            f"Analysis requires samples from a single organism."
        )

    # Validate sample count
    unique_samples = out_df['GSM'].nunique()
    logger.info("Found %d unique samples (GSM entries)", unique_samples)

    if unique_samples <= 2:
        raise ValueError(
            f"Insufficient samples for analysis: only {unique_samples} unique samples found. "
            f"Need at least 3 samples for meaningful differential expression analysis."
        )

    # Clean up temporary files
    logger.info("Cleaning up temporary files")
    for temp_file in meta_dir.glob("*.soft.gz"):
        try:
            temp_file.unlink()
        except Exception as e:
            logger.warning(f"Could not remove temporary file {temp_file}: {e}")

    return GEOMetadataResult(
        metadata_df=out_df,
        metadata_path=str(meta_dir / metadata_filename),
        organism=species_name,
        dataset_info=dataset_information,
    )


async def download_fastqs_core(
    metadata_df: pd.DataFrame,
    output_dir: Path,
    http_client: httpx.AsyncClient,
    threads: int = 6,
    max_spots: int | None = None,
) -> FASTQDownloadResult:
    """
    Download FASTQ files - core logic extracted for graph node use.

    Args:
        metadata_df: DataFrame with SRR column
        output_dir: Directory for output files
        http_client: Async HTTP client (unused but kept for interface compatibility)
        threads: Number of threads for fasterq-dump
        max_spots: Maximum spots to download (for testing)

    Returns:
        FASTQDownloadResult with path to FASTQ directory
    """
    logger.info("download_fastqs_core() called – %d threads, max_spots=%s", threads, max_spots)

    if metadata_df is None or "SRR" not in metadata_df.columns:
        raise ValueError("metadata_df with SRR column required.")

    # Determine available CPUs
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    threads = max(1, min(slurm_cpus - 2, threads))
    logger.info("Using %d threads based on system resources", threads)

    out_root = Path(output_dir).resolve()
    prefetch_dir = out_root / "sra"
    fastq_dir = out_root / "fastq"
    for d in (prefetch_dir, fastq_dir):
        d.mkdir(parents=True, exist_ok=True)

    def sra_path(srr: str) -> pathlib.Path:
        p = prefetch_dir / srr / f"{srr}.sra"
        if p.exists():
            return p
        p_lite = prefetch_dir / srr / f"{srr}.sralite"
        if p_lite.exists():
            return p_lite
        hits = list(prefetch_dir.rglob(f"{srr}.sra")) + list(prefetch_dir.rglob(f"{srr}.sralite"))
        return hits[0] if hits else p

    def fastq_ready(srr: str):
        fastq_files = list(fastq_dir.glob(f"{srr}*.fastq.gz"))
        return len(fastq_files) > 0 and all(f.stat().st_size > 0 for f in fastq_files)

    srrs = metadata_df["SRR"].dropna().astype(str).tolist()
    logger.info("Total SRRs listed: %d", len(srrs))

    ready_count = sum(1 for srr in srrs if fastq_ready(srr))
    logger.info("SRRs with ready FASTQ files: %d", ready_count)

    srrs_needing_fastq = [s for s in srrs if not fastq_ready(s)]
    need_prefetch = [s for s in srrs_needing_fastq if not sra_path(s).exists()]
    logger.info("SRRs needing SRA prefetch: %d", len(need_prefetch))

    async def prefetch_single(srr: str):
        cmd = f"prefetch {srr} -O {prefetch_dir} -t https"
        logger.debug("Running command: %s", cmd)
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("Prefetch failed for %s: %s", srr, stderr.decode().strip())
        return {
            "srr": srr,
            "returncode": proc.returncode,
            "stdout": stdout.decode() if stdout else "",
            "stderr": stderr.decode() if stderr else ""
        }

    if need_prefetch:
        logger.info("Prefetching %d SRRs in parallel batches", len(need_prefetch))
        batch_size = min(10, max(1, len(need_prefetch) // 2))
        batches = [need_prefetch[i:i+batch_size] for i in range(0, len(need_prefetch), batch_size)]

        successful_prefetch = 0
        for batch_num, batch in enumerate(batches, 1):
            logger.info("Processing prefetch batch %d/%d with %d SRRs", batch_num, len(batches), len(batch))
            tasks = [prefetch_single(srr) for srr in batch]
            results = await asyncio.gather(*tasks)

            for res in results:
                if res["returncode"] == 0:
                    successful_prefetch += 1
                    logger.info("Successfully prefetched %s", res["srr"])

        logger.info("Prefetch completed - %d/%d successful", successful_prefetch, len(need_prefetch))

    need_conversion = [srr for srr in srrs_needing_fastq if sra_path(srr).exists()]
    logger.info("SRRs needing conversion to FASTQ: %d", len(need_conversion))

    pigz = shutil.which("pigz")

    converted = 0
    for srr in need_conversion:
        sra = sra_path(srr)
        if not sra.exists():
            logger.warning("%s: .sra missing after prefetch – skip.", srr)
            continue

        conversion_threads = max(1, min(threads, 16))

        cmd = [
            "fasterq-dump", str(sra),
            "--threads", str(conversion_threads),
            "--split-files",
            "-v",
            "-O", str(fastq_dir)
        ]
        if max_spots:
            cmd += ["-X", str(max_spots)]

        logger.info("Running fasterq-dump on %s with %d threads", srr, conversion_threads)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.warning("fasterq-dump failed on %s with code %d", srr, proc.returncode)
            continue

        # Compress the FASTQ files
        fastqs = list(fastq_dir.glob(f"{srr}*.fastq"))
        if fastqs:
            logger.info("Compressing %d FASTQ files for %s", len(fastqs), srr)
            for fq in fastqs:
                if pigz:
                    threads_per_file = max(1, slurm_cpus // 2)
                    gz_cmd = [pigz, "-fv", "-p", str(threads_per_file), str(fq)]
                else:
                    gz_cmd = ["gzip", "-fv", str(fq)]

                proc = await asyncio.create_subprocess_exec(
                    *gz_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate()

            # Clean up SRA file after successful compression
            if fastq_ready(srr):
                try:
                    sra.unlink()
                    logger.info("Cleaned up SRA file: %s", sra.name)
                except Exception as e:
                    logger.warning("Failed to remove SRA file %s: %s", sra, e)

        converted += 1
        logger.info("Completed processing for %s (%d/%d)", srr, converted, len(need_conversion))

    logger.info("FASTQ conversion finished: %d new SRRs converted", converted)

    # Completeness gate: every expected SRR must have non-empty FASTQ output.
    # Without this, a transient download failure (e.g. SRA-Toolkit curl 56) for
    # a subset of samples silently shrinks the dataset and only surfaces much
    # later as a cryptic edgeR error when a whole experimental group is lost.
    missing = [srr for srr in srrs if not fastq_ready(srr)]
    if missing:
        raise RuntimeError(
            f"FASTQ download incomplete: {len(srrs) - len(missing)}/{len(srrs)} "
            f"SRRs ready, {len(missing)} missing: {', '.join(missing[:10])}"
            f"{' ...' if len(missing) > 10 else ''}"
        )

    return FASTQDownloadResult(fastq_dir=str(fastq_dir))
