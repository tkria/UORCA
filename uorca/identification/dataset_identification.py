#!/usr/bin/env python3
"""
Dataset Identification Script
============================

This script identifies relevant RNA-seq datasets from GEO based on research queries.
Key features:
- Automatic API rate limiting based on API key presence
- Parallel processing for efficient data retrieval
- Dataset-level validation with RNA-seq criteria
- AI-powered relevance scoring (enabled by default)
- Multi-dataset CSV output for batch analysis (enabled by default)

Requirements:
- ENTREZ_EMAIL environment variable (required by NCBI guidelines)
- OPENAI_API_KEY environment variable for AI features
- Optional: ENTREZ_API_KEY for faster processing

Workflow:
1. Extract search terms and query GEO database
2. Fetch complete dataset information using esummary v2.0
3. Retrieve SRA metadata for validation
4. Validate datasets based on RNA-seq criteria (>3 paired-end transcriptomic samples)
5. Assess relevance of valid datasets using AI
6. Generate results CSV and batch analysis input

Usage:
    python DatasetIdentification.py --query "research query"
"""

import argparse
import asyncio
import datetime

# Allow nested event loops: assess_subbatch is async and runs inside
# asyncio.gather(...), but call_structured uses pydantic-ai's agent.run_sync
# which tries to start a new event loop. Without nest_asyncio every batch
# fails with "This event loop is already running" and falls back to default
# scores. nest-asyncio is already a project runtime dep.
import nest_asyncio
nest_asyncio.apply()

import io
import json
import logging
import os
import statistics
import sys
import threading
import time
import traceback
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import partial
from tqdm import tqdm

import pandas as pd
import requests
from Bio import Entrez
from dotenv import load_dotenv
import GEOparse as gp
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure Entrez with explicit key setting
def _validate_entrez_api_key(api_key: str, email: str) -> bool:
    """
    Validate NCBI Entrez API key by making a test request.

    Some environments (e.g., HPC clusters behind firewalls) may cause API key
    authentication to fail with HTTP 400, even if the key is valid elsewhere.

    Returns:
        True if API key is valid and working, False otherwise.
    """
    import urllib.request
    import urllib.parse
    import urllib.error

    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi'
    params = {
        'email': email,
        'api_key': api_key,
        'retmode': 'json'
    }

    url = base_url + '?' + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 400:
            try:
                response_body = e.read().decode('utf-8')
                if 'api-key' in response_body.lower() or 'invalid' in response_body.lower():
                    return False
            except Exception:
                pass
        return False
    except Exception:
        return False


def _configure_entrez():
    """Configure Entrez with current environment variables, validating API key."""
    email = os.getenv("ENTREZ_EMAIL", "")
    if email:
        Entrez.email = email

    api_key = os.getenv("ENTREZ_API_KEY")
    if api_key:
        # Validate API key before using it
        if _validate_entrez_api_key(api_key, email):
            Entrez.api_key = api_key
            logging.info("NCBI Entrez API key validated successfully")
        else:
            # API key is invalid in this environment - disable it
            Entrez.api_key = None
            os.environ.pop("ENTREZ_API_KEY", None)  # Remove from environment
            logging.warning(
                "NCBI Entrez API key validation failed (HTTP 400: API key invalid). "
                "This can happen on HPC clusters or behind certain firewalls. "
                "Falling back to unauthenticated access with slower rate limits."
            )
    else:
        Entrez.api_key = None

# Initial configuration at module load
_configure_entrez()

# Rate limiting for API calls
class APIRateLimiter:
    """Thread-safe rate limiter for API calls with dynamic rates based on API key."""
    def __init__(self, base_delay: float = None):
        # Auto-detect optimal delay based on API key presence
        if base_delay is None:
            if os.getenv("ENTREZ_API_KEY"):
                # With API key: target 7 requests/second = 0.143 second delay
                self.min_delay = 1.0 / 7.0  # Conservative rate limiting
            else:
                # Without API key: target 2 requests/second = 0.5 second delay
                self.min_delay = 1.0 / 2.0  # Conservative rate limiting
        else:
            self.min_delay = base_delay

        self.last_call_time = 0
        self.lock = threading.Lock()

    def wait(self):
        """Wait if necessary to respect rate limits."""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            if time_since_last_call < self.min_delay:
                sleep_time = self.min_delay - time_since_last_call
                time.sleep(sleep_time)
            self.last_call_time = time.time()

# Global rate limiter instance with auto-detection
api_rate_limiter = APIRateLimiter()

# Constants
MAX_RETRIES = 3

# Configure logging
def setup_logging(verbose: bool = False, suppress_sra_warnings: bool = True) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(levelname)s - %(message)s"

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs/identification_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing handlers to avoid conflicts
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create timestamped log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"dataset_identification_{timestamp}.log"

    # Configure logging to both file and console with forced configuration
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[file_handler, console_handler],
        force=True  # Force reconfiguration even if logging was already configured
    )

    # Enforce retention: keep only the 5 most recent identification log files
    try:
        existing = sorted(
            [p for p in logs_dir.glob("dataset_identification_*.log") if p.is_file()],
            key=lambda p: p.stat().st_mtime,
        )
        # If more than 5, delete the oldest ones
        if len(existing) > 5:
            to_delete = existing[: len(existing) - 5]
            for old in to_delete:
                try:
                    old.unlink()
                except Exception:
                    pass
    except Exception:
        pass

    # Also configure warnings to use the same format
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.setLevel(level)

    # Optionally suppress SRA-related warnings by setting a higher threshold
    if suppress_sra_warnings and not verbose:
        # Create a custom filter to suppress specific SRA warnings
        class SRAWarningFilter(logging.Filter):
            def filter(self, record):
                # Suppress "No SRA records found" warnings unless in verbose mode
                if "No SRA records found" in record.getMessage():
                    return False
                return True

        # Apply filter to both file and console handlers
        for handler in logging.getLogger().handlers:
            handler.addFilter(SRAWarningFilter())

for name in ("openai", "openai._base_client", "httpx"):
    logging.getLogger(name).setLevel(logging.WARNING)

def load_prompt(file_path: str) -> str:
    """Load prompt from file. Supports both relative paths from module and absolute paths."""
    prompt_path = Path(file_path)
    # If it's a relative path starting with 'prompts/', resolve relative to this module
    if not prompt_path.is_absolute() and str(prompt_path).startswith('prompts/'):
        # Extract just the filename from prompts/dataset_identification/file.txt -> file.txt
        filename = prompt_path.name
        # Prompts are now in uorca/identification/prompts/
        module_dir = Path(__file__).parent
        prompt_path = module_dir / "prompts" / filename
    return prompt_path.read_text().strip()

# Query config management for Streamlit integration
def save_query_config(query: str) -> None:
    """Save the dataset identification query to a config file for Streamlit app."""
    # Config is now in uorca/config/
    config_dir = Path(__file__).parent.parent / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = config_dir / "dataset_query.json"
    config_data = {
        "query": query,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

def load_query_config() -> Optional[str]:
    """Load the dataset identification query from config file."""
    # Config is now in uorca/config/
    config_file = Path(__file__).parent.parent / "config" / "dataset_query.json"

    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                return config_data.get("query")
        except (json.JSONDecodeError, KeyError):
            return None
    return None

# Pydantic models for structured output
class ExtractedTerms(BaseModel):
    extracted_terms: List[str]
    expanded_terms: List[str]

class Assessment(BaseModel):
    ID: str
    RelevanceScore: int
    Justification: str

class Assessments(BaseModel):
    assessments: List[Assessment]

def call_structured(prompt: str, user_input: str, output_type: type) -> Any:
    """Make a structured LLM call using the configured provider."""
    from pydantic_ai import Agent
    from uorca.ai_provider import get_model

    agent = Agent(get_model(), output_type=output_type, system_prompt=prompt)
    result = agent.run_sync(user_input)
    return result.output

def extract_terms(research_query: str) -> ExtractedTerms:
    """Extract search terms from research query."""
    prompt = load_prompt("prompts/dataset_identification/extract_terms.txt")
    return call_structured(prompt, research_query, ExtractedTerms)

def perform_search(term: str, max_results: int = 2000) -> List[str]:
    """Search GEO database for a single term with RNA-seq filter."""
    search = f"{term} AND (\"Expression profiling by high throughput sequencing\"[Filter])"
    handle = Entrez.esearch(
        db="gds",
        term=search,
        retmode="xml",
        retmax=max_results
    )
    search_results = Entrez.read(handle)
    handle.close()

    geo_ids = search_results.get("IdList", [])

    return geo_ids

def get_basic_dataset_info(
    geo_ids: List[str],
    api_delay: float = 0.4,
) -> pd.DataFrame:
    """Get complete dataset information in parallel using esummary v2.0."""

    unique_ids = list(dict.fromkeys(geo_ids))
    workers = 6 if os.getenv("ENTREZ_API_KEY") else 1

    logging.info(f"Fetching complete dataset information for {len(unique_ids)} unique datasets...")

    datasets = []

    def fetch_info(geo_id: str) -> Optional[Dict[str, Any]]:
        """Fetch esummary information for a single GEO ID."""
        try:
            # Get UID for this GEO ID
            api_rate_limiter.wait()
            handle = Entrez.esearch(db="gds", term=geo_id)
            search_result = Entrez.read(handle)
            handle.close()
            uids = search_result.get("IdList", [])

            if not uids:
                logging.warning(f"No UID found for {geo_id}")
                return None

            uid = uids[0]

            api_rate_limiter.wait()
            handle = Entrez.esummary(db="gds", id=uid, version="2.0", retmode="xml")
            doc = Entrez.read(handle)["DocumentSummarySet"]["DocumentSummary"][0]
            handle.close()

            accession = doc["Accession"]
            if accession and accession.startswith("GSE"):
                return {
                    "ID": accession,
                    "Title": doc["title"],
                    "Summary": doc["summary"],
                    "Accession": accession,
                    "BioProject": doc["BioProject"],
                    "Species": doc["taxon"],
                    "Date": doc["PDAT"],
                    "NumSamples": doc["n_samples"],
                    "PrimaryPubMedID": (
                        doc["PubMedIds"][0] if doc.get("PubMedIds") else None
                    ),
                }
            else:
                logging.warning(f"Invalid accession for {geo_id}")
                return None

        except Exception as e:
            logging.warning(f"Error processing GEO ID {geo_id}: {e}")
            return None

    is_main_thread = threading.current_thread() is threading.main_thread()
    with ThreadPoolExecutor(max_workers=workers) as executor, \
         tqdm(total=len(unique_ids), desc="Fetching GEO summaries", unit="dataset", disable=not is_main_thread) as pbar:

        futures = {executor.submit(fetch_info, gid): gid for gid in unique_ids}
        for future in as_completed(futures):
            result = future.result()
            if result:
                datasets.append(result)
            pbar.update(1)

    df = pd.DataFrame(datasets)
    if not df.empty:
        df = df.drop_duplicates(subset=["ID"]).reset_index(drop=True)

    logging.info(f"Found {len(df)} unique GSE datasets with complete information")
    return df

# Streamlined SRA processing functions (old XML parsing functions removed)
def fetch_runinfo_from_bioproject(bioproject: str, api_delay: float = 0.4, suppress_missing_warnings: bool = True) -> pd.DataFrame:
    """Fetch RunInfo data using streamlined BioProject approach."""
    try:
        # 1. Search SRA and keep the server-side history
        srch = Entrez.read(Entrez.esearch(db="sra",
                                        term=f"{bioproject}[BioProject]",
                                        usehistory="y"))
        count, webenv, qk = int(srch["Count"]), srch["WebEnv"], srch["QueryKey"]

        if count == 0:
            if not suppress_missing_warnings:
                logging.warning(f"No SRA records found for BioProject {bioproject}")
            else:
                logging.debug(f"No SRA records found for BioProject {bioproject}")
            return pd.DataFrame()

        time.sleep(api_delay)

        # 2. Fetch run-level metadata
        handle = Entrez.efetch(db="sra",
                               rettype="runinfo",
                               retmode="text",
                               WebEnv=webenv,
                               query_key=qk,
                               retmax=count)

        # 3. Convert bytes → str → DataFrame
        csv_bytes = handle.read()                # bytes
        csv_text = csv_bytes.decode('utf-8')    # str
        runs = pd.read_csv(io.StringIO(csv_text))
        handle.close()
        return runs

    except Exception as e:
        logging.warning(f"Error fetching RunInfo for BioProject {bioproject}: {e}")
        return pd.DataFrame()



def calculate_dataset_sizes_from_runinfo(sra_df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate dataset sizes using only valid RNA-seq samples (TRANSCRIPTOMIC, RNA-Seq, PAIRED).

    Returns:
        Dictionary mapping GEO_Accession to total size in bytes (valid samples only)
    """
    if sra_df.empty or 'size_MB' not in sra_df.columns or 'GEO_Accession' not in sra_df.columns:
        logging.warning("Cannot calculate dataset sizes: missing required columns in runinfo data")
        return {}

    # Filter to only valid RNA-seq samples
    valid_samples = sra_df[
        (sra_df['LibrarySource'] == 'TRANSCRIPTOMIC') &
        (sra_df['LibraryStrategy'] == 'RNA-Seq') &
        (sra_df['LibraryLayout'] == 'PAIRED')
    ].copy()

    if valid_samples.empty:
        logging.info("No valid RNA-seq samples found for size calculation")
        return {}

    # Convert size_MB to bytes and group by GEO accession
    valid_samples['size_MB'] = pd.to_numeric(valid_samples['size_MB'], errors='coerce').fillna(0)
    valid_samples['size_bytes'] = (valid_samples['size_MB'] * 1024 * 1024).astype(int)

    dataset_sizes = valid_samples.groupby('GEO_Accession')['size_bytes'].sum().to_dict()

    # Log results
    total_datasets = len(dataset_sizes)
    total_size_gb = sum(dataset_sizes.values()) / (1024**3)

    return dataset_sizes


# ============================================================
# Metadata snapshot & PubMed enrichment for assessment
# ============================================================

def fetch_metadata_snapshot(accession: str) -> Optional[str]:
    """
    Fetch GEO sample metadata via GEOparse and return a compact snapshot.

    The snapshot shows only informative columns: constant columns (same value
    for all samples) and all-unique columns (likely identifiers) are removed.
    Multiple runs per sample are collapsed to one row per biological sample.

    Returns a human-readable string summary, or None on failure.
    """
    try:
        import subprocess

        # Download SOFT file manually (GEOparse's download can fail on some systems)
        cache_dir = Path("/tmp/uorca_geo_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        base_part = accession[3:-3] + "nnn"
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{base_part}/{accession}/soft/{accession}_family.soft.gz"
        filepath = cache_dir / f"{accession}_family.soft.gz"

        if not filepath.exists():
            subprocess.run(
                ["curl", "-fsSL", "-o", str(filepath), url],
                check=True, timeout=60, capture_output=True,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gse = gp.get_GEO(filepath=str(filepath), silent=True)

        meta = gse.phenotype_data.reset_index().rename(columns={"index": "GSM"})

        if meta.empty:
            return None

        # Remove identifier columns
        id_cols = ["GSM", "geo_accession"]
        meta = meta.drop(columns=[c for c in id_cols if c in meta.columns], errors="ignore")

        # Deduplicate rows (collapse multiple runs per sample)
        meta = meta.drop_duplicates()

        # Remove columns where all values are identical (uninformative)
        meta = meta.loc[:, meta.nunique() > 1]

        if meta.empty:
            return "No variable metadata columns found across samples."

        # Remove columns where all values are unique (likely identifiers)
        unique_cols = [c for c in meta.columns if meta[c].nunique() == len(meta)]
        meta = meta.drop(columns=unique_cols)

        if meta.empty:
            return "No informative metadata columns found (all columns are either constant or unique per sample)."

        # Build compact summary: column -> unique values
        lines = [f"Samples: {len(meta)}"]
        for col in meta.columns:
            unique_vals = meta[col].dropna().unique().tolist()
            # Count occurrences for values that appear
            val_counts = meta[col].value_counts()
            if len(unique_vals) <= 10:
                val_summary = ", ".join(
                    f"{v} (n={val_counts.get(v, 0)})" for v in unique_vals
                )
            else:
                # Too many values — show first 5 with counts
                top_vals = val_counts.head(5)
                val_summary = ", ".join(
                    f"{v} (n={c})" for v, c in top_vals.items()
                )
                val_summary += f", ... ({len(unique_vals)} unique values total)"
            lines.append(f"  {col}: {val_summary}")

        return "\n".join(lines)

    except Exception as e:
        logging.debug(f"Failed to fetch metadata snapshot for {accession}: {e}")
        return None


def fetch_pubmed_abstract(pubmed_id: str) -> Optional[str]:
    """Fetch a PubMed abstract via Entrez efetch. Returns abstract text or None."""
    if not pubmed_id or pd.isna(pubmed_id):
        return None
    try:
        api_rate_limiter.wait()
        handle = Entrez.efetch(db="pubmed", id=str(int(float(pubmed_id))),
                               rettype="abstract", retmode="text")
        text = handle.read()
        handle.close()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        # Strip header lines, keep just the abstract body
        text = text.strip()
        return text if len(text) > 50 else None
    except Exception as e:
        logging.debug(f"Failed to fetch PubMed abstract for {pubmed_id}: {e}")
        return None


def enrich_representatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich representative datasets with metadata snapshots and PubMed abstracts.
    Called before AI scoring so the scorer has richer context.
    """
    logging.info(f"Enriching {len(df)} representative datasets with metadata and PubMed abstracts...")

    metadata_snapshots = {}
    pubmed_abstracts = {}

    workers = 4 if os.getenv("ENTREZ_API_KEY") else 1
    is_main_thread = threading.current_thread() is threading.main_thread()

    # Fetch metadata snapshots in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor, \
         tqdm(total=len(df), desc="Fetching sample metadata", unit="dataset", disable=not is_main_thread) as pbar:

        futures = {}
        for _, row in df.iterrows():
            acc = row.get("Accession") or row.get("ID")
            if acc:
                futures[executor.submit(fetch_metadata_snapshot, acc)] = acc

        for future in as_completed(futures):
            acc = futures[future]
            try:
                result = future.result()
                if result:
                    metadata_snapshots[acc] = result
            except Exception as e:
                logging.debug(f"Metadata snapshot failed for {acc}: {e}")
            pbar.update(1)

    # Fetch PubMed abstracts in parallel
    pubmed_ids = df[df["PrimaryPubMedID"].notna()]["PrimaryPubMedID"].unique()
    if len(pubmed_ids) > 0:
        logging.info(f"Fetching {len(pubmed_ids)} PubMed abstracts...")

        # Build accession -> pubmed_id mapping
        acc_to_pmid = {}
        for _, row in df.iterrows():
            acc = row.get("Accession") or row.get("ID")
            pmid = row.get("PrimaryPubMedID")
            if acc and pmid and pd.notna(pmid):
                acc_to_pmid[acc] = pmid

        with ThreadPoolExecutor(max_workers=workers) as executor, \
             tqdm(total=len(acc_to_pmid), desc="Fetching PubMed abstracts", unit="abstract", disable=not is_main_thread) as pbar:

            futures = {}
            for acc, pmid in acc_to_pmid.items():
                futures[executor.submit(fetch_pubmed_abstract, pmid)] = acc

            for future in as_completed(futures):
                acc = futures[future]
                try:
                    result = future.result()
                    if result:
                        pubmed_abstracts[acc] = result
                except Exception as e:
                    logging.debug(f"PubMed abstract failed for {acc}: {e}")
                pbar.update(1)

    # Add columns to DataFrame
    df = df.copy()
    df["MetadataSnapshot"] = df.apply(
        lambda row: metadata_snapshots.get(row.get("Accession") or row.get("ID")),
        axis=1
    )
    df["PubMedAbstract"] = df.apply(
        lambda row: pubmed_abstracts.get(row.get("Accession") or row.get("ID")),
        axis=1
    )

    n_meta = sum(1 for v in metadata_snapshots.values() if v)
    n_abs = len(pubmed_abstracts)
    logging.info(f"Enrichment complete: {n_meta}/{len(df)} metadata snapshots, {n_abs}/{len(df)} PubMed abstracts")

    return df


async def assess_subbatch(
    df: pd.DataFrame, query: str, schema, key: str,
    rep: int, idx: int, total_batches: int, sem: asyncio.Semaphore,
    *,
    stage: str,
    stage1_biology_threshold: int,
    context: str = "",
) -> List:
    """Assess a sub-batch of datasets at the given pipeline stage.

    stage="stage1": loads the biology-only prompt, expects BiologyAssessments.
    stage="stage2": loads the biology+design prompt, expects RelevanceAssessments.
    On any exception, returns one default fallback assessment per input row
    (loud-errors policy: log a warning, do not silently drop the batch).
    """
    from uorca.identification.scoring import (
        BiologyAssessment,
        BiologyAssessments,
        RelevanceAssessment,
        RelevanceAssessments,
    )

    async with sem:
        try:
            # Build the per-stage assessment data
            assessment_data = []
            for _, row in df.iterrows():
                entry = {
                    "ID": row["ID"],
                    "Title": row.get("Title", ""),
                    "Species": row.get("Species", "Unknown"),
                    "NumSamples": int(row["NumSamples"]) if pd.notna(row.get("NumSamples")) else 0,
                    "Summary": row.get("Summary", ""),
                }
                if stage == "stage2":
                    snapshot = row.get("MetadataSnapshot")
                    if pd.notna(snapshot) and snapshot:
                        entry["SampleMetadata"] = snapshot
                    abstract = row.get("PubMedAbstract")
                    if pd.notna(abstract) and abstract:
                        entry["PubMedAbstract"] = abstract
                assessment_data.append(entry)

            # Load examples once and substitute into the prompt
            examples = load_prompt("prompts/dataset_identification/scoring_examples.txt")
            biology_examples_marker = "{biology_examples}"
            design_examples_marker = "{design_examples}"

            # Split the examples file at the design marker. Everything before
            # the design marker (after biology marker) is biology examples.
            if biology_examples_marker in examples and design_examples_marker in examples:
                bio_start = examples.index(biology_examples_marker) + len(biology_examples_marker)
                des_start = examples.index(design_examples_marker)
                biology_text = examples[bio_start:des_start].strip()
                design_text = examples[des_start + len(design_examples_marker):].strip()
            else:
                biology_text = examples
                design_text = examples

            if stage == "stage1":
                prompt = load_prompt("prompts/dataset_identification/assess_biology_stage1.txt")
                prompt = prompt.replace("{biology_examples}", biology_text)
                prompt = prompt.replace("{threshold}", str(stage1_biology_threshold))
                output_type = BiologyAssessments
            else:
                prompt = load_prompt("prompts/dataset_identification/assess_relevance_stage2.txt")
                prompt = prompt.replace("{biology_examples}", biology_text)
                prompt = prompt.replace("{design_examples}", design_text)
                output_type = RelevanceAssessments

            context_block = ""
            if stage == "stage2" and context.strip():
                context_block = (
                    f"\n\nAdditional Context (user-provided guidance for scoring): "
                    f"{context.strip()}"
                )
            user_input = (
                f"Research Query: {query}{context_block}\n\n"
                f"Datasets:\n{json.dumps(assessment_data, indent=2)}"
            )

            assessments = call_structured(prompt, user_input, output_type)
            return assessments.assessments
        except Exception as e:
            logging.warning(
                f"Error in assessment batch {idx+1}/{total_batches} (rep {rep+1}, stage={stage}): {e}"
            )
            # Fallback: one default assessment per row, loud not silent.
            if stage == "stage1":
                return [
                    BiologyAssessment(
                        ID=row["ID"], BiologyScore=5,
                        BiologyJustification="Assessment failed",
                    )
                    for _, row in df.iterrows()
                ]
            return [
                RelevanceAssessment(
                    ID=row["ID"], BiologyScore=5, DesignScore=5,
                    BiologyJustification="Assessment failed",
                    DesignJustification="Assessment failed",
                )
                for _, row in df.iterrows()
            ]

async def repeated_relevance(
    df: pd.DataFrame, query: str,
    repeats: int = 3, batch_size: int = 10, openai_api_jobs: int = 3,
    *,
    stage: str,
    stage1_biology_threshold: int,
    context: str = "",
) -> pd.DataFrame:
    """Run repeated relevance assessments and return per-axis aggregated scores.

    For stage="stage1": columns are ID, BiologyScore (mean), Run{N}BiologyScore,
    Run{N}BiologyJustification.
    For stage="stage2": columns are ID, BiologyScore, DesignScore (means),
    plus per-round Run{N}BiologyScore/DesignScore/BiologyJustification/DesignJustification.
    """
    logging.info(
        f"Starting {stage} relevance scoring: {repeats} reps, batch size {batch_size}, "
        f"parallel API jobs: {openai_api_jobs}"
    )
    sem = asyncio.Semaphore(openai_api_jobs)

    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]
    total_batches = len(batches)
    total_tasks = repeats * total_batches

    async def assess_with_progress(rep, idx, sub, pbar):
        result = await assess_subbatch(
            sub, query, None, "assessments", rep, idx, total_batches, sem,
            stage=stage, stage1_biology_threshold=stage1_biology_threshold,
            context=context,
        )
        pbar.update(1)
        pbar.set_postfix({"Rep": f"{rep+1}/{repeats}", "Batch": f"{(idx+1)+rep*total_batches}/{total_tasks}"})
        return result

    is_main_thread = threading.current_thread() is threading.main_thread()
    with tqdm(total=total_tasks, desc=f"Assessing relevance ({stage})", unit="batch", disable=not is_main_thread) as pbar:
        tasks = []
        for rep in range(repeats):
            for idx, sub in enumerate(batches):
                tasks.append(assess_with_progress(rep, idx, sub, pbar))
        all_results = await asyncio.gather(*tasks)

    # Aggregate per axis
    coll: Dict[str, Dict[str, Any]] = {}
    for result in all_results:
        for a in result:
            entry = coll.setdefault(a.ID, {
                "biology_scores": [], "biology_justifications": [],
                "design_scores": [], "design_justifications": [],
            })
            entry["biology_scores"].append(a.BiologyScore)
            entry["biology_justifications"].append(a.BiologyJustification)
            if stage == "stage2":
                entry["design_scores"].append(a.DesignScore)
                entry["design_justifications"].append(a.DesignJustification)

    records: List[Dict[str, Any]] = []
    for id_, v in coll.items():
        rec: Dict[str, Any] = {"ID": id_}
        # Mean BiologyScore
        rec["BiologyScore"] = round(statistics.mean(v["biology_scores"]), 2)
        for i, (score, just) in enumerate(zip(v["biology_scores"], v["biology_justifications"]), 1):
            rec[f"Run{i}BiologyScore"] = score
            rec[f"Run{i}BiologyJustification"] = just
        if stage == "stage2":
            rec["DesignScore"] = round(statistics.mean(v["design_scores"]), 2)
            for i, (score, just) in enumerate(zip(v["design_scores"], v["design_justifications"]), 1):
                rec[f"Run{i}DesignScore"] = score
                rec[f"Run{i}DesignJustification"] = just
        # The last-round justifications are useful as the "headline" reasoning
        rec["BiologyJustification"] = v["biology_justifications"][-1] if v["biology_justifications"] else ""
        if stage == "stage2":
            rec["DesignJustification"] = v["design_justifications"][-1] if v["design_justifications"] else ""
        records.append(rec)

    return pd.DataFrame(records)


def build_identification_metadata(
    *,
    research_query: str,
    search_terms,
    start_time,
    end_time,
    total_datasets_assessed: int,
    datasets_deemed_relevant: int,
    threshold: float,
) -> dict:
    """Build the identification_metadata.json payload for a completed run.

    ``search_terms`` are LLM-generated and vary between runs on the same query,
    so they are recorded here to keep the output directory self-describing —
    otherwise the only copy lives in the run log, outside the results.

    They arrive as a set, so they are sorted for a stable, diffable record and
    to keep the payload JSON-serialisable.
    """
    return {
        "input_query": research_query,
        "search_terms": sorted(search_terms),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_datasets_assessed": total_datasets_assessed,
        "datasets_deemed_relevant": datasets_deemed_relevant,
        "threshold_used": threshold,
    }


def build_theme_map_for_finished_run(output_dir: Path | str) -> bool:
    """Build the theme map for a finished identification run, and never fail the run.

    This is the **one** call site of ``build_theme_map`` that does not re-raise, and the
    exception is deliberate. Spec section 7 says every theme-map failure raises; spec
    decision 4 says the build sits at the very end of the run so it "can never cost a
    completed 60-minute run". Both cannot hold literally in the same place: the
    identification work has already succeeded and its results are already on disk, so
    letting an embedding, clustering or naming failure propagate would turn a successful
    hour of work into a non-zero exit.

    The resolution, approved with the design: log the failure at ERROR level with the
    full traceback, and record it in ``theme_map/FAILED.txt`` so the Identify page shows
    it loudly and offers a rebuild. Nothing is swallowed — it is written down in two
    places — but the identification run still exits clean.

    The other two callers (``python -m uorca.identification.theme_map`` and the GUI
    button) raise normally. Do not add a catch there.

    Returns:
        True when the map was built, False when the failure was recorded instead.
    """
    from uorca.identification import theme_map

    try:
        theme_map.build_theme_map(output_dir)
    except Exception as exc:
        logging.error(
            "Theme map build failed for %s. The identification run itself succeeded and "
            "its results are complete; the map can be rebuilt from the Identify page. "
            "%s: %s\n%s",
            output_dir,
            type(exc).__name__,
            exc,
            traceback.format_exc(),
        )
        try:
            theme_map.write_failure(output_dir, f"{type(exc).__name__}: {exc}")
        except Exception as record_exc:
            # Recording the failure must not be what finally sinks a completed run.
            # The real failure is already in the log above; this second line says the
            # panel will not be able to show it.
            logging.error(
                "The theme map failure for %s could not be recorded in FAILED.txt: "
                "%s: %s",
                output_dir,
                type(record_exc).__name__,
                record_exc,
            )
        return False

    logging.info("Theme map built for %s", output_dir)
    return True


def main():
    """
    Main function for dataset identification.

    Process:
    1. Extract search terms from research query
    2. Search GEO database for relevant datasets
    3. Validate datasets for RNA-seq compatibility
    4. Assess relevance of representatives using AI
    5. Generate comprehensive results CSV (all datasets) and batch analysis CSV (filtered)
    """
    parser = argparse.ArgumentParser(
        description='Identify and evaluate relevant RNA-seq datasets from GEO for a biological research query',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # === Core Parameters ===
    core_group = parser.add_argument_group('Core Options', 'Essential parameters for dataset identification')
    core_group.add_argument('-q', '--query', required=True,
                           help='Biological research query (e.g., "neuroblastoma tumor vs normal")')
    core_group.add_argument('-o', '--output', default='./dataset_identification_results',
                           help='Output directory for results')
    core_group.add_argument('-t', '--threshold', type=float, default=7.0,
                           help='Relevance score threshold (0-10) for including datasets')

    # === Search Parameters ===
    search_group = parser.add_argument_group('Search Options', 'Control dataset search and evaluation')
    search_group.add_argument('-m', '--max-per-term', type=int, default=500,
                             help='Maximum datasets to retrieve per search term')
    search_group.add_argument('-n', '--num-assess', type=int, default=300,
                             help='Number of datasets to assess for relevance')

    # === Advanced Parameters ===
    advanced_group = parser.add_argument_group('Advanced Options', 'Fine-tune algorithm behavior (expert users)')
    advanced_group.add_argument('-r', '--rounds', type=int, default=3,
                               help='Number of independent relevance scoring rounds for reliability')
    advanced_group.add_argument('-b', '--batch-size', type=int, default=20,
                               help='Datasets per AI evaluation batch (affects memory usage)')
    advanced_group.add_argument('-v', '--verbose', action='store_true',
                               help='Enable verbose logging (DEBUG level)')

    # === Scoring Parameters ===
    scoring_group = parser.add_argument_group(
        'Scoring Options',
        'Tune the relevance scoring derivation and Stage 1→2 gating',
    )
    scoring_group.add_argument('--biology-weight', type=float, default=0.8,
                               help='Weight on BiologyScore in RelevanceScore (default 0.8)')
    scoring_group.add_argument('--design-min', type=int, default=0,
                               help='Hard cap on RelevanceScore when DesignScore < design_min (default 0 = no cap)')
    scoring_group.add_argument('--stage1-biology-threshold', type=int, default=3,
                               help='Stage 1 BiologyScore threshold to graduate to Stage 2 (default 3)')
    scoring_group.add_argument('--library-source', type=str, default='bulk',
                               choices=['bulk', 'sc', 'both'],
                               help='Library source filter for validity check')
    scoring_group.add_argument('--context', type=str, default='',
                               help='Extra user-provided guidance fed into the Stage 2 scoring prompt')

    args = parser.parse_args()

    # Range validation (loud-error on out-of-range values)
    if not (0.0 <= args.biology_weight <= 1.0):
        parser.error("--biology-weight must be in [0.0, 1.0]")
    if not (0 <= args.design_min <= 10):
        parser.error("--design-min must be in [0, 10]")
    if not (0 <= args.stage1_biology_threshold <= 10):
        parser.error("--stage1-biology-threshold must be in [0, 10]")

    # Check for required email (Entrez guidelines)
    if not os.getenv("ENTREZ_EMAIL"):
        logging.error("Email is required for NCBI Entrez API access. Please set ENTREZ_EMAIL environment variable.")
        logging.error("Please set the ENTREZ_EMAIL environment variable with your email address.")
        logging.error("This is required by NCBI guidelines for API usage.")
        return

    # Auto-determine API delay based on API key presence
    api_delay = 0.25 if os.getenv("ENTREZ_API_KEY") else 0.6

    # Set up logging in the output directory
    from pathlib import Path
    output_path = Path(args.output)
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(verbose=args.verbose, suppress_sra_warnings=not args.verbose)

    # Reconfigure Entrez to ensure API key is set in this process/thread
    _configure_entrez()

    # Log API key detection status and rate limiting configuration
    if os.getenv("ENTREZ_API_KEY"):
        logging.info("NCBI API key detected - using faster rate limits")
        logging.info("Rate Limiting Configuration:")
        logging.info(f"API delay: {1.0/7.0:.3f}s (~7 req/sec)")
        logging.info(f"GEO workers: 6, SRA workers: 6")
    else:
        logging.info("No NCBI API key found - using standard rate limits")
        logging.info("Rate Limiting Configuration:")
        logging.info(f"API delay: {1.0/2.0:.3f}s (~2 req/sec)")
        logging.info(f"GEO workers: 1, SRA workers: 1")

    research_query = args.query
    logging.info(f"Starting dataset identification for query: {research_query}")

    # Save query to config file for Streamlit app
    save_query_config(research_query)

    # Track timing for metadata
    start_time = datetime.datetime.now()

    try:
        # Step 1: Extract terms from research query
        logging.info("Determining search terms...")
        terms = extract_terms(research_query)

        # Step 2: Search GEO database
        search_terms = set(terms.extracted_terms + terms.expanded_terms)
        logging.info(f"Searching GEO using {len(search_terms)} unique search terms: {', '.join(search_terms)}")

        geo_ids = []
        # Detect if running in a background thread (e.g., from GUI) - disable tqdm to avoid BrokenPipeError
        is_main_thread = threading.current_thread() is threading.main_thread()
        search_iterator = tqdm(search_terms, desc="Searching GEO database", unit="term", disable=not is_main_thread)

        for term in search_iterator:
            term_ids = perform_search(term, args.max_per_term)
            geo_ids.extend(term_ids)
            # Apply rate limiting between searches to follow NCBI guidelines
            time.sleep(api_delay)

        unique_geo_ids = list(dict.fromkeys(geo_ids))
        logging.info(f"Found {len(geo_ids)} total GEO IDs, {len(unique_geo_ids)} unique IDs")

        if not unique_geo_ids:
            logging.error("No datasets found in search")
            return

        # Step 3: Get basic dataset information
        basic_datasets_df = get_basic_dataset_info(
            unique_geo_ids,
            api_delay,
        )

        if basic_datasets_df.empty:
            logging.error("No valid GSE datasets found")
            return

        # Step 4: Use complete dataset information from esummary v2.0
        enriched_datasets_df = basic_datasets_df

        # Step 5: Fetch SRA data for ALL datasets to determine validity (optimal workflow)
        logging.info(f"Fetching SRA metadata for {len(enriched_datasets_df)} datasets to determine validity...")
        to_fetch = enriched_datasets_df  # Fetch SRA data for ALL datasets

        def fetch_sra_for_dataset(row, api_delay):
            """Fetch SRA data for a single dataset with rate limiting using streamlined approach."""
            acc = row['ID']
            bioproject = row.get('BioProject', '')

            try:
                # Apply rate limiting before making API calls
                api_rate_limiter.wait()

                if bioproject:
                    df_run = fetch_runinfo_from_bioproject(bioproject, api_delay, suppress_missing_warnings=True)
                    if not df_run.empty:
                        df_run.insert(0, 'GEO_Accession', acc)
                        return df_run, True
                    else:
                        return pd.DataFrame(), False
                else:
                    logging.warning(f"No BioProject found for {acc}")
                    return pd.DataFrame(), False

            except Exception as e:
                logging.warning(f'Error processing {acc}: {e}')
                return pd.DataFrame(), False

        runs = []
        successful_datasets = 0
        total_runs_fetched = 0

        # Auto-determine optimal SRA worker count based on API key
        if os.getenv("ENTREZ_API_KEY"):
            sra_workers = 6  # Increased workers with conservative rate limiting
        else:
            sra_workers = 1  # Very conservative without API key
        api_rate_limiter = APIRateLimiter()

        # Track timing for performance summary
        sra_start_time = time.time()

        with ThreadPoolExecutor(max_workers=sra_workers) as executor:
            fetch_func = partial(fetch_sra_for_dataset, api_delay=api_delay)
            future_to_row = {executor.submit(fetch_func, row): row for _, row in to_fetch.iterrows()}

            # Use tqdm progress bar for SRA fetching (disable if not main thread)
            is_main_thread = threading.current_thread() is threading.main_thread()
            with tqdm(total=len(future_to_row), desc="Fetching SRA data", unit="dataset", disable=not is_main_thread) as pbar:
                for future in as_completed(future_to_row):
                    df_result, success = future.result()
                    pbar.update(1)

                    if success and not df_result.empty:
                        runs.append(df_result)
                        successful_datasets += 1
                        total_runs_fetched += len(df_result)
                        pbar.set_postfix(successful=successful_datasets)

        # Performance summary
        sra_end_time = time.time()
        sra_elapsed_time = sra_end_time - sra_start_time
        datasets_per_second = len(to_fetch) / sra_elapsed_time if sra_elapsed_time > 0 else 0
        success_rate = (successful_datasets / len(to_fetch)) * 100 if len(to_fetch) > 0 else 0

        if runs:
            sra_df = pd.concat(runs, ignore_index=True)
        else:
            sra_df = pd.DataFrame(columns=['GEO_Accession'])
            logging.warning("No SRA runs were successfully fetched")

        # Calculate dataset sizes using runinfo data
        dataset_sizes = {}
        if not sra_df.empty:
            logging.info("Calculating dataset sizes from runinfo data...")
            dataset_sizes = calculate_dataset_sizes_from_runinfo(sra_df)

        # Add dataset sizes to enriched datasets before merging
        if dataset_sizes:
            enriched_datasets_df['DatasetSizeBytes'] = enriched_datasets_df['ID'].map(dataset_sizes)
            enriched_datasets_df['DatasetSizeGB'] = enriched_datasets_df['DatasetSizeBytes'] / (1024**3)
            # Fill NaN values with 0 for datasets where size couldn't be calculated
            enriched_datasets_df['DatasetSizeBytes'] = enriched_datasets_df['DatasetSizeBytes'].fillna(0)
            enriched_datasets_df['DatasetSizeGB'] = enriched_datasets_df['DatasetSizeGB'].fillna(0.0)
        else:
            enriched_datasets_df['DatasetSizeBytes'] = 0
            enriched_datasets_df['DatasetSizeGB'] = 0.0

        # Merge all GEO datasets with fetched SRA info
        final_results = enriched_datasets_df.merge(
            sra_df,
            left_on='ID',
            right_on='GEO_Accession',
            how='left'
        )

        # Step 6: Validate ALL datasets based on complete SRA criteria (dataset-level validation)
        validation_data = []

        # Determine which LibrarySource values count as valid for this run
        if args.library_source == 'bulk':
            accepted_sources = {'TRANSCRIPTOMIC'}
        elif args.library_source == 'sc':
            accepted_sources = {'TRANSCRIPTOMIC SINGLE CELL'}
        else:  # both
            accepted_sources = {'TRANSCRIPTOMIC', 'TRANSCRIPTOMIC SINGLE CELL'}

        # Group by dataset (GSE ID) for proper dataset-level validation
        grouped_datasets = list(final_results.groupby('ID'))
        is_main_thread = threading.current_thread() is threading.main_thread()
        for gse_id, group in tqdm(grouped_datasets, desc="Validating datasets", unit="dataset", disable=not is_main_thread):
            # Count samples that meet RNA-seq criteria
            rnaseq_samples = 0
            total_samples = len(group)

            # Track all library types present in this dataset
            lib_sources = set()
            lib_layouts = set()
            lib_strategies = set()

            # Check each sample in the dataset
            for _, row in group.iterrows():
                lib_source = row.get('LibrarySource')
                lib_layout = row.get('LibraryLayout')
                lib_strategy = row.get('LibraryStrategy')

                # Track all library types (for reporting)
                if pd.notna(lib_source):
                    lib_sources.add(lib_source)
                if pd.notna(lib_layout):
                    lib_layouts.add(lib_layout)
                if pd.notna(lib_strategy):
                    lib_strategies.add(lib_strategy)

                # Check if this sample meets RNA-seq criteria
                if (pd.notna(lib_source) and lib_source in accepted_sources and
                    pd.notna(lib_strategy) and lib_strategy == 'RNA-Seq' and
                    pd.notna(lib_layout) and lib_layout == 'PAIRED'):
                    rnaseq_samples += 1

            # Dataset-level validation: need >3 samples meeting RNA-seq criteria
            if total_samples == 0:
                valid = False
                reason = "No SRA metadata available"
            elif rnaseq_samples == 0:
                accepted_sources_str = '/'.join(sorted(accepted_sources))
                reason_parts = []
                if not (lib_sources & accepted_sources):
                    reason_parts.append(
                        f"LibrarySource: {'/'.join(lib_sources) if lib_sources else 'None'} "
                        f"(accepted: {accepted_sources_str})"
                    )
                if 'RNA-Seq' not in lib_strategies:
                    reason_parts.append(f"LibraryStrategy: {'/'.join(lib_strategies) if lib_strategies else 'None'}")
                if 'PAIRED' not in lib_layouts:
                    reason_parts.append(f"LibraryLayout: {'/'.join(lib_layouts) if lib_layouts else 'None'}")
                valid = False
                reason = f"No RNA-seq samples found ({'; '.join(reason_parts)})"
            elif rnaseq_samples <= 3:
                valid = False
                reason = f"Insufficient RNA-seq samples: {rnaseq_samples} (need >3)"
            else:
                valid = True
                # Include info about mixed library types if present
                mixed_info = []
                if len(lib_sources) > 1:
                    mixed_info.append(f"Sources: {'/'.join(lib_sources)}")
                if len(lib_strategies) > 1:
                    mixed_info.append(f"Strategies: {'/'.join(lib_strategies)}")
                if len(lib_layouts) > 1:
                    mixed_info.append(f"Layouts: {'/'.join(lib_layouts)}")

                if mixed_info:
                    reason = f"Valid: {rnaseq_samples}/{total_samples} RNA-seq samples (mixed types: {'; '.join(mixed_info)})"
                else:
                    reason = f"Valid: {rnaseq_samples}/{total_samples} RNA-seq samples"

            validation_data.append({
                'ID': gse_id,
                'valid_dataset': valid,
                'validation_reason': reason,
                'rnaseq_samples': rnaseq_samples,
                'total_samples': total_samples
            })

        validation_df = pd.DataFrame(validation_data)

        # Merge validation results back to final_results
        final_results = final_results.merge(validation_df, on='ID', how='left', validate='many_to_one')

        # Step 7: Create dataset-level summary
        # Create one row per dataset (remove duplicates from multiple samples)
        dataset_summary = final_results.groupby('ID').agg({
            'Title': 'first',
            'Summary': 'first',
            'Accession': 'first',
            'Species': 'first',
            'Date': 'first',
            'NumSamples': 'first',
            'PrimaryPubMedID': 'first',
            'DatasetSizeBytes': 'first',
            'DatasetSizeGB': 'first',
            'valid_dataset': 'first',
            'validation_reason': 'first',
            'rnaseq_samples': 'first',
            'total_samples': 'first'
        }).reset_index()

        # Filter to valid and invalid datasets for processing
        valid_datasets_df = dataset_summary[dataset_summary['valid_dataset'] == True].copy()
        invalid_datasets_df = dataset_summary[dataset_summary['valid_dataset'] == False].copy()

        logging.info(f"Found {len(valid_datasets_df)} valid datasets and {len(invalid_datasets_df)} invalid datasets")

        if len(valid_datasets_df) == 0:
            logging.error("No valid datasets found - cannot proceed with relevance assessment")
            # Still output all datasets for transparency
            final_results = pd.concat([valid_datasets_df, invalid_datasets_df], ignore_index=True)
        else:
            n_valid = len(valid_datasets_df)

            # ============================================================
            # TWO-STAGE SCORING
            # Stage 1: Lightweight score ALL valid datasets (Title+Summary, 1 round)
            # Stage 2: Full enriched score on top 20% from Stage 1
            # ============================================================

            # Step 8: Stage 1 — Lightweight relevance scoring of ALL valid datasets
            logging.info(f"Stage 1: Lightweight scoring of all {n_valid} valid datasets (1 round, Title+Summary only)...")
            stage1_df = asyncio.run(repeated_relevance(
                valid_datasets_df, research_query,
                repeats=1, batch_size=args.batch_size,
                openai_api_jobs=4,
                stage="stage1",
                stage1_biology_threshold=args.stage1_biology_threshold,
            ))

            # Merge Stage 1 scores back. Stage 1 returns BiologyScore, BiologyJustification,
            # Run1BiologyScore, Run1BiologyJustification — rename to Stage1* for clarity.
            valid_with_stage1 = valid_datasets_df.merge(
                stage1_df.rename(columns={
                    "BiologyScore": "Stage1BiologyScore",
                    "BiologyJustification": "Stage1BiologyJustification",
                }),
                on='ID', how='left',
            )
            # Drop the per-round Run* columns from Stage 1 (we only ran 1 round; the Stage1*
            # columns we just renamed carry the data we want).
            extra_cols = [c for c in valid_with_stage1.columns
                          if c.startswith("Run1") and "Stage1" not in c]
            valid_with_stage1 = valid_with_stage1.drop(columns=extra_cols, errors="ignore")

            # Threshold-based gate: any dataset with BiologyScore >= threshold graduates
            scored_mask = valid_with_stage1['Stage1BiologyScore'].notna()
            scored_datasets = valid_with_stage1[scored_mask].copy()
            stage2_candidates = scored_datasets[
                scored_datasets['Stage1BiologyScore'] >= args.stage1_biology_threshold
            ].copy()
            n_graduated = len(stage2_candidates)

            logging.info(
                f"Stage 1 complete: {len(scored_datasets)} scored; "
                f"{n_graduated} datasets graduated to Stage 2 "
                f"(threshold = BiologyScore >= {args.stage1_biology_threshold})"
            )

            if n_graduated == 0:
                logging.info(
                    f"0 datasets graduated to Stage 2. "
                    "Consider lowering --stage1-biology-threshold."
                )

            # Log Stage 1 score distribution
            if len(scored_datasets) > 0:
                logging.info(f"Stage 1 BiologyScore distribution: "
                             f"mean={scored_datasets['Stage1BiologyScore'].mean():.2f}, "
                             f"median={scored_datasets['Stage1BiologyScore'].median():.2f}, "
                             f"max={scored_datasets['Stage1BiologyScore'].max():.2f}, "
                             f">=5: {len(scored_datasets[scored_datasets['Stage1BiologyScore'] >= 5])}, "
                             f">=3: {len(scored_datasets[scored_datasets['Stage1BiologyScore'] >= 3])}")

            # Step 9: Stage 2 — Enrich top candidates with metadata + PubMed
            logging.info(f"Stage 2: Enriching {len(stage2_candidates)} candidates with sample metadata and PubMed abstracts...")
            stage2_candidates = enrich_representatives(stage2_candidates)

            # Step 10: Stage 2 — Full relevance scoring (3 rounds, enriched)
            logging.info(f"Stage 2: Full scoring of {len(stage2_candidates)} enriched candidates ({args.rounds} rounds)...")
            stage2_assessed_df = asyncio.run(repeated_relevance(
                stage2_candidates, research_query,
                repeats=args.rounds, batch_size=args.batch_size,
                openai_api_jobs=4,
                stage="stage2",
                stage1_biology_threshold=args.stage1_biology_threshold,
                context=args.context,
            ))

            # Merge Stage 2 per-axis scores back. Stage 2 columns: BiologyScore, DesignScore,
            # BiologyJustification, DesignJustification, Run{N}* — all flow into valid_with_scores.
            valid_with_scores = valid_with_stage1.merge(
                stage2_assessed_df, on='ID', how='left',
            )

            # Compute final RelevanceScore + Justification using the per-project
            # derivation formulas. Stage-1-only rows (BiologyScore == NaN) get
            # RelevanceScore = Stage1BiologyScore.
            from uorca.identification.scoring import (
                row_justification, row_relevance,
            )

            valid_with_scores["RelevanceScore"] = valid_with_scores.apply(
                lambda row: row_relevance(
                    row,
                    biology_weight=args.biology_weight,
                    design_min=args.design_min,
                ),
                axis=1,
            )
            valid_with_scores["Justification"] = valid_with_scores.apply(
                lambda row: row_justification(row, design_min=args.design_min),
                axis=1,
            )

            logging.info(f"Scoring complete: {len(stage2_candidates)} datasets fully assessed (Stage 2), "
                         f"{n_valid} datasets screened (Stage 1)")

            # Combine valid (with scores) and invalid datasets for final output
            final_results = pd.concat([
                valid_with_scores,
                invalid_datasets_df
            ], ignore_index=True)

        # Note: We include ALL datasets in the main results file regardless of threshold
        # The threshold will be applied only to the batch_analysis_input.csv file
        logging.info(f"Main results will include all {len(final_results)} datasets (threshold applied only to batch analysis file)")

        # Remove large/internal columns to prevent CSV malformation
        drop_cols = ['embedding', 'MetadataSnapshot', 'PubMedAbstract']
        final_results = final_results.drop(columns=[c for c in drop_cols if c in final_results.columns])

        # Create simplified final dataframe with only requested columns
        final = pd.DataFrame()

        # Helper functions for URLs
        def create_pubmed_url(pmid):
            if pd.notna(pmid):
                return f"https://pubmed.ncbi.nlm.nih.gov/{int(pmid)}/"
            return None

        def create_geo_url(accession):
            if pd.notna(accession):
                return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
            return None

        # Create the simplified output columns
        final['GEO_Accession'] = final_results['ID']
        final['Title'] = final_results.get('Title', '')
        final['Summary'] = final_results.get('Summary', '')
        final['Species'] = final_results.get('Species', 'Unknown')
        final['Date'] = final_results.get('Date', '')
        final['Samples'] = final_results.get('rnaseq_samples', 0)  # Number of valid RNA-seq samples
        final['Valid'] = final_results.get('valid_dataset', False).apply(lambda x: 'Yes' if x else 'No')
        final['Validation_Result'] = final_results.get('validation_reason', 'Unknown')
        final['DatasetSizeGB'] = final_results.get('DatasetSizeGB', 0.0)
        final["RelevanceScore"] = final_results.get("RelevanceScore", None)
        final["Justification"] = final_results.get("Justification", None)
        # Per-axis columns
        final["BiologyScore"] = final_results.get("BiologyScore", None)
        final["DesignScore"] = final_results.get("DesignScore", None)
        final["BiologyJustification"] = final_results.get("BiologyJustification", None)
        final["DesignJustification"] = final_results.get("DesignJustification", None)
        final["Stage1BiologyScore"] = final_results.get("Stage1BiologyScore", None)
        final["Stage1BiologyJustification"] = final_results.get("Stage1BiologyJustification", None)
        for n in (1, 2, 3):
            final[f"Run{n}BiologyScore"] = final_results.get(f"Run{n}BiologyScore", None)
            final[f"Run{n}DesignScore"] = final_results.get(f"Run{n}DesignScore", None)
            final[f"Run{n}BiologyJustification"] = final_results.get(f"Run{n}BiologyJustification", None)
            final[f"Run{n}DesignJustification"] = final_results.get(f"Run{n}DesignJustification", None)
        final['PubMed_URL'] = final_results.get('PrimaryPubMedID').apply(create_pubmed_url)
        final['GEO_URL'] = final_results['ID'].apply(create_geo_url)

        final = final.drop_duplicates(subset=['GEO_Accession'])

        # Save to output directory
        main_output_file = output_path / "Dataset_identification_result.csv"

        final.to_csv(main_output_file, index=False)
        message = f'Saved combined table to {main_output_file}'
        logging.info(message)

        # Generate multi-dataset CSV for batch analysis (default behavior)
        # This CSV only includes datasets that are valid, assessed, and above threshold
        multi_df = final[['GEO_Accession', 'Species', 'RelevanceScore', 'Valid', 'DatasetSizeGB']].copy()

        # Filter for batch analysis CSV (more restrictive than main results)
        valid_datasets = multi_df[multi_df['Valid'] == 'Yes']
        assessed_datasets = valid_datasets.dropna(subset=['RelevanceScore'])  # Only assessed datasets
        above_threshold = assessed_datasets[assessed_datasets['RelevanceScore'] >= args.threshold]  # Apply threshold

        multi_df = above_threshold.sort_values('RelevanceScore', ascending=False)
        multi_df = multi_df.rename(columns={'Species': 'organism', 'GEO_Accession': 'Accession'})

        # Log filtering steps for transparency
        logging.info(f"Batch analysis filtering: {len(valid_datasets)} valid → {len(assessed_datasets)} assessed → {len(above_threshold)} above threshold ({args.threshold})")

        # Set path for the multi-dataset CSV in the output directory
        multi_csv_path = output_path / 'selected_datasets.csv'

        multi_df.to_csv(multi_csv_path, index=False)
        message = f"Generated batch analysis input CSV at {multi_csv_path} with {len(multi_df)} datasets (threshold {args.threshold} applied)"
        logging.info(message)

        # Log summary results
        logging.info("="*80)
        logging.info("DATASET IDENTIFICATION RESULTS")
        logging.info("="*80)
        logging.info(f"Research Query: {research_query}")
        logging.info(f"Total datasets found: {len(final)}")
        logging.info(f"Valid datasets: {len(final[final['Valid'] == 'Yes'])}")
        logging.info(f"Invalid datasets: {len(final[final['Valid'] == 'No'])}")
        assessed_count = len(final.dropna(subset=['RelevanceScore']))
        logging.info(f"Assessed datasets: {assessed_count}")

        if args.threshold > 0:
            above_threshold_count = len(final[final['RelevanceScore'].notna() & (final['RelevanceScore'] >= args.threshold)])
            logging.info(f"Above threshold ({args.threshold}): {above_threshold_count}")
            logging.info(f"Included in batch analysis CSV: {len(multi_df)}")

        assessed_final = final.dropna(subset=['RelevanceScore']).sort_values('RelevanceScore', ascending=False)

        # Generate metadata JSON file
        end_time = datetime.datetime.now()

        # Calculate metadata
        total_datasets_assessed = len(final.dropna(subset=['RelevanceScore'])) if 'RelevanceScore' in final.columns else 0
        datasets_deemed_relevant = len(final[final['RelevanceScore'].notna() & (final['RelevanceScore'] >= args.threshold)]) if 'RelevanceScore' in final.columns and args.threshold > 0 else 0

        metadata = build_identification_metadata(
            research_query=research_query,
            search_terms=search_terms,
            start_time=start_time,
            end_time=end_time,
            total_datasets_assessed=total_datasets_assessed,
            datasets_deemed_relevant=datasets_deemed_relevant,
            threshold=args.threshold,
        )

        # Save metadata JSON
        metadata_file = output_path / "identification_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Saved identification metadata to: {metadata_file}")

        # Spec section 8: the theme map is the LAST block of the run, so it can never
        # cost a completed 60-minute identification. See
        # build_theme_map_for_finished_run for why this one call site records its
        # failure instead of raising it (spec section 7 vs spec decision 4).
        build_theme_map_for_finished_run(output_path)

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
