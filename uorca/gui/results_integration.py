#!/usr/bin/env python3
"""
results_integration.py - Integrates results from multiple UORCA RNA-seq analyses

This module collates differential expression results and normalized expression data
across experiments and contrasts, producing interactive visualizations
that highlight important genes and patterns.

The ResultsIntegrator class performs three main functions:
1. Identifying differentially expressed genes (DEGs) across contrasts
   - Finds genes differentially expressed in multiple contrasts
   - Identifies contrast-specific genes with large fold changes

2. Creating an interactive heatmap of log fold changes
   - Shows gene expression patterns across contrasts
   - Clusters genes and contrasts to reveal patterns

3. Generating expression plots for selected genes
   - Shows expression levels across samples grouped by condition
   - Uses sample information from edger_analysis_samples.csv

The main outputs are:
1. Interactive heatmap of important genes with LFC values (clustered on both axes)
2. Interactive boxplot/violin plots of gene expression across samples
3. HTML report integrating all visualizations

Import:
  from uorca.gui.results_integration import ResultsIntegrator
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from typing import List, Dict, Tuple, Optional, Union, Set, Sequence
import logging
import re
from pathlib import Path
from datetime import datetime
import json
import math

import textwrap
from dataclasses import dataclass
import html

# p-value columns and labels live in uorca.core.pvalues; re-exported for the GUI.
from uorca.core.pvalues import (
    ADJUSTED_P_COLUMN,
    ADJUSTED_P_HELP,
    ADJUSTED_P_LABEL,
    RAW_P_COLUMN,
    RAW_P_LABEL,
    MissingAdjustedPValueError,
    get_adjusted_p_column,
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Heatmap contrast labels
# ---------------------------------------------------------------------------

SHORT_LABEL_CHARS = 25   # one-line, angled heatmap tick labels
HEATMAP_TICK_ANGLE = -45
ELLIPSIS = "…"
HOVER_WRAP_CHARS = 80
HEATMAP_MIN_COL_PX = 28  # measured in the live app: 25-character labels at -45 deg do not overlap
FORMULA_MAX_CHARS = 300
GROUP_LINE_CHARS = 24
GROUP_MAX_LINES = 3


@dataclass(frozen=True)
class ContrastLabel:
    """One heatmap column.

    ``accession`` is the dataset, ``full`` the full contrast name (plain text,
    for hover and the key table) and ``display`` the short, unique,
    HTML-escaped tick label.
    """
    accession: str
    full: str
    display: str


def _split_label_tokens(text: str) -> List[str]:
    """Split after each '_' or ' ' so that joining the tokens restores the text."""
    return [t for t in re.split(r"(?<=[_ ])", text) if t]


def wrap_contrast_label(text: str,
                        line_chars: int = GROUP_LINE_CHARS,
                        max_lines: int = GROUP_MAX_LINES) -> str:
    """Wrap a contrast name onto at most ``max_lines`` lines joined by ``<br>``.

    Lines break after '_' or ' '. A token longer than a line is split. No
    character is dropped unless the text needs more than ``max_lines`` lines;
    then the last line ends in an ellipsis. Each line is HTML-escaped after
    wrapping, so a name containing '<' or '&' cannot inject Plotly markup.
    """
    lines: List[str] = []
    current = ""
    for token in _split_label_tokens(text):
        while len(token) > line_chars:
            if current:
                lines.append(current)
                current = ""
            lines.append(token[:line_chars])
            token = token[line_chars:]
        if len(current) + len(token) > line_chars:
            lines.append(current)
            current = token
        else:
            current += token
    if current:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        lines[-1] = last[: line_chars - len(ELLIPSIS)] + ELLIPSIS
    return "<br>".join(html.escape(line, quote=False) for line in lines)


def _shared_leading_words(names: Sequence[str]) -> str:
    """Leading '_'-separated words that every name shares, e.g. "PPARG_" for
    "PPARG_KD1_vs_control" and "PPARG_rescue_vs_dox". Empty when fewer than two
    distinct names, or when dropping the words would leave a name empty."""
    distinct = list(dict.fromkeys(names))
    if len(distinct) < 2:
        return ""
    token_lists = [_split_label_tokens(n) for n in distinct]
    shared: List[str] = []
    for tokens in zip(*token_lists):
        if all(t == tokens[0] for t in tokens) and tokens[0].endswith(("_", " ")):
            shared.append(tokens[0])
        else:
            break
    prefix = "".join(shared)
    if any(len(n) <= len(prefix) for n in distinct):
        return ""
    return prefix


def shorten_label(text: str, max_chars: int = SHORT_LABEL_CHARS) -> str:
    """Cut ``text`` to one line of at most ``max_chars`` characters, ending in "…" if cut."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(ELLIPSIS)].rstrip("_ ") + ELLIPSIS


def build_contrast_labels(contrasts: Sequence[Tuple[Optional[str], str]],
                          max_chars: int = SHORT_LABEL_CHARS) -> List[ContrastLabel]:
    """Short, unique, one-line tick labels for heatmap columns.

    Args:
        contrasts: ``(accession, contrast_name)`` pairs in column order.

    The accession is not part of the label: the heatmap shows it once above
    each dataset's block of columns. Within a dataset, the leading words that
    all of its contrast names share are dropped, then the rest is cut to
    ``max_chars`` with an ellipsis. A label already used in the plot gets the
    lowest free " (n)" suffix. ``display`` is HTML-escaped; ``full`` keeps the
    whole name for the hover and the key table.
    """
    by_acc: Dict[str, List[str]] = {}
    for acc, name in contrasts:
        by_acc.setdefault(str(acc or ""), []).append(name)
    prefixes = {acc: _shared_leading_words(names) for acc, names in by_acc.items()}

    labels: List[ContrastLabel] = []
    used: Set[str] = set()
    for acc, name in contrasts:
        acc_key = str(acc or "")
        prefix = prefixes[acc_key]
        short = shorten_label(name[len(prefix):] if prefix else name, max_chars)
        candidate = short
        n = 2
        while candidate in used:
            suffix = f" ({n})"
            candidate = shorten_label(short, max_chars - len(suffix)) + suffix
            n += 1
        used.add(candidate)
        labels.append(ContrastLabel(accession=acc_key, full=name,
                                    display=html.escape(candidate, quote=False)))
    return labels


def _common_label_prefix(names: Sequence[str]) -> str:
    """Longest prefix shared by all names, cut back to the last '_' or ' '."""
    if len(names) < 2:
        return ""
    prefix = os.path.commonprefix(list(names))
    cut = max(prefix.rfind("_"), prefix.rfind(" "))
    prefix = prefix[: cut + 1] if cut >= 0 else ""
    # Keep at least some text of every name after the prefix
    if any(len(n) <= len(prefix) for n in names):
        return ""
    return prefix


def build_group_labels(groups: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], str]:
    """Readable, unique x-axis labels for sample groups.

    Args:
        groups: ``(accession, group_name)`` pairs. Each pair is one sample group.

    Group names are often long free text that differ only after a shared
    prefix (for example 22 GSE295568 groups all starting
    "Placenta_CT27_Human trophoblast stem cells_"). Per dataset, the shared
    prefix is dropped (shown as a leading ellipsis), the rest is wrapped onto
    ``GROUP_MAX_LINES`` lines and HTML-escaped. Every distinct pair gets a
    distinct label, so a cut label can never merge two groups into one box.
    The full group name stays available for hover text.
    """
    pairs = list(dict.fromkeys((str(a), str(g)) for a, g in groups))
    by_acc: Dict[str, List[str]] = {}
    for acc, grp in pairs:
        by_acc.setdefault(acc, []).append(grp)
    prefixes = {acc: _common_label_prefix(names) for acc, names in by_acc.items()}

    labels: Dict[Tuple[str, str], str] = {}
    used: Set[str] = set()
    for acc, grp in pairs:
        prefix = prefixes[acc]
        rest = grp[len(prefix):] if prefix else grp
        display = wrap_contrast_label(rest, line_chars=GROUP_LINE_CHARS, max_lines=GROUP_MAX_LINES)
        if prefix:
            display = ELLIPSIS + display
        candidate = display
        n = 2
        while candidate in used:
            candidate = f"{display} ({n})"
            n += 1
        used.add(candidate)
        labels[(acc, grp)] = candidate
    return labels


def _accession_blocks(column_accessions: Sequence[str]) -> List[Tuple[str, int, int]]:
    """Runs of equal accessions as (accession, first index, last index)."""
    blocks: List[Tuple[str, int, int]] = []
    for i, acc in enumerate(column_accessions):
        if blocks and blocks[-1][0] == acc and blocks[-1][2] == i - 1:
            blocks[-1] = (acc, blocks[-1][1], i)
        else:
            blocks.append((acc, i, i))
    return blocks


def wrap_hover_text(text: str, width: int = HOVER_WRAP_CHARS,
                    max_chars: Optional[int] = None) -> str:
    """Wrap text for a Plotly hover box, optionally cutting it with an ellipsis.

    Lines are HTML-escaped after wrapping, so the text cannot inject markup.
    """
    text = "" if text is None else str(text)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip() + ELLIPSIS
    if not text:
        return ""
    lines = textwrap.wrap(text, width=width, break_long_words=True,
                          break_on_hyphens=False) or [text]
    return "<br>".join(html.escape(line, quote=False) for line in lines)


class ResultsIntegrator:
    """
    Integrates and visualizes results from multiple RNA-seq analyses.
    """

    def __init__(self,
                 results_dir: str,
                 output_dir: str = None,
                 pvalue_threshold: float = 0.05,
                 lfc_threshold: float = 1.0,
                 top_n_genes: int = 50):
        """
        Initialize the ResultsIntegrator.

        Parameters:
        -----------
        results_dir : str
            Directory containing analysis results
        output_dir : str, optional
            Directory where integrated results will be saved
        pvalue_threshold : float
            Adjusted p-value (BH) threshold for considering genes as differentially expressed
        lfc_threshold : float
            Log fold change threshold for considering genes as differentially expressed
        top_n_genes : int
            Number of top genes to include in visualizations by default
        """
        self.results_dir = os.path.abspath(results_dir)
        if not os.path.isdir(self.results_dir):
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

        # Store requested output directory but don't create it yet
        self._output_dir_requested = output_dir
        self.output_dir = None

        # Set threshold parameters
        self.pvalue_threshold = pvalue_threshold
        self.lfc_threshold = lfc_threshold
        self.top_n_genes = top_n_genes

        # Initialize containers for data
        self.analysis_info = {}   # Analysis metadata
        self.deg_data = {}        # DEG data for each contrast
        self.cpm_data = {}        # CPM data for each analysis
        self.contrast_info = {}   # Information about contrasts, keyed by name
        # Contrast information keyed by (analysis_id, contrast_name). Two
        # datasets can use the same contrast name with different meanings.
        self.contrast_info_by_analysis: Dict[Tuple[str, str], Dict] = {}
        self.deg_file_paths: Dict[str, Dict[str, str]] = {}  # analysis_id -> contrast_id -> DEG.csv

        logger.info(f"Initialized ResultsIntegrator with results_dir: {self.results_dir}")

    def _first_existing_file(self, paths: List[str]) -> Optional[str]:
        """Return the first path that exists from a list of possible locations."""
        for p in paths:
            if os.path.isfile(p):
                return p
        return None

    def _count_unique_samples(self, analysis_folder: str) -> int:
        """Count unique samples for an analysis using edger_analysis_samples.csv."""
        sample_file = self._first_existing_file([
            os.path.join(analysis_folder, "metadata", "edger_analysis_samples.csv"),
            os.path.join(analysis_folder, "edger_analysis_samples.csv"),
            os.path.join(self.results_dir, "edger_analysis_samples.csv"),
            os.path.join(os.path.dirname(self.results_dir), "edger_analysis_samples.csv"),
        ])
        if not sample_file:
            return 0
        try:
            tmp = pd.read_csv(sample_file, nrows=0)
            first_col = tmp.columns[0]
            has_index = first_col == "" or first_col.startswith("Unnamed")
            df = pd.read_csv(sample_file, index_col=0 if has_index else None)
            if "GSM" in df.columns:
                return df["GSM"].nunique()
            if "geo_accession" in df.columns:
                return df["geo_accession"].nunique()
        except Exception as e:
            logger.warning(f"Error counting samples from {sample_file}: {e}")
        return 0

    def _parse_dataset_information(self, text: str) -> Dict[str, str]:
        """Extract title, summary and design sections from dataset_information string."""
        info = {"title": "", "summary": "", "design": ""}
        if not text:
            return info
        title_match = re.search(r"Title:\s*(.*?)(?=\n\s*\n|\n*Summary:|\n*Overall Design:|$)", text, re.DOTALL | re.IGNORECASE)
        summary_match = re.search(r"Summary:\s*(.*?)(?=\n\s*\n|\n*Overall Design:|$)", text, re.DOTALL | re.IGNORECASE)
        design_match = re.search(r"Overall Design:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if title_match:
            info["title"] = title_match.group(1).strip()
        if summary_match:
            info["summary"] = summary_match.group(1).strip()
        if design_match:
            info["design"] = design_match.group(1).strip()
        return info

    def find_analysis_folders(self) -> List[str]:
        """Find all analysis folders in the results directory."""
        # Look for directories that contain RNAseqAnalysis folders
        analysis_folders = []

        # Check if the results directory itself contains RNAseqAnalysis
        rnaseq_dir = os.path.join(self.results_dir, "RNAseqAnalysis")
        if os.path.isdir(rnaseq_dir):
            analysis_folders.append(self.results_dir)
            logger.info(f"Found analysis folder: {self.results_dir}")

        # If no analysis folders found in the current directory, look for subdirectories
        if not analysis_folders:
            for item in os.listdir(self.results_dir):
                item_path = os.path.join(self.results_dir, item)

                # Check if it's a directory
                if os.path.isdir(item_path):
                    # Check if it contains an RNAseqAnalysis subdirectory
                    rnaseq_dir = os.path.join(item_path, "RNAseqAnalysis")
                    if os.path.isdir(rnaseq_dir):
                        analysis_folders.append(item_path)
                        logger.info(f"Found analysis folder: {item_path}")

        logger.info(f"Found {len(analysis_folders)} analysis folders")
        return analysis_folders

    def load_analysis_info(self) -> Dict[str, Dict]:
        """
        Load analysis information from JSON files in analysis folders.

        Returns:
        --------
        Dictionary mapping analysis IDs to their metadata
        """
        analysis_folders = self.find_analysis_folders()
        analysis_info = {}

        for folder in analysis_folders:
            analysis_id = os.path.basename(folder)
            # Check both old and new locations for analysis_info.json
            info_file_locations = [
                os.path.join(folder, "metadata", "analysis_info.json"),  # New location
                os.path.join(folder, "analysis_info.json")  # Old location for backward compatibility
            ]

            info_file = None
            for location in info_file_locations:
                if os.path.isfile(location):
                    info_file = location
                    break

            if info_file:
                try:
                    with open(info_file, 'r') as f:
                        info = json.load(f)

                    # Skip datasets with invalid unique_groups
                    if info.get('unique_groups') is None:
                        logger.warning(f"Skipping {analysis_id}: unique_groups is None")
                        continue

                    # Derive additional fields for convenience
                    info['number_of_contrasts'] = len(info.get('contrasts', []))
                    info['number_of_samples'] = self._count_unique_samples(folder)
                    analysis_info[analysis_id] = info
                    logger.info(f"Loaded analysis info for {analysis_id}")
                except Exception as e:
                    logger.warning(f"Error loading analysis info from {info_file}: {str(e)}")

        self.analysis_info = analysis_info
        logger.info(f"Loaded analysis info for {len(analysis_info)} analyses")
        return analysis_info

    def find_data_files(self) -> Tuple[Dict, Dict]:
        """
        Find all DEG and CPM files across all analyses.
        Only includes files from analyses that were successful.

        Returns:
        --------
        Tuple of (deg_files, cpm_files) dictionaries
        """
        analysis_folders = self.find_analysis_folders()
        deg_files = {}
        cpm_files = {}

        for folder in analysis_folders:
            analysis_id = os.path.basename(folder)

            # Check if analysis was successful before including files
            if analysis_id in self.analysis_info:
                analysis_success = self.analysis_info[analysis_id].get("analysis_success", False)
                if not analysis_success:
                    logger.info(f"Skipping {analysis_id} - analysis not successful")
                    continue
            else:
                logger.warning(f"No analysis info found for {analysis_id}, skipping")
                continue

            # Find DEG files (in contrast subdirectories)
            deg_files[analysis_id] = {}
            rnaseq_dir = os.path.join(folder, "RNAseqAnalysis")

            # Look for contrast subdirectories
            for item in os.listdir(rnaseq_dir):
                contrast_dir = os.path.join(rnaseq_dir, item)
                if os.path.isdir(contrast_dir) and item != "logs":
                    # Check for DEG.csv file
                    deg_file = os.path.join(contrast_dir, "DEG.csv")
                    if os.path.isfile(deg_file):
                        deg_files[analysis_id][item] = deg_file

            # Find CPM file
            cpm_file = os.path.join(rnaseq_dir, "CPM.csv")
            if os.path.isfile(cpm_file):
                cpm_files[analysis_id] = cpm_file

        logger.info(f"Found DEG files for {len(deg_files)} analyses")
        logger.info(f"Found CPM files for {len(cpm_files)} analyses")

        return deg_files, cpm_files

    def load_data(self):
        """
        Load all DEG and CPM files into memory.
        """
        self.dataset_info = {}  # Store dataset_info.txt content
        # Always load analysis info first
        self.load_analysis_info()

        # Find data files
        deg_files, cpm_files = self.find_data_files()

        # Load contrast information if available
        self.contrast_info = {}
        self.contrast_info_by_analysis = {}

        # Try to load contrasts from each analysis directory as well as global locations
        analysis_folders = self.find_analysis_folders()

        # First try to load from analysis-specific directories
        for analysis_folder in analysis_folders:
            analysis_id = os.path.basename(analysis_folder)
            for contrasts_file in [
                os.path.join(analysis_folder, "metadata", "contrasts.csv"),  # New primary location
                os.path.join(analysis_folder, "contrasts.csv")  # Old location for backward compatibility
            ]:
                if os.path.isfile(contrasts_file):
                    try:
                        contrasts_df = pd.read_csv(contrasts_file)
                        for _, row in contrasts_df.iterrows():
                            if 'name' in row and 'description' in row:
                                contrast_key = row['name']
                                # Store the extracted sections
                                entry = {
                                    'name': row['name'],  # Store the original name
                                    'description': row['description'],
                                    'expression': row.get('expression', '')
                                }
                                self.contrast_info[contrast_key] = entry
                                self.contrast_info_by_analysis[(analysis_id, str(contrast_key))] = entry
                        logger.info(f"Loaded contrasts from {contrasts_file}")
                    except Exception as e:
                        logger.warning(f"Error loading contrasts file {contrasts_file}: {str(e)}")

        # Then try global locations as fallback
        for contrasts_file in [
            os.path.join(self.results_dir, "metadata", "contrasts.csv"),
            os.path.join(self.results_dir, "contrasts.csv"),
            os.path.join(os.path.dirname(self.results_dir), "contrasts.csv")
        ]:
            if os.path.isfile(contrasts_file):
                try:
                    contrasts_df = pd.read_csv(contrasts_file)
                    for _, row in contrasts_df.iterrows():
                        if 'name' in row and 'description' in row:
                            contrast_key = row['name']
                            if contrast_key not in self.contrast_info:
                                self.contrast_info[contrast_key] = {
                                    'description': row['description'],
                                    'expression': row.get('expression', '')
                                }
                    logger.info(f"Loaded contrasts from global file {contrasts_file}")
                except Exception as e:
                    logger.warning(f"Error loading contrasts file {contrasts_file}: {str(e)}")

        logger.info(f"Loaded information for {len(self.contrast_info)} total contrasts")

        # Create consistent contrast names to handle duplicates
        self._create_consistent_contrast_names()

        # Load DEG data
        self.deg_file_paths = deg_files
        for analysis_id, contrasts in deg_files.items():
            self.deg_data[analysis_id] = {}

            for contrast_id, deg_file in contrasts.items():
                try:
                    df = pd.read_csv(deg_file)

                    # Ensure gene column is correctly identified
                    gene_col = None
                    for col in ['Gene', 'gene', 'SYMBOL', 'symbol', 'gene_name', 'gene_id']:
                        if col in df.columns:
                            gene_col = col
                            break

                    if gene_col is None:
                        # If no specific gene column found, assume first column
                        gene_col = df.columns[0]
                        logger.warning(f"No standard gene column found in {deg_file}, using {gene_col}")

                    # Standardize column name to 'Gene'
                    df = df.rename(columns={gene_col: 'Gene'})

                    # Store in dictionary
                    self.deg_data[analysis_id][contrast_id] = df
                    logger.debug(f"Loaded DEG data for {analysis_id}/{contrast_id} with {len(df)} genes")

                except Exception as e:
                    logger.error(f"Error loading DEG file {deg_file}: {str(e)}")

        # Load CPM data
        for analysis_id, cpm_file in cpm_files.items():
            try:
                df = pd.read_csv(cpm_file)

                # Ensure gene column is correctly identified
                gene_col = None
                for col in ['Gene', 'gene', 'SYMBOL', 'symbol', 'gene_name', 'gene_id']:
                    if col in df.columns:
                        gene_col = col
                        break

                if gene_col is None:
                    # If no specific gene column found, assume first column
                    gene_col = df.columns[0]
                    logger.warning(f"No standard gene column found in {cpm_file}, using {gene_col}")

                # Standardize column name to 'Gene'
                df = df.rename(columns={gene_col: 'Gene'})

                # Store in dictionary
                self.cpm_data[analysis_id] = df
                logger.debug(f"Loaded CPM data for {analysis_id} with {len(df)} genes and {len(df.columns)-1} samples")

            except Exception as e:
                logger.error(f"Error loading CPM file {cpm_file}: {str(e)}")

        # Load dataset_info.txt files if available
        self._load_dataset_info()

        # Debug logging to show actual DEG dataset names
        deg_dataset_names = list(self.deg_data.keys())
        logger.info(f"Data loading complete: {len(self.deg_data)} DEG datasets ({deg_dataset_names}), {len(self.cpm_data)} CPM datasets")

    def identify_important_genes(self,
                               top_frequent: int = 20,
                               p_value_threshold: float = None,
                               lfc_threshold: float = None) -> Set[str]:
        """
        Identify important genes across all analyses based on frequency of differential expression.

        Focuses on genes that are commonly differentially expressed across multiple contrasts,
        indicating consistent biological significance.

        Parameters:
        -----------
        top_frequent : int
            Number of top frequently occurring genes to select
        p_value_threshold : float, optional
            Adjusted p-value (BH) threshold for significance (uses instance default if None)
        lfc_threshold : float, optional
            Log fold change threshold for significance (uses instance default if None)

        Returns:
        --------
        Set of important gene identifiers
        """
        important_genes = set()

        # Dictionary to track gene occurrences across contrasts
        genes_by_occurrence = {}

        # Dictionary to track genes and their log fold changes in each contrast
        genes_by_contrast = {}

        # Use provided thresholds or fall back to instance defaults
        p_thresh = p_value_threshold if p_value_threshold is not None else self.pvalue_threshold
        lfc_thresh = lfc_threshold if lfc_threshold is not None else self.lfc_threshold

        # Collect significant genes and their properties
        logger.info(f"Identifying important genes across {len(self.deg_data)} analyses")
        for analysis_id, contrasts in self.deg_data.items():
            logger.info(f"Processing {len(contrasts)} contrasts in {analysis_id}")
            for contrast_id, df in contrasts.items():
                contrast_key = f"{analysis_id}_{contrast_id}"
                logger.info(f"Evaluating contrast: {contrast_key} with {len(df)} genes")

                # Skip if Gene column doesn't exist
                if 'Gene' not in df.columns:
                    logger.warning(f"No 'Gene' column in {contrast_key}")
                    continue

                # Adjusted p-value (BH) only; raises if the file lacks adj.P.Val
                p_value_col = get_adjusted_p_column(df, self.deg_source(analysis_id, contrast_id))

                # Use exact column name from the DEG.csv file
                lfc_col = 'logFC' if 'logFC' in df.columns else None

                if lfc_col is None:
                    logger.warning(f"Missing required columns in {contrast_key}. Available columns: {df.columns.tolist()}")
                    continue

                logger.info(f"Using columns for {contrast_key}: adjusted p-value={p_value_col}, logFC={lfc_col}")

                # Filter significant genes
                sig_df = df[(df[p_value_col] < p_thresh) &
                           (abs(df[lfc_col]) > lfc_thresh)]

                # Store significant genes for this contrast with their fold changes
                genes_by_contrast[contrast_key] = {}
                for _, row in sig_df.iterrows():
                    gene = row['Gene']
                    # Store the gene with its fold change
                    genes_by_contrast[contrast_key][gene] = abs(row[lfc_col])

                # Count occurrences of each gene in significant results
                for gene in sig_df['Gene']:
                    if gene not in genes_by_occurrence:
                        genes_by_occurrence[gene] = 1
                    else:
                        genes_by_occurrence[gene] += 1

        # 1. Get the top frequently occurring genes
        sorted_by_occurrence = sorted(genes_by_occurrence.items(), key=lambda x: x[1], reverse=True)
        top_frequent_genes = [gene for gene, _ in sorted_by_occurrence[:top_frequent]]
        important_genes.update(top_frequent_genes)

        logger.info(f"Selected {len(top_frequent_genes)} top frequently occurring genes")
        logger.info(f"Identified {len(top_frequent_genes)} important genes in total")
        return set(top_frequent_genes)

    def _ensure_output_dir(self):
        """Create output_dir only on first use."""
        if self.output_dir is None:  # not created yet
            if self._output_dir_requested:
                self.output_dir = os.path.abspath(self._output_dir_requested)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_dir = os.path.join(self.results_dir, f"integrated_results_{timestamp}")

            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Output will be saved to: {self.output_dir}")

        return self.output_dir

    def create_lfc_heatmap(self,
                         genes: List[str] = None,
                         contrasts: List[Tuple[str, str]] = None,
                         output_file: str = None,
                         p_value_threshold: float = None,
                         lfc_threshold: float = None,
                         hide_empty_rows_cols: bool = False,
                         font_size: int = 12,
                         show_grid_lines: bool = True,
                         grid_opacity: float = 0.3,
                         cluster_genes: bool = True) -> go.Figure:
        """
        Create an interactive heatmap of log fold changes for selected genes across contrasts.

        Parameters:
        -----------
        genes : List[str], optional
            List of genes to include in the heatmap. If None, important genes are identified.
        contrasts : List[Tuple[str, str]], optional
            List of (analysis_id, contrast_id) tuples to include. If None, all contrasts are used.
        output_file : str, optional
            Path to save the interactive HTML output. If None, no file is saved (useful for Streamlit).
        p_value_threshold : float, optional
            Adjusted p-value (BH) threshold for significance filtering. If None, uses instance default.
        lfc_threshold : float, optional
            Log fold change threshold for significance filtering. If None, uses instance default.
        hide_empty_rows_cols : bool, optional
            If True, hide genes/contrasts where no significant values exist.
        font_size : int, optional
            Font size for labels and annotations
        show_grid_lines : bool, optional
            Whether to show grid lines
        grid_opacity : float, optional
            Opacity of grid lines

        Returns:
        --------
        plotly.graph_objects.Figure
        """
        # Use provided thresholds or fall back to instance defaults
        p_thresh = p_value_threshold if p_value_threshold is not None else self.pvalue_threshold
        lfc_thresh = lfc_threshold if lfc_threshold is not None else self.lfc_threshold

        # If genes not provided, identify important genes
        if genes is None:
            try:
                genes = list(self.identify_important_genes(
                    top_frequent=20
                ))
                if not genes:
                    logger.warning("No important genes identified")
                    return None
                logger.info(f"Automatically identified {len(genes)} important genes")
            except Exception as e:
                logger.error(f"Error identifying important genes: {str(e)}")
                return None

        # Prepare data for heatmap
        heatmap_data = []
        logger.info(f"Creating LFC heatmap with {len(genes)} genes and {len(contrasts) if contrasts else 'all'} contrasts")

        # If contrasts not specified, use all
        if contrasts is None:
            contrasts = []
            for analysis_id, contrast_dict in self.deg_data.items():
                for contrast_id in contrast_dict.keys():
                    contrasts.append((analysis_id, contrast_id))
            logger.info(f"No contrasts specified, using all {len(contrasts)} contrasts")

        def simplify_contrast_labels(pairs: List[Tuple[str, str]]) -> List[ContrastLabel]:
            """Full name (hover) and wrapped, unique display label (axis) per contrast."""
            named = []
            for analysis_id, contrast_id in pairs:
                entry = self.get_contrast_entry(analysis_id, contrast_id) or {}
                name = entry.get('name') or contrast_id
                accession = self.analysis_info.get(analysis_id, {}).get('accession', analysis_id)
                named.append((accession, name))
            return build_contrast_labels(named)

        # Internal column keys, plus full and display labels in the same order
        contrast_labels = [f"{a_id}_{c_id}" for a_id, c_id in contrasts]
        label_objs = simplify_contrast_labels(contrasts)
        simplified_labels = [lab.display for lab in label_objs]
        full_labels = [lab.full for lab in label_objs]
        logger.info(f"Using {len(contrast_labels)} contrast labels: {', '.join(simplified_labels[:5])}{' ...' if len(simplified_labels) > 5 else ''}")

        # What the hover says for each (gene, column): cells set to 0 are either
        # not significant or absent from the contrast's DEG table, and "0.00"
        # would misreport both.
        NOT_IN_RESULTS = "not in this contrast's results"
        cell_hover: Dict[Tuple[str, str], str] = {}

        # Extract log fold change values for each gene across contrasts
        for gene in genes:
            row = {'Gene': gene}

            for (analysis_id, contrast_id), contrast_label in zip(contrasts, contrast_labels):
                if analysis_id in self.deg_data and contrast_id in self.deg_data[analysis_id]:
                    df = self.deg_data[analysis_id][contrast_id]

                    if 'Gene' not in df.columns:
                        logger.warning(f"No 'Gene' column found in {analysis_id}/{contrast_id}")
                        row[contrast_label] = 0  # Skip this contrast
                        cell_hover[(gene, contrast_label)] = NOT_IN_RESULTS
                        continue

                    # Find log fold change column
                    lfc_col = None
                    for col in ['logFC', 'log2FoldChange', 'log2FC', 'LogFC']:
                        if col in df.columns:
                            lfc_col = col
                            break

                    if lfc_col is None:
                        logger.warning(f"No log fold change column found in {analysis_id}/{contrast_id}")
                        row[contrast_label] = 0  # Skip this contrast
                        cell_hover[(gene, contrast_label)] = NOT_IN_RESULTS
                        continue

                    # Adjusted p-value (BH) only; raises if the file lacks adj.P.Val
                    p_value_col = get_adjusted_p_column(df, self.deg_source(analysis_id, contrast_id))

                    # Find the gene and get its logFC, but set to 0 if not significant
                    gene_row = df[df['Gene'] == gene]
                    if not gene_row.empty:
                        p_value = gene_row.iloc[0][p_value_col]
                        lfc_value = gene_row.iloc[0][lfc_col]
                        abs_lfc = abs(lfc_value)
                        if p_value >= p_thresh or abs_lfc <= lfc_thresh:
                            row[contrast_label] = 0  # Not significant, set to white
                            cell_hover[(gene, contrast_label)] = (
                                f"{lfc_value:.2f} (not significant: {ADJUSTED_P_LABEL} "
                                f"{p_value:.2g}; shown as 0)"
                            )
                        else:
                            row[contrast_label] = lfc_value
                            cell_hover[(gene, contrast_label)] = f"{lfc_value:.2f}"
                    else:
                        # Gene not found in this contrast: neutral colour, honest hover
                        row[contrast_label] = 0
                        cell_hover[(gene, contrast_label)] = NOT_IN_RESULTS
                else:
                    logger.debug(f"Contrast {analysis_id}/{contrast_id} not found in DEG data")
                    row[contrast_label] = 0  # Skip this contrast
                    cell_hover[(gene, contrast_label)] = NOT_IN_RESULTS

            # Only add the row if there's at least one logFC value
            if len(row) > 1:  # More than just the Gene column
                heatmap_data.append(row)
            else:
                logger.warning(f"No logFC values found for gene {gene}, skipping")

        # Convert to DataFrame
        heatmap_df = pd.DataFrame(heatmap_data)

        # Log the shape of the data
        if len(heatmap_data) > 0:
            logger.info(f"Heatmap data shape: {len(heatmap_df)} rows (genes) × {len(heatmap_df.columns)} columns (including Gene column)")
            if len(heatmap_df.columns) > 1:
                logger.info(f"Heatmap columns: {', '.join(heatmap_df.columns[:10])}{' ...' if len(heatmap_df.columns) > 10 else ''}")

        if len(heatmap_df) == 0:
            logger.warning("No data for heatmap visualization - no matching genes found. Try selecting more genes or adjusting thresholds.")
            return None
        elif len(heatmap_df.columns) <= 1:
            logger.warning("Not enough data for heatmap visualization - no contrast data found")
            return None

        # Count non-zero values to ensure we have meaningful data
        if 'Gene' in heatmap_df.columns:
            heatmap_df_values = heatmap_df.drop(columns=['Gene'])
        else:
            heatmap_df_values = heatmap_df

        if heatmap_df_values.shape[1] == 0:
            logger.warning("No contrast columns found in heatmap data. Please select at least one valid contrast.")
            return None

        # Count non-zero entries
        non_zero_count = (heatmap_df_values != 0).sum().sum()
        total_cells = heatmap_df_values.size
        non_zero_percent = (non_zero_count / total_cells) * 100 if total_cells > 0 else 0

        logger.info(f"Heatmap contains {non_zero_count}/{total_cells} non-zero values ({non_zero_percent:.1f}%)")

        # Ensure we have at least some non-zero values
        if non_zero_count == 0:
            logger.warning("No non-zero values found in heatmap data - all genes may be non-significant")
            # Continue anyway, but log a warning

        # Prepare for heatmap - set Gene as index and fill NAs
        if 'Gene' in heatmap_df.columns:
            heatmap_df = heatmap_df.set_index('Gene')
        heatmap_df = heatmap_df.fillna(0)

        # Initialize tracking for filtered items
        self.last_heatmap_filtered_genes = []
        self.last_heatmap_filtered_contrasts = []

        # Remove empty rows/columns if requested
        if hide_empty_rows_cols:
            # Track genes that will be removed (rows where all values are 0)
            non_zero_rows = (heatmap_df != 0).any(axis=1)
            filtered_genes = heatmap_df[~non_zero_rows].index.tolist()
            self.last_heatmap_filtered_genes = filtered_genes

            # Track contrasts that will be removed (columns where all values are 0)
            non_zero_cols = (heatmap_df != 0).any(axis=0)
            filtered_contrast_labels = heatmap_df.loc[:, ~non_zero_cols].columns.tolist()

            # Convert filtered contrast labels to readable format
            filtered_contrasts = []
            try:
                for label in filtered_contrast_labels:
                    # Find the corresponding contrast in our contrasts list
                    for i, (analysis_id, contrast_id) in enumerate(contrasts):
                        expected_label = f"{analysis_id}_{contrast_id}"
                        if expected_label == label:
                            # Get accession for display
                            accession = self.analysis_info.get(analysis_id, {}).get('accession', analysis_id)
                            readable_name = f"{accession}: {contrast_id}"
                            filtered_contrasts.append(readable_name)
                            break
                self.last_heatmap_filtered_contrasts = filtered_contrasts
            except Exception as e:
                logger.warning(f"Error creating readable contrast names for filtered items: {e}")
                self.last_heatmap_filtered_contrasts = filtered_contrast_labels

            # Apply the filtering
            heatmap_df = heatmap_df[non_zero_rows]
            heatmap_df = heatmap_df.loc[:, non_zero_cols]

            # Update contrasts list to match remaining columns
            if len(heatmap_df.columns) < len(contrasts):
                remaining_contrast_indices = [i for i, label in enumerate(contrast_labels) if label in heatmap_df.columns]
                contrasts = [contrasts[i] for i in remaining_contrast_indices]
                # Recompute contrast labels after filtering to keep everything in sync
                # (labels stay those built for the full set, so the axis matches
                # what the user selected and numbering stays stable)
                simplified_labels = [simplified_labels[i] for i in remaining_contrast_indices]
                full_labels = [full_labels[i] for i in remaining_contrast_indices]
                contrast_labels = [f"{a_id}_{c_id}" for a_id, c_id in contrasts]

            logger.info(f"After filtering empty rows/columns: {len(heatmap_df)} genes × {len(heatmap_df.columns)} contrasts")
            logger.info(f"Filtered out {len(filtered_genes)} genes and {len(filtered_contrasts)} contrasts with no significant values")

        # Check if we still have data after filtering
        if len(heatmap_df) == 0:
            logger.warning("No genes remain after filtering empty rows")
            return None
        elif len(heatmap_df.columns) == 0:
            logger.warning("No contrasts remain after filtering empty columns")
            return None

        # Perform hierarchical clustering on both axes
        # 1. Cluster genes (rows) unless explicitly disabled
        if not cluster_genes:
            clustered_genes = heatmap_df.index.tolist()
        else:
            if len(heatmap_df) > 1:  # Only cluster if we have multiple genes
                try:
                    # Only cluster if we have non-zero values
                    if non_zero_count > 0:
                        gene_linkage = sch.linkage(pdist(heatmap_df.values), method='complete')
                        gene_order = sch.leaves_list(gene_linkage)
                        clustered_genes = heatmap_df.index[gene_order].tolist()
                    else:
                        # Skip clustering if all values are zero
                        logger.warning("Skipping gene clustering due to lack of non-zero values")
                        clustered_genes = heatmap_df.index.tolist()
                except Exception as e:
                    logger.warning(f"Error clustering genes: {str(e)}. Using original order.")
                    clustered_genes = heatmap_df.index.tolist()
            else:
                clustered_genes = heatmap_df.index.tolist()

        # 2. Cluster contrasts (columns). With several datasets, each dataset
        # keeps one contiguous block of columns (clustered within the block),
        # so its accession can be shown once above the block.
        label_to_accession = {
            f"{a_id}_{c_id}": self.analysis_info.get(a_id, {}).get('accession', a_id)
            for a_id, c_id in contrasts
        }

        def _cluster_columns(columns: List[str]) -> List[str]:
            if len(columns) < 2:
                return list(columns)
            block = heatmap_df.loc[:, columns].to_numpy(dtype=float)
            if not (block != 0).any():
                logger.warning("Skipping contrast clustering due to lack of non-zero values")
                return list(columns)
            try:
                order = sch.leaves_list(sch.linkage(pdist(block.T), method='complete'))
            except ValueError as e:
                logger.warning(f"Error clustering contrasts: {e}. Using original order.")
                return list(columns)
            return [columns[i] for i in order]

        block_accessions = list(dict.fromkeys(label_to_accession[c] for c in heatmap_df.columns))
        if len(block_accessions) > 1:
            clustered_contrasts = []
            for acc in block_accessions:
                clustered_contrasts.extend(_cluster_columns(
                    [c for c in heatmap_df.columns if label_to_accession[c] == acc]))
        else:
            clustered_contrasts = _cluster_columns(heatmap_df.columns.tolist())

        # Reorder the DataFrame according to clustering
        heatmap_df = heatmap_df.loc[clustered_genes, clustered_contrasts]
        logger.info(f"Final heatmap dimensions: {len(clustered_genes)} genes × {len(clustered_contrasts)} contrasts")

        # Store last heatmap data and context for reproducibility/download
        try:
            # Keep a copy of the DataFrame and relevant context
            self.last_heatmap_df = heatmap_df.copy()
            self.last_heatmap_context = {
                'genes': genes,
                'contrasts': contrasts,
                'clustered_genes': clustered_genes,
                'clustered_contrasts': clustered_contrasts,
                'simplified_labels': simplified_labels,
                'full_labels': full_labels,
                'contrast_labels': contrast_labels,
                'column_accessions': [label_to_accession[c] for c in clustered_contrasts],
                'p_value_threshold': p_thresh,
                'lfc_threshold': lfc_thresh,
            }
        except Exception:
            logger.debug("Could not store last heatmap context for reproducibility")

        # Create heatmap using plotly
        max_abs_value = max(abs(heatmap_df.values.min()), abs(heatmap_df.values.max()))
        if max_abs_value == 0:
            max_abs_value = 1.0  # Avoid division by zero
            logger.warning("All values in heatmap are zero - using default color scale range of -1 to 1")

        # Compute dynamic sizing to better accommodate long gene lists and labels
        try:
            max_gene_label_len = max((len(str(g)) for g in heatmap_df.index), default=10)
        except Exception:
            max_gene_label_len = 10

        # Row height scales with font_size (approx px per row)
        row_height = max(18, int(font_size * 1.8))
        dynamic_height = int(140 + len(heatmap_df) * row_height)
        # Allow taller figures when many genes, but cap to avoid extreme sizes
        height_px = max(650, min(4500, dynamic_height))

        # Left margin scales with gene label length and font size; automargin
        # (set below) grows it further if the tick labels need more room.
        left_margin = max(80, min(520, int(40 + max_gene_label_len * font_size * 0.7)))
        # Bottom margin for one-line labels at HEATMAP_TICK_ANGLE (automargin can grow it)
        bottom_margin = int(40 + SHORT_LABEL_CHARS * font_size * 0.6 * abs(math.sin(math.radians(HEATMAP_TICK_ANGLE))))

        # Create the heatmap
        fig = px.imshow(
            heatmap_df,
            color_continuous_scale='RdBu_r',  # Red-Blue color scale, reversed to match biology convention
            labels=dict(x="Contrast", y="Gene", color="Log2FC"),
            title=(
                "Differential Expression Heatmap (Log2 Fold Change)"
                f"<br><sup>Coloured cells: {ADJUSTED_P_LABEL} < {p_thresh} and |Log2FC| > {lfc_thresh}</sup>"
            ),
            aspect="auto",  # Maintain aspect ratio based on data
            height=height_px,
            # Minimum readable width: each contrast column needs room for up to
            # three vertical label lines. The tab shows the chart in a scrolling
            # container, so a narrow page scrolls instead of squeezing columns.
            width=max(700, left_margin + len(heatmap_df.columns) * HEATMAP_MIN_COL_PX + 160),
            zmin=-max_abs_value,  # Symmetrical color scale
            zmax=max_abs_value,
            color_continuous_midpoint=0  # Ensure white is at 0
        )

        # Improve layout
        fig.update_layout(
            margin=dict(l=left_margin, r=20, t=80, b=bottom_margin),
            coloraxis_colorbar=dict(title="Log2FC"),
            font_family="Inter, sans-serif",
            font_size=font_size
        )

        # Let both axes reserve the space their tick labels need
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)

        # Configure grid lines
        if show_grid_lines:
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor=f"rgba(200,200,200,{grid_opacity})")
            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor=f"rgba(200,200,200,{grid_opacity})")
        else:
            fig.update_yaxes(showgrid=False)
            fig.update_xaxes(showgrid=False)

        # We'll create custom hover data with descriptions in the update_traces call below

        # Create a 3D array with gene logFC, contrast description, contrast formula, accession info, and original contrast name
        # First dimension: rows (genes)
        # Second dimension: columns (contrasts)
        # Third dimension: data types (0: contrast description, 1: accession, 2: original contrast name, 3: contrast formula, 4: Log2FC hover text)
        hover_info = np.empty((*heatmap_df.shape, 5), dtype=object)

        # Keep track of original contrast names for both hover and axis labels
        original_contrast_names = []

        # Create mapping from contrast labels to original indices for clustered order
        label_to_index = {f"{a_id}_{c_id}": idx for idx, (a_id, c_id) in enumerate(contrasts)}

        # Fill with contrast descriptions and accession info using clustered order
        for i, gene in enumerate(heatmap_df.index):
            for j, clustered_col in enumerate(heatmap_df.columns):
                # Map clustered column back to original contrast index
                original_idx = label_to_index[clustered_col]
                analysis_id, contrast_id = contrasts[original_idx]

                # Get contrast description using the helper method
                description = self._get_contrast_description(analysis_id, contrast_id)

                # Get accession ID
                accession = self.analysis_info.get(analysis_id, {}).get('accession', analysis_id)

                # Full (unwrapped, untruncated) contrast name for the hover
                full_name = full_labels[original_idx]
                if i == 0:  # Only collect once per contrast
                    original_contrast_names.append(full_name)

                # Contrast formula/expression for this dataset's contrast
                entry = self.get_contrast_entry(analysis_id, contrast_id) or {}
                expression = entry.get('expression', '')
                if not isinstance(expression, str):
                    expression = ''  # NaN from a blank CSV cell

                # Store description, accession, full contrast name, and expression
                hover_info[i, j, 0] = wrap_hover_text(description)
                hover_info[i, j, 1] = html.escape(str(accession), quote=False)
                hover_info[i, j, 2] = html.escape(full_name, quote=False)
                hover_info[i, j, 3] = wrap_hover_text(expression, max_chars=FORMULA_MAX_CHARS)
                hover_info[i, j, 4] = html.escape(
                    cell_hover.get((gene, clustered_col), NOT_IN_RESULTS), quote=False
                )

        # Update hover template with conditional formatting, accession, and description
        hover_template = (
            "<b>Gene:</b> %{y}<br>" +
            "<b>Contrast:</b> %{customdata[2]}<br>" +  # Full contrast name
            "<b>Log2FC:</b> %{customdata[4]}<br>" +
            "<b>Accession:</b> %{customdata[1]}<br>" +
            "<b>Description:</b> %{customdata[0]}<br>" +
            "<b>Formula:</b> %{customdata[3]}<br>" +
            "<extra></extra>"  # Hide secondary info
        )
        fig.update_traces(
            hovertemplate=hover_template,
            customdata=hover_info,
            # Use custom color for hover
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
        )

        # Update x-axis with simplified contrast names
        if simplified_labels:
            # Log simplified contrast names for debugging
            logger.debug(f"Original contrast IDs: {[cid for _, cid in contrasts]}")
            logger.debug(f"Using simplified contrast names: {simplified_labels}")

            # Ensure we have labels for all contrasts
            assert len(simplified_labels) == len(contrasts), "Mismatch between number of contrasts and simplified names"

            # Build new tick positions after clustering (need to map to clustered order)
            # Create mapping from contrast label to simplified label
            label_mapping = dict(zip(contrast_labels, simplified_labels))
            clustered_simplified_labels = [label_mapping[col] for col in clustered_contrasts]

            # Build new tick positions after clustering
            tick_vals = list(range(len(clustered_simplified_labels)))
            fig.update_xaxes(
                tickmode="array",
                tickvals=tick_vals,
                ticktext=clustered_simplified_labels,
                # Short one-line labels at an angle: about 28 px per column is
                # enough for neighbours not to overlap
                tickangle=HEATMAP_TICK_ANGLE,
                title=dict(text="Biological contrast", standoff=12)
            )

            # Dataset accession once per block of columns, with separators
            column_accessions = [label_to_accession[c] for c in clustered_contrasts]
            blocks = _accession_blocks(column_accessions)
            if len(blocks) > 1:
                for acc, start, end in blocks:
                    fig.add_annotation(
                        x=(start + end) / 2, y=1.0, xref="x", yref="paper",
                        text=f"<b>{html.escape(acc, quote=False)}</b>",
                        showarrow=False, yanchor="bottom", yshift=4,
                        font=dict(size=font_size),
                    )
                    fig.add_shape(type="line", xref="x", yref="paper",
                                  x0=start - 0.45, x1=end + 0.45, y0=1.0, y1=1.0,
                                  line=dict(color="#555", width=1.5))
                for _, start, _ in blocks[1:]:
                    fig.add_shape(type="line", xref="x", yref="paper",
                                  x0=start - 0.5, x1=start - 0.5, y0=0, y1=1,
                                  line=dict(color="#333", width=1.5))
                fig.update_layout(margin=dict(t=110))

            # Numbered key: one row per column, for the table under the chart,
            # its CSV download and the reproduction package
            label_to_pair = {f"{a_id}_{c_id}": (a_id, c_id) for a_id, c_id in contrasts}
            full_mapping = dict(zip(contrast_labels, full_labels))
            column_key = []
            for number, col in enumerate(clustered_contrasts, start=1):
                a_id, c_id = label_to_pair[col]
                description = self._get_contrast_description(a_id, c_id)
                column_key.append({
                    "#": number,
                    "Label": html.unescape(label_mapping[col]),
                    "Dataset": label_to_accession[col],
                    "Contrast": full_mapping[col],
                    "Description": description if isinstance(description, str) else "",
                })
            fig.update_layout(meta={"column_key": column_key,
                                    "dataset_blocks": [list(b) for b in blocks]})
            self.last_heatmap_context['column_key'] = column_key
            self.last_heatmap_context['dataset_blocks'] = [list(b) for b in blocks]

        # Save to file if specified
        if output_file is not None:
            self._ensure_output_dir()
            fig.write_html(output_file)
            logger.info(f"Saved LFC heatmap to {output_file}")

        return fig

    def get_last_heatmap_filtered_info(self) -> Dict[str, List[str]]:
        """
        Get information about genes and contrasts that were filtered out in the last heatmap.

        Returns:
            Dictionary with 'genes' and 'contrasts' keys containing lists of filtered items
        """
        return {
            'genes': getattr(self, 'last_heatmap_filtered_genes', []),
            'contrasts': getattr(self, 'last_heatmap_filtered_contrasts', [])
        }



    def create_expression_plots(self,
                              genes: List[str],
                              plot_type: str = 'violin',
                              analyses: List[str] = None,
                              output_file: str = None,
                              hide_x_labels: bool = False,
                              page_number: int = 1,
                              facet_font_size: int = 10,
                              lock_y_axis: bool = False,
                              show_raw_points: bool = True,
                              legend_position: str = "Bottom",
                              show_grid_lines: bool = True,
                              grid_opacity: float = 0.3,
                              selected_groups: List[str] = None,
                              plots_per_row: int = 2,
                              show_box: bool = True) -> go.Figure:
        """
        Create interactive expression plots for selected genes across samples.

        Parameters:
        -----------
        genes : List[str]
            List of genes to plot expression for
        plot_type : str
            Type of plot: 'violin', 'box', or 'both'
        analyses : List[str], optional
            List of analysis IDs to include. If None, all analyses are used.
        output_file : str, optional
            Path to save the interactive HTML output. If None, no file is saved (useful for Streamlit).
        facet_font_size : int, optional
            Font size for facet titles
        lock_y_axis : bool, optional
            Whether to use same y-axis range across all facets
        show_raw_points : bool, optional
            Whether to show individual data points
        legend_position : str, optional
            Legend position: "Bottom", "Right", or "Top"
        show_grid_lines : bool, optional
            Whether to show grid lines
        grid_opacity : float, optional
            Opacity of grid lines
        selected_groups : List[str], optional
            List of sample groups to include. If None, all groups are used.
        plots_per_row : int, optional
            Number of plots per row in the layout (default: 2)
        show_box : bool, optional
            Whether to show boxplot within violin plot (default: True)

        Returns:
        --------
        plotly.graph_objects.Figure
        """
        if not self.cpm_data:
            logger.error("No CPM data loaded. Cannot create expression plots.")
            return None

        # If analyses not specified, use all
        if analyses is None:
            analyses = list(self.cpm_data.keys())

        # Create a tidy (long) dataframe for all expression data
        all_long_data = []

        # Save original gene list and validate
        gene_list = genes.copy() if genes is not None else []
        if not gene_list:
            logger.warning("No genes provided for expression plots")
            return None

        # Use all genes (no pagination)
        genes = gene_list

        # Process each analysis and build tidy data
        for analysis_id in analyses:
            if analysis_id not in self.cpm_data:
                logger.warning(f"No CPM data for analysis {analysis_id}")
                continue

            # Get CPM data for this analysis
            cpm_df = self.cpm_data[analysis_id].copy()

            # Skip if Gene column doesn't exist
            if 'Gene' not in cpm_df.columns:
                logger.warning(f"No 'Gene' column in CPM data for {analysis_id}")
                continue

            # Filter to only the genes we're working with
            cpm_filtered = cpm_df[cpm_df['Gene'].isin(genes)]

            if cpm_filtered.empty:
                logger.warning(f"No matching genes found in CPM data for {analysis_id}")
                continue

            # Get the sample groups from the edger_analysis_samples.csv file
            sample_to_group = {}

            # Helper function to find first existing file
            def first_existing(paths):
                for path in paths:
                    if os.path.isfile(path):
                        return path
                return None

            # Look inside the current analysis directory first
            analysis_dir = os.path.join(self.results_dir, analysis_id)
            sample_file_locations = [
                os.path.join(analysis_dir, "metadata", "edger_analysis_samples.csv"),  # New primary location
                os.path.join(analysis_dir, "edger_analysis_samples.csv"),  # Old location for backward compatibility
                os.path.join(self.results_dir, "edger_analysis_samples.csv"),
                os.path.join(os.path.dirname(self.results_dir), "edger_analysis_samples.csv")
            ]

            edger_samples_file = first_existing(sample_file_locations)
            if edger_samples_file:
                logger.info(f"Found sample mapping file at {edger_samples_file}")

            if edger_samples_file:
                try:
                    # First try loading the file to check its structure
                    sample_mapping_df = pd.read_csv(edger_samples_file, nrows=0)

                    # If the first column has no name or starts with 'Unnamed', it's probably an index
                    first_col = sample_mapping_df.columns[0]
                    has_index = first_col == '' or first_col.startswith('Unnamed')

                    # Now load the full file with appropriate index_col setting
                    sample_mapping_df = pd.read_csv(edger_samples_file, index_col=0 if has_index else None)
                    logger.info(f"Loaded sample mapping with columns: {', '.join(sample_mapping_df.columns)}")

                    # Find the group column from analysis_info
                    group_col = None
                    if self.analysis_info and analysis_id in self.analysis_info and 'merged_column' in self.analysis_info[analysis_id]:
                        group_col = self.analysis_info[analysis_id]['merged_column']
                        logger.info(f"Using merged column '{group_col}' from analysis_info")

                    if group_col and group_col in sample_mapping_df.columns:
                        # Look for a 'Sample1', 'Sample2', etc. naming pattern in CPM columns
                        cpm_columns = self.cpm_data[analysis_id].columns.tolist()
                        sample_cols = [col for col in cpm_columns if col.startswith('Sample') and col != 'Gene']

                        if sample_cols:
                            # Create a mapping from sample name to group
                            for i in range(len(sample_mapping_df)):
                                if i < len(sample_cols):  # Ensure we don't go out of bounds
                                    sample_key = sample_cols[i]
                                    sample_to_group[sample_key] = sample_mapping_df.iloc[i][group_col]
                            logger.info(f"Created mapping for {len(sample_to_group)} samples using Sample# pattern")
                        else:
                            # Fall back to index-based mapping
                            for i in range(len(sample_mapping_df)):
                                # Use sample index (i) as the key (starting with Sample1)
                                sample_key = f"Sample{i+1}"
                                sample_to_group[sample_key] = sample_mapping_df.iloc[i][group_col]
                            logger.info(f"Created mapping for {len(sample_to_group)} samples using index")
                    else:
                        logger.warning(f"No group column found in sample mapping file")
                except Exception as e:
                    logger.warning(f"Error loading sample mapping file: {str(e)}")
            else:
                logger.warning("Could not find edger_analysis_samples.csv file. Using default sample names.")

            # Convert to long format
            melted_df = cpm_filtered.melt(
                id_vars='Gene',
                var_name='Sample',
                value_name='Expression'
            )

            # Add analysis ID
            melted_df['Analysis'] = analysis_id

            # Create dataset display name with title for hover info
            accession = self.analysis_info.get(analysis_id, {}).get('accession', analysis_id)
            dataset_title = ""
            if hasattr(self, 'dataset_info') and analysis_id in self.dataset_info:
                title = self.dataset_info[analysis_id].get('title', '')
                if title and title.startswith('Title:'):
                    title = title[6:].strip()
                dataset_title = title

            if dataset_title:
                melted_df['DatasetDisplay'] = f"{accession} - {dataset_title}"
            else:
                melted_df['DatasetDisplay'] = accession

            # Map samples to groups if available
            if sample_to_group:
                melted_df['Group'] = melted_df['Sample'].map(sample_to_group)
                # Fill any missing mappings
                melted_df['Group'] = melted_df['Group'].fillna(melted_df['Sample'])
            else:
                # Use analysis ID as group if no mapping available
                melted_df['Group'] = analysis_id

            # Filter by selected groups if specified
            if selected_groups:
                melted_df = melted_df[melted_df['Group'].isin(selected_groups)]
                if melted_df.empty:
                    logger.warning(f"No samples found for selected groups in analysis {analysis_id}")
                    continue

            # Create custom hover text with all the information we want (after Group is assigned)
            melted_df['HoverText'] = (
                "<b>Dataset:</b> " + melted_df['DatasetDisplay'] + "<br>" +
                "<b>Group:</b> " + melted_df['Group'].astype(str) + "<br>" +
                "<b>Expression:</b> " + melted_df['Expression'].round(2).astype(str)
            )

            # Append to all data
            all_long_data.append(melted_df)

        # Combine all data
        if not all_long_data:
            logger.warning("No expression data found for specified genes and analyses")
            return None

        # Concatenate all data
        plot_df = pd.concat(all_long_data, ignore_index=True)

        if plot_df.empty:
            logger.warning("No expression data after filtering")
            return None

        # Number of genes actually found in data
        genes_found = plot_df['Gene'].nunique()

        # Determine facet column wrapping based on plots_per_row
        facet_col_wrap = min(plots_per_row, genes_found)

        # Calculate rows for height
        n_rows = math.ceil(genes_found / facet_col_wrap)

        # Apply log transformation for better visualization if needed
        if plot_df['Expression'].min() < 0 or (plot_df['Expression'].min() >= 0 and plot_df['Expression'].max() <= 30):
            # Data appears to be already log-transformed
            plot_df['LogExpression'] = plot_df['Expression']
            y_axis_title = "Log<sub>2</sub>(CPM)"
        else:
            # Apply log2 transformation (after adding 1 to avoid log(0))
            plot_df['LogExpression'] = np.log2(plot_df['Expression'] + 1)
            y_axis_title = "Log<sub>2</sub>(CPM+1)"

        facet_col_spacing = 0.03

        # Build combined x-axis labels in the format: "GSEXXXX - group <group>"
        # We include a dedicated column so we can show clear, per-dataset group labels on the x-axis
        if 'Accession' not in plot_df.columns:
            # Each melted_df chunk already had a single accession; ensure a column exists for clarity
            # If not present, infer from DatasetDisplay when possible; otherwise fall back to Analysis
            # Prefer using analysis_info accession directly where available
            plot_df['Accession'] = plot_df.apply(
                lambda r: self.analysis_info.get(r['Analysis'], {}).get('accession', r['Analysis'])
                if isinstance(r.get('Analysis', None), str) else r.get('Analysis', ''), axis=1
            )

        # x-axis label per (dataset, group). The label is also the box key, so
        # it must be unique: a cut label shared by two groups would pool their
        # samples into one box. build_group_labels drops the prefix shared by a
        # dataset's groups, wraps the rest and numbers any remaining clash.
        group_pairs = list(zip(plot_df['Accession'].astype(str), plot_df['Group'].astype(str)))
        group_labels = build_group_labels(group_pairs)
        plot_df['XLabel'] = [group_labels[pair] for pair in group_pairs]
        x_order = list(dict.fromkeys(plot_df['XLabel']))
        self.last_expression_group_labels = dict(group_labels)

        # Vertical tick labels: reserve their height between facet rows so the
        # next row's gene title and the labels do not overlap.
        tick_font_px = 10
        longest_line = max(
            (len(html.unescape(line)) for lab in x_order for line in lab.split("<br>")),
            default=10,
        )
        label_px = int(longest_line * tick_font_px * 0.62) + 10
        if n_rows <= 2:
            plot_px_per_row = 420
        elif n_rows <= 6:
            plot_px_per_row = 380
        else:
            plot_px_per_row = 340
        height_per_row = plot_px_per_row + label_px + 60
        total_height = height_per_row * max(1, n_rows)
        if n_rows > 1:
            facet_row_spacing = min((label_px + 60) / total_height, 0.95 / (n_rows - 1))
        else:
            facet_row_spacing = 0.0

        # Build an explicit color map for datasets to ensure colors are embedded (important for static export)
        try:
            import plotly.express as px
            palette = px.colors.qualitative.Plotly
        except Exception:
            palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            ]

        unique_datasets = list(pd.unique(plot_df['DatasetDisplay']))
        color_discrete_map = {ds: palette[i % len(palette)] for i, ds in enumerate(unique_datasets)}
        # diagnostics removed

        # Create boxplot (explicit color mapping + all other settings)
        fig = px.box(
            plot_df,
            x='XLabel',
            y='LogExpression',
            color='DatasetDisplay',
            facet_col='Gene',
            facet_col_wrap=facet_col_wrap,
            custom_data=['HoverText'],
            points="all",
            title="Gene Expression Across Samples",
            labels={'LogExpression': y_axis_title, 'DatasetDisplay': 'Dataset'},
            facet_row_spacing=facet_row_spacing,
            facet_col_spacing=facet_col_spacing,
            color_discrete_map=color_discrete_map,
            category_orders={'XLabel': x_order},
        )

        # Each facet needs ~44 px per group for vertical multi-line labels; the
        # tab shows the chart in a horizontally scrolling container when the
        # page is narrower than this width.
        facet_px = max(500, 44 * len(x_order) + 80)
        fig.update_layout(
            height=total_height,
            width=facet_px * facet_col_wrap,
            legend_title_text="Dataset",
            margin=dict(l=40, r=20, t=80, b=6),
            font_family="Inter, sans-serif",
        )

        # Configure legend position
        if legend_position == "Bottom":
            fig.update_layout(legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center", yanchor="top"))
        elif legend_position == "Right":
            fig.update_layout(legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left", yanchor="middle"))
        elif legend_position == "Top":
            fig.update_layout(legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center", yanchor="bottom"))
        else:  # Default to bottom
            fig.update_layout(legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center", yanchor="top"))

        # Configure x-axis visibility, labeling, and gridlines
        if hide_x_labels:
            # Hide when explicitly requested (e.g., tight layouts in static exports)
            fig.update_xaxes(visible=False)
        else:
            # Show axis tick labels on all facets with minimal standoff, align close to plot, and no axis title
            fig.update_xaxes(visible=True, showticklabels=True, tickangle=-90,
                             ticklabelstandoff=0, automargin=True, title_text="",
                             tickfont=dict(size=tick_font_px))

        # Add light dotted vertical gridlines to separate groups across the full plotting area
        if show_grid_lines:
            fig.update_xaxes(showgrid=True,
                             gridwidth=0.5,
                             gridcolor=f"rgba(200,200,200,{grid_opacity})",
                             griddash='dot')
        else:
            fig.update_xaxes(showgrid=False)

        # Configure grid lines and ensure y-axis ticks are shown on each plot
        if show_grid_lines:
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor=f"rgba(200,200,200,{grid_opacity})", showticklabels=True, title_text="")
        else:
            fig.update_yaxes(showgrid=False, showticklabels=True, title_text="")

        # Add y-axis labels to the left-most facet of each row
        for r in range(1, n_rows + 1):
            fig.update_yaxes(
                title_text=y_axis_title,
                row=r,
                col=1,
                title_standoff=6
            )

        # Configure points visibility and transparency for boxplots
        if show_raw_points:
            fig.update_traces(marker=dict(size=4, opacity=0.5), jitter=0,
                              selector=dict(type='box'))
            fig.update_traces(pointpos=0, width=0.7, selector=dict(type='box'))

        # Use the custom hover text we created in the data
        fig.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>"
        )

        # Update facet formatting (remove "Gene=" prefix, enlarge, and nudge upward to reduce overlap)
        fig.for_each_annotation(lambda a: a.update(
            text=f"{a.text.split('=')[1]}",
            font=dict(size=int(facet_font_size * 2), color="#333", family="Inter, sans-serif"),
            yshift=14
        ))

        # Ensure background and template are explicitly defined (avoids theme-dependent exports)
        fig.update_layout(template='plotly', paper_bgcolor='white', plot_bgcolor='white')

        # Explicitly set per-trace colors to embed into the figure (helps Kaleido PDF)
        colored_traces = 0
        for tr in fig.data:
            try:
                if hasattr(tr, 'type') and tr.type == 'box':
                    # Match by trace name to DatasetDisplay
                    name = getattr(tr, 'name', None)
                    if name in color_discrete_map:
                        color_val = color_discrete_map[name]
                        if hasattr(tr, 'marker'):
                            tr.marker.color = color_val
                        if hasattr(tr, 'line'):
                            tr.line.color = color_val
                        colored_traces += 1
            except Exception as _:
                pass
        # diagnostics removed

        # Configure y-axis scaling - each plot gets its own scale unless lock_y_axis is True
        if lock_y_axis and not plot_df.empty:
            global_min = plot_df['LogExpression'].min()
            global_max = plot_df['LogExpression'].max()
            # Add some padding
            padding = (global_max - global_min) * 0.05
            fig.update_yaxes(range=[global_min - padding, global_max + padding])
        else:
            # Each facet gets its own y-axis scale (default behavior we want to ensure)
            fig.update_yaxes(matches=None)

        # Prefer y-axis ticks every 2 units for readability (avoid 5-unit steps)
        fig.update_yaxes(dtick=2)

        # No pagination annotations; all selected genes are shown in one figure

        # Save to file if specified
        if output_file is not None:
            self._ensure_output_dir()
            fig.write_html(output_file)
            logger.info(f"Saved expression plots to {output_file}")

        return fig

    def deg_source(self, analysis_id: str, contrast_id: str) -> str:
        """Name the DEG file behind a contrast, for error messages."""
        path = self.deg_file_paths.get(analysis_id, {}).get(contrast_id)
        return f"DEG file {path}" if path else f"DEG table for {analysis_id}/{contrast_id}"

    def get_contrast_entry(self, analysis_id: str, contrast_id: str) -> Optional[Dict]:
        """Return the contrast record for one dataset's contrast, or None.

        Looks up ``(analysis_id, contrast_id)`` first, so two datasets that use
        the same contrast name keep their own description and formula. Falls
        back to the name-only table for results without per-dataset metadata.
        """
        entry = self.contrast_info_by_analysis.get((analysis_id, contrast_id))
        if entry is not None:
            return entry
        return self.contrast_info.get(contrast_id)

    def _get_contrast_description(self, analysis_id: str, contrast_id: str) -> str:
        """Get the description for a contrast from the loaded contrast info."""
        keyed = self.contrast_info_by_analysis.get((analysis_id, contrast_id))
        if keyed is not None and isinstance(keyed.get('description'), str) and keyed['description']:
            return keyed['description']

        # Try exact match first
        if hasattr(self, "contrast_info") and contrast_id in self.contrast_info:
            return self.contrast_info[contrast_id].get('description', f"Contrast: {contrast_id}")

        # Try simplified pattern matching (without dataset/accession prefix)
        if "_" in contrast_id:
            # Try parts after potential prefixes
            parts = contrast_id.split("_")
            for i in range(1, len(parts)):
                possible_id = "_".join(parts[i:])
                if hasattr(self, "contrast_info") and possible_id in self.contrast_info:
                    return self.contrast_info[possible_id].get('description', f"Contrast: {possible_id}")

        # Try to look in an analysis-specific contrasts file (check multiple locations)
        for contrasts_file in [
            os.path.join(self.results_dir, analysis_id, "metadata", "contrasts.csv"),  # New primary location
            os.path.join(self.results_dir, analysis_id, "contrasts.csv"),  # Old location for backward compatibility
            os.path.join(self.results_dir, "contrasts.csv")  # Also check global location
        ]:
            if os.path.isfile(contrasts_file):
                try:
                    contrasts_df = pd.read_csv(contrasts_file)
                    # Try exact match on 'name' field
                    match = contrasts_df[contrasts_df['name'] == contrast_id]
                    if not match.empty:
                        return match.iloc[0].get('description', f"Contrast: {contrast_id}")

                    # Try parts of the contrast ID for partial matches
                    if "_" in contrast_id:
                        parts = contrast_id.split("_")
                        for i in range(1, len(parts)):
                            possible_id = "_".join(parts[i:])
                            match = contrasts_df[contrasts_df['name'] == possible_id]
                            if not match.empty:
                                return match.iloc[0].get('description', f"Contrast: {possible_id}")
                except Exception as e:
                    logger.debug(f"Error loading contrasts from {contrasts_file}: {str(e)}")

        return f"Contrast: {contrast_id}"

    def _load_dataset_info(self):
        """
        Load dataset_info.txt files for each analysis and extract title, summary, and design sections.

        The expected format of dataset_info.txt is:
        - Title at the beginning
        - Summary in the middle section
        - Overall design at the end (usually starts with "Overall design:")

        The function attempts to parse these sections using both regex patterns and fallback line-by-line parsing.
        """
        import re  # Ensure re is imported for pattern matching
        analysis_folders = self.find_analysis_folders()

        for analysis_folder in analysis_folders:
            analysis_id = os.path.basename(analysis_folder)
            # Prefer dataset_information field from analysis_info.json if available
            if analysis_id in self.analysis_info and 'dataset_information' in self.analysis_info[analysis_id]:
                info_text = self.analysis_info[analysis_id].get('dataset_information', '')
                self.dataset_info[analysis_id] = self._parse_dataset_information(info_text)
                continue
            # Look for dataset_info.txt in various possible locations (though it's now stored in analysis_info.json)
            for info_path in [
                os.path.join(analysis_folder, "metadata", "dataset_info.txt"),
                os.path.join(analysis_folder, "dataset_info.txt"),
                os.path.join(self.results_dir, "metadata", "dataset_info.txt")
            ]:
                if os.path.isfile(info_path):
                    try:
                        with open(info_path, 'r') as f:
                            content = f.read()

                        # Initialize dataset info with empty values
                        self.dataset_info[analysis_id] = {
                            "title": "",
                            "summary": "",
                            "design": ""
                        }

                        # Try to extract sections using common patterns
                        import re

                        # Match common section patterns
                        title_match = re.search(r'^(.*?)(?=\n\s*\n|\n*Overall design:|\n*Summary:|\Z)', content, re.DOTALL | re.IGNORECASE)
                        design_match = re.search(r'(?:Overall design:)(.*?)(?=\n\s*\n|\Z)', content, re.DOTALL | re.IGNORECASE)

                        # If we can't find the design section explicitly, look for it at the end
                        if not design_match:
                            design_match = re.search(r'(?:Overall design\s*[-:])?(.*?)(?=\Z)', content, re.DOTALL | re.IGNORECASE)

                        # Extract the summary (everything between title and design)
                        if title_match:
                            title_end = title_match.end()
                            design_start = design_match.start() if design_match else len(content)
                            summary_text = content[title_end:design_start].strip()

                            # Clean up summary text - remove any "Summary:" prefix
                            summary_text = re.sub(r'^Summary:\s*', '', summary_text, flags=re.IGNORECASE)

                            # Store the extracted sections (remove "Title:" prefix if present)
                            title_text = title_match.group(1).strip()
                            if title_text.startswith("Title:"):
                                title_text = title_text[6:].strip()
                            self.dataset_info[analysis_id]["title"] = title_text
                            self.dataset_info[analysis_id]["summary"] = summary_text

                            if design_match:
                                design_text = design_match.group(1).strip()
                                self.dataset_info[analysis_id]["design"] = design_text
                        else:
                            # Fallback to simple parsing if regex approach fails
                            lines = content.split('\n')
                            title_lines = []
                            summary_lines = []
                            design_lines = []

                            section = "title"
                            for line in lines:
                                if section == "title":
                                    if line.strip() == "":
                                        section = "summary"
                                    else:
                                        title_lines.append(line)
                                elif section == "summary":
                                    if (line.lower().startswith("overall design:") or
                                        line.lower().startswith("overall design") or
                                        line.lower().strip() == "overall design"):
                                        section = "design"
                                        # Add the "Overall design:" part to design
                                        design_lines.append(line)
                                    else:
                                        summary_lines.append(line)
                                else:  # design section
                                    design_lines.append(line)

                            # Store the extracted sections
                            title_text = '\n'.join(title_lines).strip()
                            if title_text.startswith("Title:"):
                                title_text = title_text[6:].strip()
                            self.dataset_info[analysis_id]["title"] = title_text
                            self.dataset_info[analysis_id]["summary"] = '\n'.join(summary_lines).strip()
                            self.dataset_info[analysis_id]["design"] = '\n'.join(design_lines).strip()

                        logger.info(f"Loaded and parsed dataset info for {analysis_id}")
                        break
                    except Exception as e:
                        logger.warning(f"Error loading dataset info from {info_path}: {str(e)}")

        logger.info(f"Loaded dataset info for {len(self.dataset_info)} analyses")

        # Log contrast info for debugging
        if hasattr(self, "contrast_info") and self.contrast_info:
            logger.info(f"Available contrast IDs: {', '.join(self.contrast_info.keys())}")

    def _create_consistent_contrast_names(self):
        """
        Create consistent contrast names by appending dataset accession to duplicates.
        This ensures unique contrast names across all datasets while preserving original information.
        """
        # First, collect all contrast names and their associated analysis info
        contrast_occurrences = {}
        contrast_analysis_mapping = {}

        # Count occurrences of each contrast name across all analyses
        for analysis_id in self.analysis_info.keys():
            # Get contrasts for this analysis from analysis_info
            analysis_data = self.analysis_info[analysis_id]
            contrasts_list = analysis_data.get('contrasts', [])

            for contrast in contrasts_list:
                contrast_name = contrast.get('name', '')
                if contrast_name:
                    if contrast_name not in contrast_occurrences:
                        contrast_occurrences[contrast_name] = []
                    contrast_occurrences[contrast_name].append(analysis_id)

                    # Store the mapping for later use
                    key = (analysis_id, contrast_name)
                    contrast_analysis_mapping[key] = contrast

        # Create new contrast info with consistent names
        new_contrast_info = {}

        for contrast_name, analysis_ids in contrast_occurrences.items():
            if len(analysis_ids) > 1:
                # Name appears in multiple datasets - append accession
                for analysis_id in analysis_ids:
                    accession = self.analysis_info[analysis_id].get('accession', analysis_id)
                    new_name = f"{contrast_name}_{accession}"

                    # Get original contrast info
                    key = (analysis_id, contrast_name)
                    if key in contrast_analysis_mapping:
                        original_contrast = contrast_analysis_mapping[key]
                        new_contrast_info[new_name] = {
                            'name': new_name,
                            'original_name': contrast_name,
                            'analysis_id': analysis_id,
                            'accession': accession,
                            'description': original_contrast.get('description', ''),
                            'expression': original_contrast.get('expression', ''),
                            'justification': original_contrast.get('justification', '')
                        }
            else:
                # Name is unique - keep original
                analysis_id = analysis_ids[0]
                key = (analysis_id, contrast_name)
                if key in contrast_analysis_mapping:
                    original_contrast = contrast_analysis_mapping[key]
                    accession = self.analysis_info[analysis_id].get('accession', analysis_id)
                    new_contrast_info[contrast_name] = {
                        'name': contrast_name,
                        'original_name': contrast_name,
                        'analysis_id': analysis_id,
                        'accession': accession,
                        'description': original_contrast.get('description', ''),
                        'expression': original_contrast.get('expression', ''),
                        'justification': original_contrast.get('justification', '')
                    }

        # Update the contrast_info with new consistent names
        self.contrast_info.update(new_contrast_info)

        # analysis_info.json also lists each dataset's contrasts; use it for any
        # (analysis_id, contrast) pair that had no contrasts.csv row.
        for (analysis_id, contrast_name), contrast in contrast_analysis_mapping.items():
            self.contrast_info_by_analysis.setdefault((analysis_id, contrast_name), {
                'name': contrast_name,
                'description': contrast.get('description', ''),
                'expression': contrast.get('expression', ''),
            })

        logger.info(f"Created {len(new_contrast_info)} consistent contrast names")

    def create_integrated_report(self,
                               top_frequent: int = 20,
                               plot_type: str = 'violin',
                               gene_list: List[str] = None,
                               analyses: list = None,
                               max_genes: int = 100,
                               p_value_threshold: float = None,
                               lfc_threshold: float = None,
                               hide_x_labels: bool = True,
                               output_file: str = None):
        """
        Create a comprehensive integrated report with all visualizations.

        Parameters:
        -----------
        top_frequent : int
            Number of top frequently occurring genes to include
        plot_type : str
            Type of expression plot: 'violin', 'box', or 'both'
        gene_list : List[str], optional
            Specific list of genes to plot (overrides automatic selection)
        analyses : list, optional
            List of analysis IDs to include (all by default)
        max_genes : int
            Maximum number of genes to include in visualizations
        p_value_threshold : float, optional
            Override the default adjusted p-value (BH) threshold for gene selection
        lfc_threshold : float, optional
            Override the default log fold change threshold for gene selection
        hide_x_labels : bool
            Whether to hide x-axis labels in expression plots
        output_file : str, optional
            Output file path for the report (defaults to results_dir/integrated_report.html)

        Returns:
        --------
        str: Path to the generated HTML report file
        """
        # Set output file path if specified
        if output_file is not None:
            import os
            output_dir = os.path.dirname(output_file)
            if output_dir:
                self._output_dir_requested = output_dir
                self.output_dir = None

        # Now ensure the output directory exists
        self._ensure_output_dir()

        # Load data if not already done
        if not self.deg_data:
            self.load_data()

        # Identify important genes if not provided
        if gene_list is None:
            gene_list = list(self.identify_important_genes(
                top_frequent=top_frequent,
                # Use provided thresholds or default to 0.01 for automatic selection to highlight stronger signals
                p_value_threshold=p_value_threshold if p_value_threshold is not None else 0.01,
                lfc_threshold=lfc_threshold
            ))

            # Limit the number of genes to avoid performance issues
            if len(gene_list) > max_genes:
                logger.warning(f"Too many genes selected ({len(gene_list)}), limiting to {max_genes}")
                gene_list = gene_list[:max_genes]

        # Create LFC heatmap
        try:
            # Adjust max_genes for heatmap to allow for more genes
            max_heatmap_genes = min(100, len(gene_list))
            output_path = os.path.join(self.output_dir, "lfc_heatmap.html")
            lfc_heatmap = self.create_lfc_heatmap(
                genes=gene_list[:max_heatmap_genes],
                output_file=output_path
            )
        except Exception as e:
            logger.error(f"Error creating LFC heatmap: {str(e)}")
            lfc_heatmap = None

        # Create expression plots (single page with all selected genes)
        try:
            output_path = os.path.join(self.output_dir, f"expression_{plot_type}_plots.html")
            expr_plot = self.create_expression_plots(
                genes=gene_list,
                plot_type=plot_type,
                output_file=output_path,
                hide_x_labels=hide_x_labels  # Use parameter value
            )
        except Exception as e:
            logger.error(f"Error creating expression plots: {str(e)}")
            expr_plot = None

        # Generate contrast details HTML
        if not self.deg_data:
            contrast_details_html = '<p>No contrast information available.</p>'
        else:
            contrast_rows = []
            for analysis_id, contrasts in self.deg_data.items():
                for contrast_id in contrasts.keys():
                    contrast_rows.append(f'<tr><td>{analysis_id}</td><td>{contrast_id}</td><td>{self._get_contrast_description(analysis_id, contrast_id)}</td></tr>')
            contrast_details_html = f"""
                    <table>
                        <tr>
                            <th>Analysis</th>
                            <th>Contrast</th>
                            <th>Description</th>
                        </tr>
                        {''.join(contrast_rows)}
                    </table>"""

        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>UORCA Results Integration Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ border: 1px solid #ddd; border-radius: 4px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .vis-link {{ display: inline-block; background: #3498db; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; margin-top: 10px; }}
                .vis-link:hover {{ background: #2980b9; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>UORCA Results Integration Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <div class="card">
                    <h2>Visualizations</h2>
                    <p>The following interactive visualizations are available:</p>
                    <ul>
                        <li>
                            <strong>Log Fold Change Heatmap</strong>: Displays differential expression patterns across contrasts.
                            <br><a href="lfc_heatmap.html" class="vis-link" target="_blank">View Heatmap</a>
                        </li>
                        <li>
                            <strong>Gene Expression Plots</strong>: Shows expression levels (CPM) for selected genes across samples.
                            <br><a href="expression_{plot_type}_plots.html" class="vis-link" target="_blank">View Expression Plots</a>
                        </li>
                    </ul>
                </div>

                <div class="card">
                    <h2>Integration Summary</h2>
                    <p>This report integrates results from:</p>
                    <ul>
                        <li><strong>{len(self.deg_data)} datasets</strong> with a total of {sum(len(contrasts) for contrasts in self.deg_data.values())} contrasts</li>
                        <li><strong>{len(gene_list)} genes</strong> selected for visualization</li>
                    </ul>

                    <h3>Analyses Overview</h3>
                    <table>
                        <tr>
                            <th>Accession</th>
                            <th>Organism</th>
                            <th># Samples</th>
                            <th># Contrasts</th>
                        </tr>
                        {''.join(f'<tr><td>{info.get("accession", "Unknown")}</td><td>{info.get("organism", "Unknown")}</td><td>{info.get("number_of_samples", 0)}</td><td>{info.get("number_of_contrasts", 0)}</td></tr>' for analysis_id in self.cpm_data.keys() if analysis_id in self.analysis_info for info in [self.analysis_info[analysis_id]])}
                    </table>

                    <h3>Contrast Details</h3>
                    {contrast_details_html}
                </div>

                <div class="card">
                    <h2>Gene List</h2>
                    <table>
                        <tr>
                            <th>#</th>
                            <th>Gene</th>
                        </tr>
                        {''.join(f'<tr><td>{i+1}</td><td>{gene}</td></tr>' for i, gene in enumerate(gene_list))}
                    </table>
                </div>
            </div>
        </body>
        </html>
        """

        # Write HTML to file
        index_path = os.path.join(self.output_dir, "index.html")
        with open(index_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Generated index HTML report at {index_path}")
        return self.output_dir


def main():
    """
    Main entry point for the script.

    Integrates results from multiple RNA-seq analyses and creates visualizations.
    """
    parser = argparse.ArgumentParser(
        description="Integrate and visualize results from multiple UORCA RNA-seq analyses"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing analysis results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where integrated results will be saved"
    )
    parser.add_argument(
        "--pvalue_threshold",
        type=float,
        default=0.05,
        help="Adjusted p-value (BH) threshold for considering genes as differentially expressed (default: 0.05)"
    )
    parser.add_argument(
        "--lfc_threshold",
        type=float,
        default=1.0,
        help="Log fold change threshold for considering genes as differentially expressed (default: 1.0)"
    )
    parser.add_argument(
        "--top_frequent",
        type=int,
        default=20,
        help="Number of most frequently occurring genes to include (default: 20)"
    )

    parser.add_argument(
        "--max_genes",
        type=int,
        default=100,
        help="Maximum number of genes to include in visualizations (default: 100)"
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["violin", "box", "both"],
        default="violin",
        help="Type of expression plot to generate (default: violin)"
    )
    parser.add_argument(
        "--gene_list",
        type=str,
        default=None,
        help="Path to text file with specific list of genes to plot (one gene per line)"
    )

    args = parser.parse_args()

    # Initialize the integrator
    integrator = ResultsIntegrator(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        pvalue_threshold=args.pvalue_threshold,
        lfc_threshold=args.lfc_threshold,
        top_n_genes=args.top_frequent
    )

    # Load gene list if provided
    gene_list = None
    if args.gene_list:
        try:
            with open(args.gene_list, 'r') as f:
                gene_list = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(gene_list)} genes from {args.gene_list}")
        except Exception as e:
            logger.error(f"Error loading gene list from {args.gene_list}: {str(e)}")

    # Create the integrated report
    output_dir = integrator.create_integrated_report(
        top_frequent=args.top_frequent,
        plot_type=args.plot_type,
        gene_list=gene_list,
        max_genes=args.max_genes
    )

    print(f"\nResults integration complete!")
    print(f"Interactive report available at: {os.path.join(output_dir, 'index.html')}")


if __name__ == "__main__":
    main()
