"""
Ortholog Mapper Module for UORCA Explorer using a local CSV.

This module maps genes between species using a precomputed mammalian
orthologues CSV located at data/mammalian_orthologues.csv.
It replaces network-based lookups with fast, cached local lookups.

The CSV is not in the repository. Fetch it with ``./download_orthologues.sh``
(macOS/Linux) or ``download_orthologues.ps1`` (Windows); see data/README.md
for its source (Allen Institute GeneOrthology) and licence. When the file is
missing, the loader raises FileNotFoundError rather than returning an empty
table, so the UI cannot report "no orthologues found" by mistake.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any

import os
import pandas as pd
import streamlit as st
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the orthologues CSV (in project data directory)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
ORTHOLOG_CSV_PATH = str(_PROJECT_ROOT / "data" / "mammalian_orthologues.csv")
ORTHOLOG_DOWNLOAD_COMMAND = (
    "./download_orthologues.sh (macOS/Linux) or "
    "powershell -ExecutionPolicy Bypass -File download_orthologues.ps1 (Windows), "
    "run from the UORCA repository root"
)

# Map scientific/common names (as found in analysis_info.json) to CSV species base names
# CSV species base names correspond to the prefixes before "_Symbol" in the CSV header
SCIENTIFIC_TO_CSV_SPECIES = {
    # Core species
    "Homo sapiens": "Human",
    "Mus musculus": "Mouse",
    "Rattus norvegicus": "Rat",
    "Danio rerio": "zebrafish",
    # Common name variants that might appear
    "human": "Human",
    "mouse": "Mouse",
    "rat": "Rat",
    "zebrafish": "zebrafish",
    # Additional mammals present in the CSV
    "Felis catus": "Cat",
    "Canis lupus familiaris": "dog",
    "Sus scrofa": "Pig",
    "Oryctolagus cuniculus": "Rabbit",
    "Pan troglodytes": "Chimpanzee",
    "Gorilla gorilla": "Gorilla",
    "Macaca mulatta": "Macaque.rhesus",
    "Macaca nemestrina": "Macaque.pig.tailed",
    "Callithrix jacchus": "Marmoset",
    "Saimiri sciureus": "Squirrel.monkey",
    "Aotus nancymaae": "Owl.monkey",
    "Chlorocebus sabaeus": "African.green.monkey",
    "Aotus trivirgatus": "Owl.monkey",
    "Microcebus murinus": "Mouse.lemur",
    "Heterocephalus glaber": "Naked.mole.rat",
    "Dasypus novemcinctus": "Armadillo.Nine.banded",
    "Tupaia belangeri": "Chinese.Treeshrew",
    "Phocoena sinus": "Vaquita",
    "Mustela putorius furo": "Ferret",
    # Fallback mapping for display names that might already match CSV
    "Cat": "Cat",
    "dog": "dog",
    "Pig": "Pig",
    "Rabbit": "Rabbit",
    "Chimpanzee": "Chimpanzee",
    "Gorilla": "Gorilla",
    "Marmoset": "Marmoset",
    "Opossum": "Opossum",
    "Ferret": "Ferret",
    "Galago": "Galago",
    "Vaquita": "Vaquita",
}


def _csv_symbol_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Return mapping from CSV species base name -> symbol column name."""
    symbol_cols = [c for c in df.columns if c.endswith("_Symbol")]
    return {c.replace("_Symbol", ""): c for c in symbol_cols}


def _missing_file_message(path: str) -> str:
    return (
        f"Orthologue file not found: {path}. "
        f"Download it with {ORTHOLOG_DOWNLOAD_COMMAND}."
    )


def ortholog_file_error() -> Optional[str]:
    """Return an error message if the orthologue CSV is missing, else None.

    The tabs call this to show ``st.error`` and disable the orthologue checkbox.
    """
    if os.path.isfile(ORTHOLOG_CSV_PATH):
        return None
    return _missing_file_message(ORTHOLOG_CSV_PATH)


def _load_ortholog_df() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load the orthologue CSV and return DataFrame and species->symbol column map.

    Raises:
        FileNotFoundError: if the CSV is missing; the message names the download command.
        ValueError: if the CSV has no ``*_Symbol`` columns.

    Only successful loads are cached (``st.cache_data`` does not cache an
    exception), so downloading the file takes effect on the next call.
    """
    return _load_ortholog_csv(ORTHOLOG_CSV_PATH)


def clear_ortholog_cache() -> None:
    """Drop the cached orthologue table (for tests, or after re-downloading)."""
    _load_ortholog_csv.clear()  # type: ignore[attr-defined]  # added by st.cache_data


@st.cache_data(show_spinner=False, ttl=3600)
def _load_ortholog_csv(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(_missing_file_message(path))
    df = pd.read_csv(path)
    species_to_symbol = _csv_symbol_columns(df)
    if not species_to_symbol:
        raise ValueError(
            f"Orthologue file {path} has no '*_Symbol' columns. Re-download it with "
            f"{ORTHOLOG_DOWNLOAD_COMMAND}."
        )
    return df, species_to_symbol



def _organism_to_csv_species(organism: str) -> Optional[str]:
    """Map scientific/common organism name to CSV species base name."""
    if not organism:
        return None
    # Exact match first
    if organism in SCIENTIFIC_TO_CSV_SPECIES:
        return SCIENTIFIC_TO_CSV_SPECIES[organism]
    # Case-insensitive fallback
    key_lower = organism.lower()
    for k, v in SCIENTIFIC_TO_CSV_SPECIES.items():
        if k.lower() == key_lower:
            return v
    return None


def get_species_info(organism: str) -> Dict[str, Any]:
    """Return info containing the CSV species base and symbol column for an organism."""
    df, species_to_symbol = _load_ortholog_df()
    csv_species = _organism_to_csv_species(organism)
    if not csv_species:
        logger.warning(f"No CSV species mapping found for organism: {organism}")
        return None
    symbol_col = species_to_symbol.get(csv_species)
    if not symbol_col:
        logger.warning(f"No symbol column found in CSV for species: {csv_species}")
        return None
    return {"csv_species": csv_species, "symbol_col": symbol_col}


def get_taxid_from_organism(organism: str) -> Optional[int]:
    """Kept for compatibility; not used by CSV-based logic. Returns common taxids when known."""
    TAXIDS = {
        "Homo sapiens": 9606,
        "Mus musculus": 10090,
        "Rattus norvegicus": 10116,
        "Danio rerio": 7955,
    }
    return TAXIDS.get(organism)


def _find_row_indices_for_gene(df: pd.DataFrame, symbol_cols: List[str], gene: str) -> pd.Series:
    """Return boolean Series of rows where any selected symbol column matches gene (case-insensitive)."""
    gene_upper = str(gene).upper()
    # Build a boolean mask across provided columns
    masks = []
    for col in symbol_cols:
        # Ensure string comparison and handle NaN
        masks.append(df[col].astype(str).str.upper() == gene_upper)
    if not masks:
        return pd.Series([False] * len(df), index=df.index)
    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    return mask


def _collect_symbols_from_rows(df: pd.DataFrame, rows_mask: pd.Series, target_cols: List[str]) -> List[str]:
    """Collect unique non-null symbols from target columns for rows where rows_mask is True."""
    if rows_mask.sum() == 0 or not target_cols:
        return []
    subset = df.loc[rows_mask, target_cols]
    # Flatten and drop NA
    syms = pd.unique(subset.values.ravel("K"))
    return [s for s in syms if isinstance(s, str) and s.strip()]


def expand_genes_with_orthologs(
    input_genes: List[str],
    input_organism: str,
    target_organisms: List[str],
    return_mapping: bool = False
) -> Tuple[List[str], Optional[Dict[str, Dict[str, List[str]]]]]:
    """
    Expand gene list with orthologues using the local CSV.

    Args:
        input_genes: Input gene symbols (strings)
        input_organism: Scientific/common name of the source organism (from analysis_info.json)
        target_organisms: List of organism names present in selected datasets
        return_mapping: If True, return mapping dict {query_gene: {organism: [symbols]}}

    Returns:
        Tuple of (expanded_genes, mapping or None)
    """
    if not input_genes or not target_organisms:
        return input_genes, {} if return_mapping else None

    df, species_to_symbol = _load_ortholog_df()

    # Map organisms to CSV species base names and symbol columns
    csv_targets = []
    target_symbol_cols = []
    for org in target_organisms:
        info = get_species_info(org)
        if info:
            csv_targets.append(org)
            target_symbol_cols.append(info["symbol_col"])

    if not target_symbol_cols:
        return input_genes, {} if return_mapping else None

    # Also include input organism column for matching if present
    input_info = get_species_info(input_organism)
    match_symbol_cols = list(target_symbol_cols)
    if input_info and input_info["symbol_col"] not in match_symbol_cols:
        match_symbol_cols.append(input_info["symbol_col"])

    expanded_set = set(input_genes)
    mapping: Dict[str, Dict[str, List[str]]] = {} if return_mapping else None

    for gene in input_genes:
        rows_mask = _find_row_indices_for_gene(df, match_symbol_cols, gene)
        if rows_mask.any():
            # For each target organism, collect symbols
            if return_mapping and gene not in mapping:
                mapping[gene] = {}
            for org in csv_targets:
                org_info = get_species_info(org)
                if not org_info:
                    continue
                col = org_info["symbol_col"]
                symbols = _collect_symbols_from_rows(df, rows_mask, [col])
                # Exclude exact self-match (case-sensitive) but keep case-different
                symbols = [s for s in symbols if s != gene]
                if symbols:
                    expanded_set.update(symbols)
                    if return_mapping:
                        mapping[gene].setdefault(org, [])
                        # Deduplicate while preserving order
                        for s in symbols:
                            if s not in mapping[gene][org]:
                                mapping[gene][org].append(s)

    return list(expanded_set), mapping


def expand_genes_all_vs_all(
    input_genes: List[str],
    organisms: List[str],
    return_mapping: bool = False
) -> Tuple[List[str], Optional[Dict[str, Dict[str, List[str]]]]]:
    """
    Expand genes using all-vs-all approach based on the local CSV.

    For each input gene, find any orthologue row where the gene appears in any
    of the selected organisms' symbol columns, then collect symbols from all
    selected organisms and merge into a combined set and optional mapping.
    """
    if not input_genes or len(organisms) < 2:
        return input_genes, None

    df, species_to_symbol = _load_ortholog_df()

    # Determine which symbol columns correspond to selected organisms
    org_infos = []
    for org in organisms:
        info = get_species_info(org)
        if info:
            org_infos.append((org, info["symbol_col"]))

    symbol_cols = [c for _, c in org_infos]
    if not symbol_cols:
        return input_genes, None

    expanded_set = set(input_genes)
    combined_mapping: Optional[Dict[str, Dict[str, List[str]]]] = {} if return_mapping else None

    for gene in input_genes:
        rows_mask = _find_row_indices_for_gene(df, symbol_cols, gene)
        if not rows_mask.any():
            continue

        # For each organism, collect its symbols and update mapping
        for org, col in org_infos:
            symbols = _collect_symbols_from_rows(df, rows_mask, [col])
            # Exclude exact self-matches (but allow case-different)
            symbols = [s for s in symbols if s != gene]
            if symbols:
                expanded_set.update(symbols)
                if return_mapping:
                    combined_mapping.setdefault(gene, {}).setdefault(org, [])
                    for s in symbols:
                        if s not in combined_mapping[gene][org]:
                            combined_mapping[gene][org].append(s)

    logger.info(f"All-vs-all expansion (CSV): {len(input_genes)} -> {len(expanded_set)} genes")

    return list(expanded_set), combined_mapping


def create_ortholog_mapping_table(
    input_genes: List[str],
    mapping: Dict[str, Dict[str, List[str]]],
    organisms: List[str]
) -> pd.DataFrame:
    """
    Create a DataFrame table showing ortholog mappings.

    Args:
        input_genes: Original input gene list
        mapping: Ortholog mapping dictionary
        organisms: List of all organisms (not used in new format)

    Returns:
        DataFrame with input genes and their orthologs
    """
    data = []

    for gene in input_genes:
        row = {'Input Gene': gene}

        # Collect all unique orthologs (excluding exact self-matches)
        all_orthologs = []
        if gene in mapping:
            for org in mapping[gene]:
                for ortholog in mapping[gene][org]:
                    # Don't filter case-different orthologs (Cd74 -> CD74 is valid)
                    if ortholog != gene and ortholog not in all_orthologs:
                        all_orthologs.append(ortholog)

        # Add orthologs as numbered columns
        for i, ortholog in enumerate(all_orthologs, 1):
            row[f'Ortholog {i}'] = ortholog

        # If no orthologs found, add empty column
        if not all_orthologs:
            row['Ortholog 1'] = ''

        data.append(row)

    df = pd.DataFrame(data)
    return df.fillna('')


def create_hierarchical_gene_list(
    input_genes: List[str],
    mapping: Dict[str, Dict[str, List[str]]],
    expanded_genes: List[str]
) -> str:
    """
    Create a simple text list of all genes (one per line, no indentation).

    Args:
        input_genes: Original input gene list
        mapping: Ortholog mapping dictionary
        expanded_genes: All genes including orthologs

    Returns:
        Formatted string with one gene per line
    """
    # Collect all unique genes
    all_genes = set(input_genes)

    # Add all orthologs (excluding exact self-matches)
    for gene in input_genes:
        if gene in mapping:
            for org, orthologs in mapping[gene].items():
                for ortholog in orthologs:
                    # Don't filter case-different orthologs
                    if ortholog != gene:
                        all_genes.add(ortholog)

    # Sort alphabetically
    sorted_genes = sorted(all_genes)

    return '\n'.join(sorted_genes)


def get_ortholog_summary(
    mapping: Dict[str, Dict[str, List[str]]]
) -> str:
    """
    Generate a summary of ortholog mappings.

    Args:
        mapping: Ortholog mapping dictionary

    Returns:
        Summary string
    """
    if not mapping:
        return "No ortholog mappings found."

    total_orthologs = 0
    genes_with_orthologs = 0

    for gene, orgs in mapping.items():
        if orgs:
            has_orthologs = False
            for org, orthologs in orgs.items():
                if orthologs:
                    has_orthologs = True
                    total_orthologs += len(orthologs)
            if has_orthologs:
                genes_with_orthologs += 1

    lines = [
        f"Found orthologs for {genes_with_orthologs}/{len(mapping)} input genes",
        f"Total orthologs added: {total_orthologs}"
    ]

    return "\n".join(lines)
