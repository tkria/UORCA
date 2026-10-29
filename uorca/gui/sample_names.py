"""Join the sample columns of CPM.csv to the rows of edger_analysis_samples.csv.

RNAseq.R runs before 2026-10-29 gave tximport an unnamed file list, so DGEList()
named the CPM columns ``Sample1``..``SampleN``, in the row order of the sample
table that R read. Newer runs name the columns by GSM accession. Both forms
resolve here to one sample ID per column, so the explorer can show the GSM
accession and join groups by name instead of by position.
"""

import os
import re
from typing import List, Optional

import pandas as pd

SAMPLE_TABLE_NAME = "edger_analysis_samples.csv"

# Columns that can name a sample, in order of preference: the GSM accession, then
# the run accession. A column is used only if its values are all present and unique.
# Datasets with several runs per GSM therefore use the run accession.
# RNAseq.R applies the same rule, so both sides give the same names.
SAMPLE_ID_COLUMNS = ("geo_accession", "GSM", "SRR")

_DEFAULT_NAME = re.compile(r"^Sample(\d+)$")


class SampleTableMismatch(ValueError):
    """The CPM columns cannot be joined to the sample table without a guess."""


def find_sample_table(results_dir: str, analysis_id: str) -> Optional[str]:
    """Path of the sample table for one dataset, or None if there is none."""
    analysis_dir = os.path.join(results_dir, analysis_id)
    for path in (
        os.path.join(analysis_dir, "metadata", SAMPLE_TABLE_NAME),
        os.path.join(analysis_dir, SAMPLE_TABLE_NAME),
        os.path.join(results_dir, SAMPLE_TABLE_NAME),
        os.path.join(os.path.dirname(results_dir), SAMPLE_TABLE_NAME),
    ):
        if os.path.isfile(path):
            return path
    return None


def read_sample_table(path: str) -> pd.DataFrame:
    """Read the sample table; an unnamed first column is the pandas index."""
    first_col = str(pd.read_csv(path, nrows=0).columns[0])
    has_index = first_col == "" or first_col.startswith("Unnamed")
    return pd.read_csv(path, index_col=0 if has_index else None)


def default_column_names(n: int) -> List[str]:
    """The names DGEList() gives to n unnamed samples."""
    return [f"Sample{i + 1}" for i in range(n)]


def sample_ids(sample_df: pd.DataFrame) -> List[str]:
    """One ID per sample-table row, from the first column in SAMPLE_ID_COLUMNS
    whose values are all present and unique; else the DGEList default names."""
    for col in SAMPLE_ID_COLUMNS:
        if col not in sample_df.columns:
            continue
        ids = sample_df[col].fillna("").astype(str).str.strip()
        if (ids != "").all() and ids.is_unique:
            return ids.tolist()
    return default_column_names(len(sample_df))


def name_cpm_columns(cpm_df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
    """Return cpm_df with each sample column named by its sample ID.

    - Columns ``Sample1``..``SampleN`` (old runs) are renamed by position. This is
      safe only when N equals the number of sample-table rows, so any other N raises.
    - Columns that are already sample IDs (new runs) are returned unchanged.
    - Any other column name raises, because a join would be a guess.
    """
    cols = [c for c in cpm_df.columns if c != "Gene"]
    ids = sample_ids(sample_df)
    if cols == default_column_names(len(cols)):
        if len(cols) != len(ids):
            raise SampleTableMismatch(
                f"CPM.csv has {len(cols)} sample columns but the sample table has "
                f"{len(ids)} rows, so the columns cannot be matched to samples"
            )
        return cpm_df.rename(columns=dict(zip(cols, ids)))
    unknown = [c for c in cols if c not in set(ids)]
    if unknown:
        raise SampleTableMismatch(
            f"{len(unknown)} CPM.csv sample columns are not in the sample table: {unknown[:5]}"
        )
    return cpm_df


def sample_groups(sample_df: pd.DataFrame, group_col: str) -> pd.Series:
    """Group of each sample, indexed by sample ID."""
    return pd.Series(sample_df[group_col].tolist(), index=sample_ids(sample_df))


def sample_titles(sample_df: pd.DataFrame) -> pd.Series:
    """GEO sample title of each sample, indexed by sample ID (empty if no title column)."""
    titles = sample_df["title"].fillna("").astype(str).tolist() if "title" in sample_df.columns else [""] * len(sample_df)
    return pd.Series(titles, index=sample_ids(sample_df))
