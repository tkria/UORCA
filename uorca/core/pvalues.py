"""Adjusted p-value columns and labels for limma DEG tables.

limma writes the Benjamini-Hochberg adjusted p-value as ``adj.P.Val`` and the
raw p-value as ``P.Value``. Every UORCA threshold uses the adjusted one.
"""

import pandas as pd


ADJUSTED_P_COLUMN = "adj.P.Val"
RAW_P_COLUMN = "P.Value"
ADJUSTED_P_LABEL = "Adjusted p-value (BH)"
RAW_P_LABEL = "Raw p-value"
ADJUSTED_P_HELP = "Benjamini-Hochberg adjusted p-value (limma column adj.P.Val)"


class MissingAdjustedPValueError(ValueError):
    """A DEG table has no adjusted p-value column."""


def get_adjusted_p_column(df: pd.DataFrame, source: str) -> str:
    """Return the adjusted p-value column of a DEG table, or raise.

    The raw ``P.Value`` column is never used in its place: a threshold labelled
    "adjusted" must not silently filter on raw p-values.

    Args:
        df: DEG table.
        source: The file or contrast the table came from, named in the error.

    Raises:
        MissingAdjustedPValueError: if ``adj.P.Val`` is absent.
    """
    if ADJUSTED_P_COLUMN in df.columns:
        return ADJUSTED_P_COLUMN
    raise MissingAdjustedPValueError(
        f"{source} has no '{ADJUSTED_P_COLUMN}' column (Benjamini-Hochberg adjusted "
        f"p-value). UORCA does not substitute the raw '{RAW_P_COLUMN}' column. "
        f"Columns present: {list(df.columns)}"
    )
