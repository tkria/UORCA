from __future__ import annotations
import logging, os, re, subprocess, json, glob, argparse, asyncio, pathlib, datetime
from typing import List, Optional, Dict, Any, Union, Tuple, Literal

import pandas as pd
import numpy as np
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent, RunContext
from uorca.shared import AnalysisContext
from uorca.ai_provider import get_model
from uorca.shared.workflow_logging import log_tool, log_agent_tool
from unidecode import unidecode
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class MetadataContext:
    metadata_path: str
    metadata_df: Optional[pd.DataFrame] = None
    analysis_metadata_df: Optional[pd.DataFrame] = None
    merged_column: Optional[str] = None
    unique_groups: Optional[List[str]] = None
    contrast_details: Optional[Dict[str, Any]] = None


#############################
# SECTION: Metadata Parsing Agent tools
#############################
# Try reading your system prompt, otherwise use the fallback prompt
import pathlib
system_prompt_path = pathlib.Path(__file__).parent.parent / "prompts" / "analysis.txt"
try:
    system_prompt = system_prompt_path.read_text()

    # Print the first and last few lines for verification
    lines = system_prompt.split('\n')
    print(f"\n--- METADATA AGENT SYSTEM PROMPT FILE LOADED ---")
    print("First 3 lines:")
    for line in lines[:3]:
        print(f"> {line}")

    print("...")

    print("Last 3 lines:")
    for line in lines[-3:]:
        print(f"> {line}")
    print(f"--- END OF PREVIEW ({len(lines)} total lines) ---\n")

except Exception as e:
    system_prompt = """
    #### Integrated Prompt for Metadata Processing and Grouping Variable Selection

    You are provided with RNAseq metadata from different experiments. Your task is to identify the column(s) that contain biologically relevant information for differential expression analysis and to merge them into a single grouping variable if necessary. This grouping variable will be used in a single-factor analysis with edgeR/limma. In doing so, you must also evaluate each column to decide which ones provide informative biological variation and which ones should be excluded.

    General Guidelines:
    1. Focus on Biologically Relevant Information:
    • Include columns that capture sample-specific biological attributes such as tissue/disease type, genotype, or treatment conditions.
    • Exclude technical columns (e.g., sample IDs, run/experiment numbers) and those with no variation (all values identical) or with unique values that do not group samples.
    2. Merging Columns:
    • If more than one column is informative (e.g., one column for tissue type and one for treatment), merge these into a single composite grouping variable (for example, merged_analysis_group).
    • Ensure that the final grouping factor includes only information that is biologically significant for differential expression analysis.
    3. Output:
    • Return the name(s) of the final grouping column(s) and a brief explanation of your selection process and why the other columns were excluded.
    """
    print("Using fallback system prompt instead.")

load_dotenv()

metadata_agent = Agent(
    get_model(),
    deps_type=MetadataContext,
    system_prompt=system_prompt,
    model_settings={"temperature": 1}
)

# ----------------------------
# Utility function: Clean a string
# ----------------------------


def clean_string(s: str) -> str:
    """
    Normalize and clean an input string by removing non-ASCII characters, extra
    whitespace, and unwanted symbols. Additionally ensures that the resulting
    string is a valid identifier for R by prefixing it with ``"X"`` if the first
    character would otherwise be a digit. If cleaning would result in an empty
    string, returns "None" instead.
    """
    if pd.isna(s):
        logger.debug("🧹 Input is NaN, returning 'NA'")
        return "NA"
    s = str(s).strip()
    if not s or s == "-":  # Handle empty strings and dash-only entries
        logger.debug("🧹 Input is empty or just a dash, returning 'None'")
        return "None"
    s = unidecode(s)
    s = s.replace(" ", "_")
    s = re.sub(r'[^\w]', '', s)

    # Prefix with 'X' if the cleaned string begins with a digit so that it is
    # a syntactically valid name when used in R
    if re.match(r"^[0-9]", s):
        s = f"X{s}"

    # Handle case where cleaning results in an empty string
    if not s:
        logger.debug("🧹 Cleaning resulted in empty string, returning 'None'")
        return "None"

    return s

# ----------------------------
# Tool 1: Process metadata
# ----------------------------

@metadata_agent.tool
async def process_metadata(ctx: RunContext[MetadataContext]) -> dict:
    """
    Load metadata from ctx.deps.metadata_path, clean all column names and
    cell values using clean_string, and store the cleaned DataFrame in the context.

    Creates TWO versions of the metadata:
    1. Full metadata (ctx.deps.metadata_df): Retains all columns including identifiers
    2. Analysis metadata (ctx.deps.analysis_metadata_df): Specialized for column selection
       - Removes GSM, SRX, SRR columns
       - Deduplicates rows
       - Removes columns where all values are unique

    Returns information about the processed data.
    """
    try:
        logger.info("📋 Processing metadata from %s", ctx.deps.metadata_path)

        # Load metadata based on file extension
        if ctx.deps.metadata_path.endswith('.csv'):
            df = pd.read_csv(ctx.deps.metadata_path)
        elif ctx.deps.metadata_path.endswith(('.tsv', '.txt')):
            df = pd.read_csv(ctx.deps.metadata_path, sep='\t')
        else:
            df = pd.read_csv(ctx.deps.metadata_path, sep=None, engine='python')

        # Print message of metadata dimensions
        logger.info("📊 Loaded metadata with %d rows and %d columns", df.shape[0], df.shape[1])

        # Remove duplicate rows
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        if df.shape[0] < initial_rows:
            logger.info("🧹 Removed %d duplicate rows", initial_rows - df.shape[0])

        # Remove columns where all values are identical
        initial_cols = df.shape[1]
        df = df.loc[:, df.nunique() > 1]
        if df.shape[1] < initial_cols:
            logger.info("🧹 Removed %d columns with identical values", initial_cols - df.shape[1])

        # Clean column names
        new_columns = {col: clean_string(col) for col in df.columns}
        df.rename(columns=new_columns, inplace=True)
        logger.info("✅ Cleaned column names")

        # Clean each cell value
        logger.info("🧹 Cleaning cell values...")
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: clean_string(x) if pd.notna(x) else x)

        # Store the complete cleaned metadata in context (VERSION 1)
        ctx.deps.metadata_df = df.copy()
        logger.info("💾 Stored complete cleaned metadata in context")

        # Create a specialized version for column/contrast determination (VERSION 2)
        analysis_df = df.copy()

        # 1. Remove identifier columns: GSM, SRX, SRR
        id_columns = ['GSM', 'SRX', 'SRR']
        dropped_ids = [col for col in id_columns if col in analysis_df.columns]
        if dropped_ids:
            analysis_df = analysis_df.drop(columns=dropped_ids)
            logger.info("🧹 Removed identifier columns from analysis metadata: %s", dropped_ids)

        # 2. Deduplicate rows (again, after removing ID columns)
        initial_rows = analysis_df.shape[0]
        analysis_df = analysis_df.drop_duplicates()
        if analysis_df.shape[0] < initial_rows:
            logger.info("🧹 Removed %d duplicate rows from analysis metadata", initial_rows - analysis_df.shape[0])

        # 3. Remove columns where all values are unique (likely another form of ID)
        unique_cols = [col for col in analysis_df.columns
                       if analysis_df[col].nunique() == len(analysis_df)]
        if unique_cols:
            analysis_df = analysis_df.drop(columns=unique_cols)
            logger.info("🧹 Removed columns with all unique values from analysis metadata: %s", unique_cols)

        # Store the analysis-specific metadata
        ctx.deps.analysis_metadata_df = analysis_df
        logger.info("💾 Stored analysis-specific metadata (without IDs, unique columns) in context")

        # If no columns remain after cleaning, raise an error so the workflow can halt early
        if analysis_df.shape[1] == 0:
            error_msg = (
                "No valid columns remain for analysis after removing identifier and unique-value columns. "
                "This dataset cannot be analysed."
            )
            logger.error("❌ %s", error_msg)
            raise ValueError(error_msg)

        # Count unique values for each column in the analysis df
        column_stats = {}
        for col in analysis_df.columns:
            unique_vals = analysis_df[col].unique()
            column_stats[col] = {
                "unique_count": len(unique_vals),
                "values": list(unique_vals) if len(unique_vals) < 20 else list(unique_vals[:20])
            }

        summary = f"""
Metadata processed into two versions:

1. Complete metadata: {ctx.deps.metadata_df.shape[0]} rows and {ctx.deps.metadata_df.shape[1]} columns
   - All columns retained including identifiers: {', '.join(ctx.deps.metadata_df.columns)}

2. Analysis metadata: {ctx.deps.analysis_metadata_df.shape[0]} rows and {ctx.deps.analysis_metadata_df.shape[1]} columns
   - Removed ID columns: {', '.join(dropped_ids) if dropped_ids else 'None'}
   - Removed unique-value columns: {', '.join(unique_cols) if unique_cols else 'None'}
   - Remaining columns: {', '.join(ctx.deps.analysis_metadata_df.columns)}
"""
        logger.info("✅ %s", summary)

        # Return information about the processed data
        return {
            "message": summary,
            "complete_metadata_shape": ctx.deps.metadata_df.shape,
            "analysis_metadata_shape": ctx.deps.analysis_metadata_df.shape,
            "columns_for_analysis": list(ctx.deps.analysis_metadata_df.columns),
            "column_stats": column_stats
        }
    except Exception as e:
        error_msg = f"Error processing metadata: {str(e)}"
        logger.error("❌ %s", error_msg, exc_info=True)
        return {"message": error_msg, "error": True}


# ----------------------------
# Tool 2: Merge Analysis Columns
# ----------------------------


@metadata_agent.tool
@log_tool
async def merge_analysis_columns(ctx: RunContext[MetadataContext], columns_input: Union[str, List[str], dict] = None) -> dict:
    """
    Merge specified columns into a single analysis column.

    This tool can accept input in different formats:
    1. JSON string with {"columns": [...]} structure
    2. List of column names directly
    3. A single column name as string

    If only one column is provided, it's set as the analysis column.
    If multiple columns are provided, they are merged by joining values with an underscore.

    Uses the analysis_metadata_df (without ID columns) for column selection,
    but applies the merging to both metadata versions.

    The result is stored in ctx.deps.merged_column.
    """
    try:
        logger.info("🔀 Merging analysis columns from input: %s", columns_input)

        # Handle different input formats flexibly
        candidate_cols = []

        # Case 1: JSON string input
        if isinstance(columns_input, str):
            try:
                # Try to parse as JSON
                parsed = json.loads(columns_input)
                if isinstance(parsed, dict) and "columns" in parsed:
                    candidate_cols = parsed["columns"]
                elif isinstance(parsed, list):
                    candidate_cols = parsed
                else:
                    # Treat as a single column name
                    candidate_cols = [columns_input]
                logger.info("📋 Parsed JSON input to: %s", candidate_cols)
            except json.JSONDecodeError:
                # Not JSON, treat as a single column name
                candidate_cols = [columns_input]
                logger.info("📋 Using input as single column name: %s", candidate_cols)

        # Case 2: List input
        elif isinstance(columns_input, list):
            candidate_cols = columns_input
            logger.info("📋 Using list input directly: %s", candidate_cols)

        # Case 3: Dict input with "columns" key
        elif isinstance(columns_input, dict) and "columns" in columns_input:
            candidate_cols = columns_input["columns"]
            logger.info("📋 Extracted columns from dict input: %s", candidate_cols)

        # Validate we have something to work with
        if not candidate_cols:
            msg = "No columns provided for merging."
            logger.warning("⚠️ %s", msg)
            return {"message": msg, "success": False, "merged_column": None}

        # Check if we have the analysis metadata
        if ctx.deps.analysis_metadata_df is None:
            error_msg = "Error: Analysis metadata not available. Please process metadata first."
            logger.error("❌ %s", error_msg)
            return {"message": error_msg, "success": False, "merged_column": None}

        # Use the analysis_metadata_df for column selection
        analysis_df = ctx.deps.analysis_metadata_df.copy()

        # Filter to ensure columns actually exist in the analysis dataframe
        valid_cols = [col for col in candidate_cols if col in analysis_df.columns]
        if not valid_cols:
            msg = f"None of the specified columns {candidate_cols} exist in the analysis metadata."
            logger.warning("⚠️ %s", msg)
            return {"message": msg, "success": False, "merged_column": None}

        # Create the merged column in both dataframes
        if len(valid_cols) == 1:
            # Single column; use it directly
            ctx.deps.merged_column = valid_cols[0]
            msg = f"Single column '{valid_cols[0]}' selected as the analysis column."
            logger.info("✅ %s", msg)
        else:
            # Multiple columns; merge them
            merged_col = "merged_analysis_group"

            # Apply merging to the analysis metadata
            analysis_df[merged_col] = analysis_df[valid_cols].astype(str).apply(
                lambda row: "_".join(row.values), axis=1)
            ctx.deps.analysis_metadata_df = analysis_df  # update the analysis DataFrame

            # Also apply the same merging to the full metadata
            if ctx.deps.metadata_df is not None:
                full_df = ctx.deps.metadata_df.copy()
                # Ensure all columns exist in the full metadata before merging
                columns_in_full = [col for col in valid_cols if col in full_df.columns]
                if len(columns_in_full) == len(valid_cols):
                    full_df[merged_col] = full_df[valid_cols].astype(str).apply(
                        lambda row: "_".join(row.values), axis=1)
                    ctx.deps.metadata_df = full_df  # update the full DataFrame
                    logger.info("✅ Added merged column to both metadata versions")
                else:
                    logger.warning("⚠️ Not all selected columns exist in full metadata: %s",
                                  set(valid_cols) - set(columns_in_full))

            ctx.deps.merged_column = merged_col
            msg = f"Multiple columns {', '.join(valid_cols)} merged into column '{merged_col}'."
            logger.info("✅ %s", msg)

            # Show a preview of the merged values
            unique_merged = analysis_df[merged_col].unique().tolist()
            preview = unique_merged[:5] if len(unique_merged) > 5 else unique_merged
            logger.info("📊 Merged values preview: %s", preview)
            msg += f"\nMerged values (preview): {preview}"

        return {
            "message": msg,
            "success": True,
            "merged_column": ctx.deps.merged_column,
            "input_columns": valid_cols
        }
    except Exception as e:
        error_msg = f"Error in merge_analysis_columns: {str(e)}"
        logger.error("❌ %s", error_msg, exc_info=True)
        return {"message": error_msg, "success": False, "merged_column": None}


# ----------------------------
# Tool 3: Extract Unique Values
# ----------------------------

@metadata_agent.tool
@log_tool
async def extract_unique_values(ctx: RunContext[MetadataContext]) -> dict:
    """
    Simply extracts and returns the unique values from the selected analysis column.

    This tool:
    1. Identifies unique values in the selected analysis column from analysis_metadata_df
    2. Stores these values in ctx.deps.unique_groups
    3. Returns a dictionary with the unique values

    No additional analysis or contrast generation is performed.
    """
    try:
        logger.info("🔍 Extracting unique values from analysis column")

        if ctx.deps.analysis_metadata_df is None:
            error_msg = "Error: Analysis metadata has not been processed."
            logger.error("❌ %s", error_msg)
            return {"success": False, "message": error_msg, "unique_values": []}

        if not ctx.deps.merged_column:
            error_msg = "Error: Analysis column not defined. Please run merge_analysis_columns first."
            logger.error("❌ %s", error_msg)
            return {"success": False, "message": error_msg, "unique_values": []}

        df = ctx.deps.analysis_metadata_df
        analysis_col = ctx.deps.merged_column
        logger.info("📊 Using analysis column: %s", analysis_col)

        # Extract unique values from the analysis column
        unique_values = sorted(df[analysis_col].dropna().unique().tolist())
        ctx.deps.unique_groups = unique_values
        logger.info("✅ Extracted %d unique values: %s", len(unique_values), unique_values)

        # Return only the unique values with basic metadata
        return {
            "success": True,
            "column": analysis_col,
            "unique_values": unique_values,
            "count": len(unique_values)
        }

    except Exception as e:
        error_msg = f"Error extracting unique values: {str(e)}"
        logger.error("❌ %s", error_msg, exc_info=True)
        return {"success": False, "message": error_msg, "unique_values": []}


@log_agent_tool
async def run_agent_async(prompt: str, deps: MetadataContext, usage=None, output_type=None):
    logger.info("📋 Metadata agent invoked with prompt: %s", prompt)
    result = await metadata_agent.run(prompt, deps=deps, usage=usage, output_type=output_type)

    # Log the agent's output
    logger.info("📄 Metadata agent output: %s", result.output)

    # Log any contrasts generated (if available)
    if hasattr(result, 'output') and hasattr(result.output, 'contrasts'):
        logger.info("📊 Metadata agent generated %d contrasts", len(result.output.contrasts))

    # If you want to log usage statistics
    if hasattr(result, 'usage') and result.usage:
        try:
            usage_stats = result.usage()
            logger.info("📊 Metadata agent usage: %s", usage_stats)
        except Exception as e:
            logger.debug("Could not get usage stats: %s", e)

    return result
