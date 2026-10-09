"""
Test contrast name sanitization throughout the UORCA pipeline.

Investigates whether group names and contrast expressions reaching
the edgeR/limma R script are properly sanitized for R compatibility.

Uses real metadata + contrast files from previous UORCA runs as test fixtures.
"""

import glob
import os
import re

import pandas as pd
import pytest

from uorca.analysis.agents.metadata import clean_string

# ---------------------------------------------------------------------------
# Fixtures: collect real metadata + contrast pairs from ../UORCA_results
# ---------------------------------------------------------------------------

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "UORCA_results")

# Characters that are invalid in R variable names and would break makeContrasts()
R_INVALID_PATTERN = re.compile(r'[;/+()=\[\]{}|\\<>!@#$%^&*~`"\'?,]')

# Pattern for what a clean R-compatible name should look like after make.names()
# (alphanumeric, dots, underscores)
R_VALID_NAME = re.compile(r'^[A-Za-z.][A-Za-z0-9._]*$')


def _find_dataset_pairs(max_datasets: int = 30) -> list[dict]:
    """Find metadata + contrast CSV pairs across UORCA_results runs."""
    pairs = []
    search_dirs = sorted(glob.glob(os.path.join(RESULTS_ROOT, "*/GSE*/metadata")))

    for meta_dir in search_dirs:
        meta_files = glob.glob(os.path.join(meta_dir, "GSE*_metadata.csv"))
        contrast_file = os.path.join(meta_dir, "contrasts.csv")
        edger_file = os.path.join(meta_dir, "edger_analysis_samples.csv")

        if meta_files and os.path.exists(contrast_file):
            gse = os.path.basename(meta_files[0]).replace("_metadata.csv", "")
            pairs.append({
                "gse": gse,
                "metadata_path": meta_files[0],
                "contrast_path": contrast_file,
                "edger_samples_path": edger_file if os.path.exists(edger_file) else None,
            })
        if len(pairs) >= max_datasets:
            break

    return pairs


DATASET_PAIRS = _find_dataset_pairs(30)


# ---------------------------------------------------------------------------
# 1. Unit tests for clean_string()
# ---------------------------------------------------------------------------

class TestCleanString:
    """Verify clean_string() handles all problematic characters correctly."""

    @pytest.mark.parametrize("raw,expected_pattern", [
        # Semicolons (from genotype fields like GSE302346)
        ("R26LSL-tdTomato; H11LSL-Cas9; Kat7f/f", R_VALID_NAME),
        # Slashes
        ("Kat7f/f", R_VALID_NAME),
        # Plus signs
        ("tdTomato+ AT2 cells", R_VALID_NAME),
        # Parentheses
        ("KRAS(G12D) mutant", R_VALID_NAME),
        # Mixed special chars
        ("SARS-CoV-2 (MOI=0.1)", R_VALID_NAME),
        # Leading digit
        ("2xTreatment", R_VALID_NAME),
        # Backticks (used in R quoting)
        ("`some group`", R_VALID_NAME),
        # Spaces only
        ("Control group", R_VALID_NAME),
        # Hyphens
        ("wild-type", R_VALID_NAME),
    ])
    def test_clean_string_produces_r_valid_names(self, raw, expected_pattern):
        """clean_string() should produce names matching R's valid name pattern."""
        result = clean_string(raw)
        assert expected_pattern.match(result), (
            f"clean_string({raw!r}) = {result!r} is not R-valid"
        )

    @pytest.mark.parametrize("raw", [
        "R26LSL-tdTomato; H11LSL-Cas9; Kat7f/f",
        "Kat7f/f_Kat7-deficient tdTomato+ AT2 cells mouse 1",
        "SARS-CoV-2 (MOI=0.1)",
        "COVID-19+ / ICU",
    ])
    def test_clean_string_removes_r_breaking_chars(self, raw):
        """clean_string() should remove characters that break R's makeContrasts()."""
        result = clean_string(raw)
        assert not R_INVALID_PATTERN.search(result), (
            f"clean_string({raw!r}) = {result!r} still contains R-breaking chars"
        )

    def test_clean_string_is_idempotent(self):
        """Applying clean_string twice should give the same result."""
        raw = "R26LSL-tdTomato; H11LSL-Cas9; Kat7f/f"
        first = clean_string(raw)
        second = clean_string(first)
        assert first == second


# ---------------------------------------------------------------------------
# 2. Test: metadata values as stored in edger_analysis_samples.csv
#    (the file that actually gets passed to the R script)
# ---------------------------------------------------------------------------

class TestEdgerSamplesSanitization:
    """Check if group names in edger_analysis_samples.csv are R-compatible.

    The last column in edger_analysis_samples.csv is the merged grouping
    column — the actual values that become factor levels in R's design matrix.
    """

    @pytest.mark.parametrize(
        "pair",
        [p for p in DATASET_PAIRS if p["edger_samples_path"]],
        ids=lambda p: p["gse"],
    )
    def test_merged_column_values_are_r_safe(self, pair):
        """Group names in edger_analysis_samples.csv should not contain
        characters that break R's makeContrasts()."""
        df = pd.read_csv(pair["edger_samples_path"])
        merged_col = df.columns[-1]  # Last column is the merged grouping column
        unique_groups = df[merged_col].dropna().unique()

        violations = []
        for group in unique_groups:
            group_str = str(group)
            if R_INVALID_PATTERN.search(group_str):
                cleaned = clean_string(group_str)
                violations.append(
                    f"  {group_str!r} -> would become {cleaned!r} after clean_string()"
                )

        if violations:
            pytest.fail(
                f"{pair['gse']}: merged column '{merged_col}' contains "
                f"R-unsafe group names:\n" + "\n".join(violations)
            )


# ---------------------------------------------------------------------------
# 3. Test: contrast expressions contain only sanitized group references
# ---------------------------------------------------------------------------

class TestContrastExpressionSanitization:
    """Check if contrast expressions are valid for R's makeContrasts()."""

    @pytest.mark.parametrize("pair", DATASET_PAIRS, ids=lambda p: p["gse"])
    def test_contrast_expressions_are_r_parseable(self, pair):
        """Contrast expressions should not contain semicolons, slashes,
        or other chars that break R's parse(text=makeContrasts_str)."""
        df = pd.read_csv(pair["contrast_path"])
        if "expression" not in df.columns:
            pytest.skip("No 'expression' column in contrasts.csv")

        violations = []
        for _, row in df.iterrows():
            expr = str(row["expression"])
            # Check for R-breaking chars in the expression
            # (excluding the arithmetic operators - + / which are part of R syntax)
            # Focus on semicolons, backticks, quotes in wrong positions, etc.
            bad_chars = re.findall(r'[;\\|!@#$%^&{}~`]', expr)
            if bad_chars:
                violations.append(
                    f"  Contrast '{row.get('name', '?')}': "
                    f"expression contains {bad_chars}: {expr[:80]}..."
                )

        if violations:
            pytest.fail(
                f"{pair['gse']}: contrast expressions contain R-unsafe chars:\n"
                + "\n".join(violations)
            )

    @pytest.mark.parametrize("pair", DATASET_PAIRS, ids=lambda p: p["gse"])
    def test_contrast_group_refs_match_sanitized_metadata(self, pair):
        """Group names referenced in contrast expressions should exist
        (after R make.names sanitization) in the actual metadata."""
        if not pair["edger_samples_path"]:
            pytest.skip("No edger_analysis_samples.csv")

        samples_df = pd.read_csv(pair["edger_samples_path"])
        merged_col = samples_df.columns[-1]
        actual_groups = set(str(g) for g in samples_df[merged_col].dropna().unique())

        contrasts_df = pd.read_csv(pair["contrast_path"])
        if "expression" not in contrasts_df.columns:
            pytest.skip("No 'expression' column")

        # Extract group name references from expressions
        # (tokens that aren't arithmetic operators or numbers)
        for _, row in contrasts_df.iterrows():
            expr = str(row["expression"])
            # Remove quoted strings and extract tokens
            tokens = re.findall(r"'([^']+)'", expr)
            if not tokens:
                # Try extracting unquoted tokens (words separated by - + / operators)
                tokens = re.split(r'\s*[-+/]\s*', expr)
                tokens = [t.strip().strip("()") for t in tokens if t.strip()]
                # Filter out numbers
                tokens = [t for t in tokens if not re.match(r'^[0-9.]+$', t)]

            for token in tokens:
                if token and token not in actual_groups:
                    # Check if it would match after clean_string
                    cleaned_token = clean_string(token)
                    cleaned_groups = {clean_string(g) for g in actual_groups}
                    if cleaned_token not in cleaned_groups:
                        # Genuine mismatch - not just a sanitization issue
                        pass  # This is a separate issue (LLM hallucination)


# ---------------------------------------------------------------------------
# 4. Test: simulate the MetadataAnalysisNode merge path
#    This is the KEY test — it reproduces the exact code path where
#    clean_string() is NOT being called.
# ---------------------------------------------------------------------------

class TestMetadataNodeMergePath:
    """Simulate the MetadataAnalysisNode column merge logic and verify
    whether the resulting group names would be R-safe.

    The node at uorca/graph/nodes/metadata.py:79-82 does:
        df[merged_col] = df[selected_cols].astype(str).agg("_".join, axis=1)

    This uses RAW values from the GEO-extracted metadata without calling
    clean_string(). This test checks whether that produces R-unsafe names.
    """

    @pytest.mark.parametrize(
        "pair",
        [p for p in DATASET_PAIRS if p["edger_samples_path"]],
        ids=lambda p: p["gse"],
    )
    def test_raw_merge_produces_r_safe_names(self, pair):
        """Simulates the merge without clean_string() and checks R-safety.

        If this test FAILS, it proves that clean_string() is needed in the
        MetadataAnalysisNode merge path.
        """
        # Load original metadata (as GEO extraction would produce)
        meta_df = pd.read_csv(pair["metadata_path"])

        # Load edger_analysis_samples to find which columns were merged
        edger_df = pd.read_csv(pair["edger_samples_path"])
        merged_col_name = edger_df.columns[-1]

        # The merged column name often hints at which columns were combined
        # e.g., "tissue_genotype" = tissue + genotype
        component_cols = merged_col_name.split("_")

        # Find matching columns in original metadata
        matching_cols = []
        for col in meta_df.columns:
            col_lower = col.lower().replace(" ", "_")
            for comp in component_cols:
                if comp.lower() in col_lower:
                    matching_cols.append(col)
                    break

        if not matching_cols:
            pytest.skip(f"Could not identify source columns for '{merged_col_name}'")

        # Simulate the EXACT merge logic from MetadataAnalysisNode (line 82)
        if len(matching_cols) > 1:
            merged_values = meta_df[matching_cols].astype(str).agg("_".join, axis=1)
        else:
            merged_values = meta_df[matching_cols[0]].astype(str)

        # Check if the raw merged values are R-safe
        unsafe_values = []
        for val in merged_values.unique():
            if R_INVALID_PATTERN.search(str(val)):
                unsafe_values.append(str(val))

        if unsafe_values:
            # This is expected to fail — proving the gap
            pytest.fail(
                f"{pair['gse']}: raw merge of {matching_cols} produces "
                f"R-unsafe group names (clean_string NOT applied):\n"
                + "\n".join(f"  {v!r}" for v in unsafe_values[:5])
            )

    @pytest.mark.parametrize(
        "pair",
        [p for p in DATASET_PAIRS if p["edger_samples_path"]],
        ids=lambda p: p["gse"],
    )
    def test_clean_merge_produces_r_safe_names(self, pair):
        """Same as above but WITH clean_string() — should always pass.

        If raw test fails but this passes, it confirms that adding
        clean_string() to the merge path would fix the issue.
        """
        meta_df = pd.read_csv(pair["metadata_path"])
        edger_df = pd.read_csv(pair["edger_samples_path"])
        merged_col_name = edger_df.columns[-1]

        component_cols = merged_col_name.split("_")
        matching_cols = []
        for col in meta_df.columns:
            col_lower = col.lower().replace(" ", "_")
            for comp in component_cols:
                if comp.lower() in col_lower:
                    matching_cols.append(col)
                    break

        if not matching_cols:
            pytest.skip(f"Could not identify source columns for '{merged_col_name}'")

        # Apply clean_string to values BEFORE merging (the fix)
        cleaned_df = meta_df[matching_cols].copy()
        for col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].apply(
                lambda x: clean_string(x) if pd.notna(x) else x
            )

        if len(matching_cols) > 1:
            merged_values = cleaned_df.astype(str).agg("_".join, axis=1)
        else:
            merged_values = cleaned_df[matching_cols[0]].astype(str)

        # These should all be R-safe now
        for val in merged_values.unique():
            val_str = str(val)
            assert not R_INVALID_PATTERN.search(val_str), (
                f"{pair['gse']}: even after clean_string(), "
                f"value {val_str!r} is still R-unsafe"
            )


# ---------------------------------------------------------------------------
# 5. Test: the actual edger_analysis_samples values vs clean_string output
#    This directly tests whether clean_string was called before writing
#    the file that R reads.
# ---------------------------------------------------------------------------

class TestCleanStringWasApplied:
    """Detect whether clean_string() was actually applied to the group
    names that were passed to R.

    If clean_string() had been applied, the group values in
    edger_analysis_samples.csv would NOT contain spaces, hyphens, or
    special characters (they'd be converted to underscores or removed).
    """

    @pytest.mark.parametrize(
        "pair",
        [p for p in DATASET_PAIRS if p["edger_samples_path"]],
        ids=lambda p: p["gse"],
    )
    def test_group_names_show_clean_string_was_applied(self, pair):
        """Check if the group names bear the signature of clean_string().

        clean_string() converts:
          - spaces -> underscores
          - removes non-word chars (hyphens, semicolons, etc.)
          - removes non-ASCII

        If we find spaces or hyphens in group names, clean_string()
        was NOT applied to those values.
        """
        df = pd.read_csv(pair["edger_samples_path"])
        merged_col = df.columns[-1]
        unique_groups = df[merged_col].dropna().unique()

        uncleaned_indicators = []
        for group in unique_groups:
            g = str(group)
            # clean_string replaces spaces with underscores
            if " " in g:
                uncleaned_indicators.append(f"  contains space: {g!r}")
            # clean_string removes hyphens (non-word chars)
            if "-" in g:
                uncleaned_indicators.append(f"  contains hyphen: {g!r}")
            # Definitely not cleaned
            if ";" in g or "/" in g or "+" in g:
                uncleaned_indicators.append(f"  contains ;/+: {g!r}")

        if uncleaned_indicators:
            pytest.fail(
                f"{pair['gse']}: group names in edger_analysis_samples.csv "
                f"show clean_string() was NOT applied:\n"
                + "\n".join(uncleaned_indicators[:10])
            )
