"""
UORCA Summary Tab: shows a lightweight overview.

Displays:
- Original research question (if available)
- Total datasets
- Total contrasts
- Datasets per species
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import Optional, Dict, Any
from datetime import datetime

import streamlit as st

from uorca.gui.results_integration import ResultsIntegrator
from .helpers import log_streamlit_function, get_valid_contrasts_with_data


def _load_research_query(results_dir: Optional[str]) -> Optional[str]:
    """Load the research question using the same logic as the AI tab.

    Reads ``<results_dir>/research_question.json`` (key: "research_question"). There is
    no global fallback: a package-wide "last query" file showed the question of an
    unrelated identification run for these results.
    """
    # 1) results_dir/research_question.json
    if results_dir:
        p = Path(results_dir) / "research_question.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                rq = data.get("research_question")
                if isinstance(rq, str) and rq.strip():
                    return rq.strip()
            except Exception as exc:
                st.warning(f"Could not read {p}: {type(exc).__name__}: {exc}")

    return None


@log_streamlit_function
def render_uorca_summary_tab(ri: ResultsIntegrator, results_dir: str):
    """Render a simple overview of the analysis contents."""
    # 1) Research question
    research_q = _load_research_query(results_dir)
    st.subheader("Research Question")
    if research_q:
        st.write(research_q)
    else:
        st.caption("No saved research question found.")

    # 2) Total datasets
    total_datasets = len(getattr(ri, "cpm_data", {}) or {})

    # 3) Total contrasts (prefer analysis_info if available, else DEGs with data)
    total_contrasts = 0
    if hasattr(ri, "analysis_info") and ri.analysis_info:
        for dsid, info in ri.analysis_info.items():
            contr = info.get("contrasts")
            if isinstance(contr, list):
                total_contrasts += len(contr)
    if total_contrasts == 0:
        # Fallback to contrasts with DEG data available
        total_contrasts = len(get_valid_contrasts_with_data(ri))

    # 4) Total samples across datasets (estimate from CPM tables)
    total_samples = 0
    if hasattr(ri, "cpm_data") and isinstance(ri.cpm_data, dict):
        for dsid, cpm_df in ri.cpm_data.items():
            try:
                cols = list(getattr(cpm_df, 'columns', []))
                if not cols:
                    continue
                sample_count = len(cols) - (1 if 'Gene' in cols else 0)
                if sample_count > 0:
                    total_samples += sample_count
            except Exception:
                continue

    # 5) Datasets per species
    species_counter = Counter()
    if hasattr(ri, "analysis_info"):
        for dsid, info in ri.analysis_info.items():
            org = info.get("organism", "Unknown")
            if not org:
                org = "Unknown"
            species_counter[org] += 1

    st.subheader("Summary")
    # Bullet points for the summary items
    st.markdown(f"- Total datasets: {total_datasets}")
    st.markdown(f"- Total contrasts: {total_contrasts}")
    st.markdown(f"- Total samples: {total_samples}")

    # Earliest and latest analysis timestamps across datasets
    earliest_ts: Optional[datetime] = None
    latest_ts: Optional[datetime] = None
    if hasattr(ri, "analysis_info") and isinstance(ri.analysis_info, dict):
        for info in ri.analysis_info.values():
            try:
                cps = info.get("checkpoints", {}) if isinstance(info, dict) else {}
                # Prefer rnaseq_analysis, else fall back to edger_limma_preparation, kallisto_quantification, metadata_extraction
                for key in [
                    "rnaseq_analysis",
                    "edger_limma_preparation",
                    "kallisto_quantification",
                    "metadata_extraction",
                ]:
                    if key in cps and isinstance(cps[key], dict):
                        ts = cps[key].get("timestamp")
                        if isinstance(ts, str) and ts:
                            try:
                                dt = datetime.fromisoformat(ts)
                            except Exception:
                                continue
                            if earliest_ts is None or dt < earliest_ts:
                                earliest_ts = dt
                            if latest_ts is None or dt > latest_ts:
                                latest_ts = dt
                            break
            except Exception:
                continue
    # Friendly time formatting helper
    def _fmt(dt: datetime) -> str:
        try:
            # Show in local time with minutes precision
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return dt.isoformat()



    st.subheader("Datasets by Species")
    if species_counter:
        for org, count in species_counter.items():
            st.write(f"- {org}: {count}")
    else:
        st.caption("No species information available.")

    # Bottom analysis info in normal size
    if earliest_ts is not None or latest_ts is not None:
        start_txt = _fmt(earliest_ts) if earliest_ts else "N/A"
        end_txt = _fmt(latest_ts) if latest_ts else "N/A"
        st.subheader("Other information")
        st.markdown(f"- Start: {start_txt}")
        st.markdown(f"- End: {end_txt}")
        st.markdown("- UORCA version: 1.0")
