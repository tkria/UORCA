"""
AI Assistant Tab for UORCA Explorer.

This tab provides AI-powered analysis and exploration capabilities.
"""
from __future__ import annotations

import os
import json
import asyncio
import concurrent.futures
import logging
import time
import streamlit as st
import pandas as pd
import pydantic
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from uorca.gui.ai import (
    get_contrast_relevance_with_selection_config,
    get_ai_agent_config,
    GeneAnalysisOutput,
    run_tool_relevance_analysis_sync,
    get_relevant_tool_calls,
    clear_relevant_tool_log,
    create_uorca_agent,
    run_contrast_relevance,
    run_contrast_relevance_with_selection,
    SelectedContrast
)
from uorca.gui.mcp_server import get_filtered_dataframe

from .helpers import (
    check_ai_generating,
    setup_fragment_decorator,
    log_streamlit_tab,
    log_streamlit_function,
    log_streamlit_event,
    log_streamlit_agent,
    get_valid_contrasts_with_data,
    is_analysis_successful,
    get_valid_contrasts_from_analysis_info,
    validate_contrast_has_deg_data
)
from .helpers.ai_agent_tool_logger import (
    start_ai_analysis_session,
    get_ai_tool_logs_for_display,
    get_current_log_file,
    read_log_file_contents
)
from uorca.gui.results_integration import ResultsIntegrator

# Set up fragment decorator
setup_fragment_decorator()

logger = logging.getLogger(__name__)


def _get_ai_analysis_scope(ri: ResultsIntegrator, selected_datasets: List[str]) -> Dict[str, int]:
    """Get the scope of AI analysis (dataset and contrast counts)."""
    contrast_data = _create_ai_contrast_data_filtered(ri, selected_datasets)
    return {
        'datasets': len(selected_datasets),
        'contrasts': len(contrast_data)
    }


def _create_ai_contrast_data_filtered(ri: ResultsIntegrator, selected_datasets: List[str]) -> List[Dict[str, Any]]:
    """Create contrast data filtered by selected datasets using standardized validation logic."""
    # Use the centralized validation function
    valid_contrasts = get_valid_contrasts_with_data(ri, selected_datasets)

    # Convert to format expected by AI analysis using consistent names
    contrast_data = []
    for contrast in valid_contrasts:
        contrast_data.append({
            "accession": contrast["accession"],
            "contrast": contrast["contrast_name"],  # Already has consistent name from validation
            "description": contrast["description"]
            })

    return contrast_data



def _run_filtered_contrast_relevance_with_selection(ri, filtered_contrast_data, query: str, repeats: int, batch_size: int, parallel_jobs: int):
    """Run contrast relevance with selection on filtered contrast data."""
    if CONTRAST_RELEVANCE_WITH_SELECTION_AVAILABLE:
        # Filter the RI analysis_info to only include selected datasets
        original_analysis_info = ri.analysis_info.copy()

        # Create filtered analysis_info containing only selected datasets
        filtered_analysis_ids = set([contrast['accession'] for contrast in filtered_contrast_data])
        ri.analysis_info = {aid: info for aid, info in original_analysis_info.items()
                          if info.get('accession', aid) in filtered_analysis_ids}

        try:
            results_df, selected_contrasts = run_contrast_relevance_with_selection(
                ri, query, repeats, batch_size, parallel_jobs
            )
            return results_df, selected_contrasts
        finally:
            # Restore original analysis_info
            ri.analysis_info = original_analysis_info
    else:
        return pd.DataFrame(), []

def _run_filtered_contrast_relevance(ri, filtered_contrast_data, query: str, repeats: int, batch_size: int, parallel_jobs: int):
    """Run contrast relevance on filtered contrast data."""
    if CONTRAST_RELEVANCE_AVAILABLE:
        # Filter the RI analysis_info to only include selected datasets
        original_analysis_info = ri.analysis_info.copy()

        # Create filtered analysis_info containing only selected datasets
        filtered_analysis_ids = set([contrast['accession'] for contrast in filtered_contrast_data])
        ri.analysis_info = {aid: info for aid, info in original_analysis_info.items()
                          if info.get('accession', aid) in filtered_analysis_ids}

        try:
            results_df = run_contrast_relevance(
                ri, query, repeats, batch_size, parallel_jobs
            )
            return results_df
        finally:
            # Restore original analysis_info
            ri.analysis_info = original_analysis_info
    else:
        return pd.DataFrame()


def load_query_config(results_dir: Optional[str] = None) -> Optional[str]:
    """Load the dataset identification query from results directory or config file."""

    # First, try to load from results directory if provided
    if results_dir:
        research_question_file = os.path.join(results_dir, "research_question.json")
        if os.path.exists(research_question_file):
            try:
                with open(research_question_file, 'r') as f:
                    question_data = json.load(f)
                    research_question = question_data.get("research_question")
                    if research_question:
                        return research_question
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                pass

    # Fallback to original config file approach
    # Config is now in uorca/config/
    from pathlib import Path
    config_file_path = Path(__file__).parent.parent.parent / "config" / "dataset_query.json"

    try:
        if os.path.exists(config_file_path):
            with open(config_file_path, 'r') as f:
                config_data = json.load(f)
                return config_data.get("query")
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        pass
    return None


# Check for AI functionality availability

# AI components are now always available since they're imported at the top
UORCA_AGENT_AVAILABLE = True
CONTRAST_RELEVANCE_AVAILABLE = True
CONTRAST_RELEVANCE_WITH_SELECTION_AVAILABLE = True


@log_streamlit_tab("AI Assistant")
def render_ai_assistant_tab(ri: ResultsIntegrator, results_dir: str, selected_datasets: List[str]):
    """
    Render the AI assistant tab with subtabs for analysis and tool logs.

    Args:
        ri: ResultsIntegrator instance
        results_dir: Path to the results directory
        selected_datasets: List of selected dataset IDs from sidebar
    """
    st.header("AI Assistant")
    st.markdown("Enter your research question and let the AI agent perform an automated analysis of the data.")

    # Display current configuration
    with st.expander("Current Configuration", expanded=False):
        try:
            ai_config = get_ai_agent_config()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Request Limit:** {ai_config.request_limit}")
                st.markdown(f"**Analysis Timeout:** {ai_config.analysis_timeout}s")
            with col2:
                st.markdown(f"**Model:** {ai_config.model}")
                st.markdown(f"**Temperature:** {ai_config.temperature}")
        except Exception as e:
            st.warning(f"Could not load configuration: {e}")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        log_streamlit_event("OpenAI API key not found for AI assistant")
        st.warning("⚠️ OpenAI API key not configured")

        with st.expander("🔧 Setup Instructions", expanded=True):
            st.markdown("""
            **To enable AI-powered analysis, you need to set up an OpenAI API key:**

            ### Option 1: Environment Variable (Recommended)
            ```bash
            # Add to your .env file or shell environment
            export OPENAI_API_KEY="<your-openai-api-key>"
            ```

            ### Option 2: Create .env file
            Create a `.env` file in your project directory:
            ```
            OPENAI_API_KEY=<your-openai-api-key>
            ```

            ### Getting an API Key
            1. Visit [OpenAI's API platform](https://platform.openai.com/api-keys)
            2. Sign up or log in to your account
            3. Create a new API key
            4. Copy the key and set it using one of the methods above
            5. Restart the Streamlit app

            **Note**: You can still use all other features of UORCA Explorer without the API key.
            """)

        st.info("💡 **Tip**: The other tabs (Heatmap, Expression Plots, etc.) work without an API key!")
        return

    # Render streamlined AI analysis workflow (no tabs needed)
    _render_streamlined_ai_workflow(ri, results_dir, selected_datasets)


def _render_streamlined_ai_workflow(ri: ResultsIntegrator, results_dir: str, selected_datasets: List[str]):
    """Render the streamlined AI analysis workflow."""

    # Check if datasets are selected
    if not selected_datasets:
        st.markdown("To get started, select datasets from the sidebar on the left.")
        return

    # Show scope information
    scope = _get_ai_analysis_scope(ri, selected_datasets)
    st.markdown(f"The agent will consider **{scope['contrasts']} contrasts** across **{scope['datasets']} datasets**")

    # Initialise caching system
    _initialise_ai_cache()

    # Clear cache if datasets have changed
    if 'previous_selected_datasets' not in st.session_state:
        st.session_state['previous_selected_datasets'] = set()

    current_datasets = set(selected_datasets)
    if current_datasets != st.session_state['previous_selected_datasets']:
        # Datasets changed - clear AI analysis cache
        st.session_state['ai_analysis_cache'] = {}
        st.session_state['current_analysis_id'] = None
        st.session_state['show_cached_results'] = False
        st.session_state['previous_selected_datasets'] = current_datasets

    # Load saved query from dataset identification (check results directory first)
    saved_query = load_query_config(results_dir)
    default_placeholder = "e.g., What contrasts are most relevant to T cell activation and differentiation?"

    # Use saved query as placeholder if available
    if saved_query:
        placeholder_text = f"Dataset query: {saved_query}"
        help_text = "Using the research question from your dataset identification. You can modify this or enter a new question focusing on specific aspects of your data."
    else:
        placeholder_text = default_placeholder
        help_text = "Describe your research question or area of interest. The AI will score each contrast based on how relevant it is to this query, then analyse key genes."

    # Research query input
    research_query = st.text_input(
        "Research Question",
        value=saved_query if saved_query else "",
        placeholder=placeholder_text,
        help=help_text,
        key="ai_research_query_input"
    )

    # Single workflow button
    col1, col2 = st.columns([1, 3])
    with col1:
        run_button = st.button("Run Complete AI Analysis", type="primary", key="ai_run_analysis_button")
    with col2:
        if not research_query.strip():
            st.info("Enter a research question above to start AI analysis.")

    if run_button:
        log_streamlit_event(f"User started complete AI analysis: '{research_query.strip()}'")
        _run_complete_ai_analysis(ri, results_dir, research_query.strip(), selected_datasets)

    # Show cached results if available (below the form) - but not if we just ran a fresh analysis
    if _should_restore_cached_analysis(selected_datasets):
        st.markdown("---")
        _restore_and_display_cached_analysis(ri, results_dir, selected_datasets)


@log_streamlit_agent
def _build_analysis_prompt(
    research_query: str,
    selected_contrast_dicts: List[Dict],
    enhanced: bool,
) -> str:
    """Build the per-run analysis prompt for the interpretation agent.

    Args:
        research_query: The user's research question.
        selected_contrast_dicts: Contrasts to analyse.
        enhanced: True when contrasts were narrowed by relevance selection.

    The toolbox description must stay in step with the tools the MCP server
    exposes (see uorca/gui/mcp_server/server_core.py) and with the system prompt
    in uorca/gui/ai/prompts/ai_agent_analysis.txt. It deliberately describes what
    each tool is *for* rather than prescribing an order: the agent is expected to
    explore, repeat calls with different parameters, and follow its own leads.
    """
    budget = get_ai_agent_config().request_limit

    tool_plan = f"""Your toolbox — use any of these, in any order, as many times as the evidence warrants:
- get_most_common_genes() — genes recurring across contrasts; vary the thresholds and top_n to see how the signal behaves
- filter_genes_by_contrast_sets() — genes specific to one group of contrasts versus another; try several groupings
- calculate_expression_variability() — how consistently genes respond across contrasts
- calculate_gene_correlation() — co-expression structure among candidate genes
- get_gene_contrast_stats() — per-contrast detail for specific genes
- summarise_contrast() — the overall picture for an individual contrast

You have a budget of {budget} tool calls. Use it generously — this is a budget to spend,
not to conserve. Work iteratively: form a hypothesis, test it with whichever tool answers
it, and let each result decide your next call. Repeat tools with different parameters,
thresholds, contrast groupings or gene sets whenever that would sharpen or challenge a
finding. Follow up on anything surprising. Do not settle for a single pass: keep going
until further calls stop changing your conclusions, then report what you found."""

    if enhanced:
        return f"""
Research question: "{research_query}"

I have intelligently selected the following {len(selected_contrast_dicts)} contrasts from a larger set based on relevance, diversity, and analytical value:

{json.dumps(selected_contrast_dicts, indent=2)}

These contrasts were chosen to provide:
1. High relevance to the research question
2. Diversity of biological contexts
3. Appropriate controls and comparisons
4. Maximum analytical power for comparative analysis

Investigate this question as thoroughly as the data allows.

{tool_plan}

Choose thresholds yourself, and when you are done reporting, return:
1. Key genes identified with their biological significance
2. Patterns that distinguish different contrast categories
3. A brief biological interpretation of the findings
"""

    return f"""
Research question: "{research_query}"

Available contrasts:
{json.dumps(selected_contrast_dicts, indent=2)}

Investigate this question as thoroughly as the data allows, choosing all thresholds yourself.

{tool_plan}

When you are done exploring, return:
1) A structured summary showing the key genes identified for each contrast or gene set
2) A brief 2-3 sentence biological interpretation explaining your rationale and what patterns you discovered.
"""


def _run_complete_ai_analysis(ri: ResultsIntegrator, results_dir: str, research_query: str, selected_datasets: List[str]):
    """Run the complete AI analysis workflow including contrast relevance and gene analysis."""
    # Add validation for empty research query
    if not research_query.strip():
        st.error("Please enter a research question before running the analysis.")
        return

    if not CONTRAST_RELEVANCE_AVAILABLE:
        st.error("Contrast relevance assessment is not available. Please check your environment setup.")
        return

    if not UORCA_AGENT_AVAILABLE:
        st.error("UORCA agent is not available. Please check your environment setup.")
        return

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OpenAI API key not found. Please configure your API key in the instructions above.")
        return

    # Run complete analysis workflow with dynamic progress
    progress_placeholder = st.empty()

    try:
        # Step 1: Contrast Relevance Assessment with Selection (no display)
        with progress_placeholder:
            with st.spinner("Selecting relevant contrasts..."):
                if CONTRAST_RELEVANCE_WITH_SELECTION_AVAILABLE:
                    # Filter contrasts to selected datasets only
                    filtered_contrast_data = _create_ai_contrast_data_filtered(ri, selected_datasets)
                    if not filtered_contrast_data:
                        st.warning("No contrasts found in selected datasets.")
                        return

                    # Run contrast relevance assessment with intelligent selection on filtered data
                    relevance_config = get_contrast_relevance_with_selection_config()
                    results_df, selected_contrasts = _run_filtered_contrast_relevance_with_selection(
                        ri,
                        filtered_contrast_data,
                        query=research_query,
                        repeats=relevance_config.repeats,
                        batch_size=relevance_config.batch_size,
                        parallel_jobs=relevance_config.parallel_jobs
                    )

                    if not results_df.empty and selected_contrasts:
                        # Store SELECTED contrasts for AI gene analysis (not all contrasts)
                        # Use consistent contrast names for AI
                        selected_contrast_dicts = []
                        for sc in selected_contrasts:
                            # Find the consistent name from ri.contrast_info
                            consistent_name = sc.contrast_id
                            for contrast_key, contrast_data in ri.contrast_info.items():
                                if (contrast_data.get('original_name') == sc.contrast_id and
                                    contrast_data.get('analysis_id') == sc.analysis_id):
                                    consistent_name = contrast_data.get('name', sc.contrast_id)
                                    break

                            selected_contrast_dicts.append({
                                'analysis_id': sc.analysis_id,
                                'contrast_id': consistent_name
                            })
                        st.session_state['selected_contrasts_for_ai'] = selected_contrast_dicts
                        st.session_state['research_query'] = research_query
                    else:
                        if not os.getenv("OPENAI_API_KEY"):
                            st.warning("⚠️ OpenAI API key not configured. Cannot assess contrast relevance.")
                            st.info("Please configure your OpenAI API key using the instructions above to enable AI analysis")
                        else:
                            st.warning("No contrasts found for assessment.")
                        return
                else:
                    # Filter contrasts to selected datasets only
                    filtered_contrast_data = _create_ai_contrast_data_filtered(ri, selected_datasets)
                    if not filtered_contrast_data:
                        st.warning("No contrasts found in selected datasets.")
                        return

                    # Fall back to original approach with filtered data
                    relevance_config = get_contrast_relevance_with_selection_config()
                    results_df = _run_filtered_contrast_relevance(
                        ri,
                        filtered_contrast_data,
                        query=research_query,
                        repeats=relevance_config.repeats,
                        batch_size=relevance_config.batch_size,
                        parallel_jobs=relevance_config.parallel_jobs
                    )

                    if not results_df.empty:
                        # Convert to consistent contrast names for AI
                        selected_contrast_dicts = []
                        for _, row in results_df.iterrows():
                            analysis_id = row['analysis_id']
                            contrast_id = row['contrast_id']

                            # Find the consistent name from ri.contrast_info
                            consistent_name = contrast_id
                            for contrast_key, contrast_data in ri.contrast_info.items():
                                if (contrast_data.get('original_name') == contrast_id and
                                    contrast_data.get('analysis_id') == analysis_id):
                                    consistent_name = contrast_data.get('name', contrast_id)
                                    break

                            selected_contrast_dicts.append({
                                'analysis_id': analysis_id,
                                'contrast_id': consistent_name
                            })

                        st.session_state['selected_contrasts_for_ai'] = selected_contrast_dicts
                        st.session_state['research_query'] = research_query
                        selected_contrasts = None  # No selection objects in fallback mode
                    else:
                        if not os.getenv("OPENAI_API_KEY"):
                            st.warning("⚠️ OpenAI API key not configured. Cannot assess contrast relevance.")
                            st.info("Please configure your OpenAI API key using the instructions above to enable AI analysis")
                        else:
                            st.warning("No contrasts found for assessment.")
                        return

        # Step 2: AI Gene Analysis (using selected contrasts)
        logger.info("WORKFLOW: Starting AI Gene Analysis phase")
        with progress_placeholder:
            with st.spinner(f"Analysing expression patterns..."):
                # Set the RESULTS_DIR environment variable for the MCP server
                os.environ['RESULTS_DIR'] = results_dir
                logger.info(f"WORKFLOW: Set RESULTS_DIR environment variable to: {results_dir}")

                # Retrieve selected contrasts from session state with debugging
                if 'selected_contrasts_for_ai' not in st.session_state:
                    logger.error("WORKFLOW: Missing 'selected_contrasts_for_ai' in session state")
                    st.error("Internal error: Selected contrasts not found in session state.")
                    return

                selected_contrast_dicts = st.session_state['selected_contrasts_for_ai']
                logger.info(f"WORKFLOW: Retrieved {len(selected_contrast_dicts)} selected contrasts from session state")
                logger.debug(f"WORKFLOW: Selected contrasts: {selected_contrast_dicts}")

                # Validate selected contrasts
                if not selected_contrast_dicts:
                    logger.warning("WORKFLOW: No contrasts selected for AI analysis")
                    st.warning("No contrasts were selected for analysis. Please try running the analysis again.")
                    return

                # Pass selected contrasts to MCP server for filtering
                import json
                os.environ['SELECTED_CONTRASTS_FOR_AI'] = json.dumps(selected_contrast_dicts)
                logger.info("WORKFLOW: Set SELECTED_CONTRASTS_FOR_AI environment variable")

                # Create cache key based on selected contrasts to invalidate agent cache when contrasts change
                import hashlib
                contrasts_key = hashlib.md5(json.dumps(selected_contrast_dicts, sort_keys=True).encode()).hexdigest()
                logger.info(f"WORKFLOW: Generated contrasts cache key: {contrasts_key[:16]}...")

                # Create AI agent with debugging
                logger.info("WORKFLOW: Creating UORCA AI agent...")
                try:
                    agent = create_uorca_agent(selected_contrasts_key=contrasts_key)
                    logger.info("WORKFLOW: AI agent creation attempt completed")
                except Exception as e:
                    logger.error(f"WORKFLOW: AI agent creation failed with exception: {str(e)}", exc_info=True)
                    st.error(f"Failed to create AI agent: {str(e)}")
                    return

                # Check if agent creation failed due to missing API key
                if agent is None:
                    logger.error("WORKFLOW: AI agent creation returned None")
                    st.error("Failed to create AI agent. Please check that your OpenAI API key is properly configured.")
                    return

                logger.info("WORKFLOW: AI agent created successfully")

                # Enhanced prompt that leverages the selection
                if CONTRAST_RELEVANCE_WITH_SELECTION_AVAILABLE and len(selected_contrast_dicts) <= 20:
                    prompt = _build_analysis_prompt(
                        research_query, selected_contrast_dicts, enhanced=True
                    )
                    # Log the AI analysis prompt
                    logger.info(f"AI ANALYSIS PROMPT (Enhanced): Research question: '{research_query}' | Selected contrasts: {len(selected_contrast_dicts)} | Prompt length: {len(prompt)} chars")
                    logger.debug(f"Full AI analysis prompt:\n{prompt}")
                else:
                    prompt = _build_analysis_prompt(
                        research_query, selected_contrast_dicts, enhanced=False
                    )
                    # Log the AI analysis prompt
                    logger.info(f"AI ANALYSIS PROMPT (Fallback): Research question: '{research_query}' | Available contrasts: {len(selected_contrast_dicts)} | Prompt length: {len(prompt)} chars")
                    logger.debug(f"Full AI analysis prompt:\n{prompt}")

                # Run the agent analysis with enhanced monitoring
                logger.info("WORKFLOW: Starting AI agent analysis execution")
                logger.info(f"WORKFLOW: Analysis prompt prepared (length: {len(prompt)} chars)")

                try:
                    result_output, tool_calls = _execute_ai_analysis(agent, prompt)
                    logger.info("WORKFLOW: AI agent analysis completed successfully")
                    logger.info(f"WORKFLOW: Analysis returned {len(tool_calls) if tool_calls else 0} tool calls")
                except Exception as e:
                    logger.error(f"WORKFLOW: AI agent analysis failed: {str(e)}", exc_info=True)
                    st.error(f"AI analysis failed: {str(e)}")
                    return

                # Get tool calls for display
                log_file = get_current_log_file()
                if log_file and log_file.exists():
                    display_tool_calls = get_ai_tool_logs_for_display()
                else:
                    display_tool_calls = []

                # Run tool relevance analysis during the "Displaying analysis results" spinner
                relevant_tool_calls = None
                if display_tool_calls:
                    try:
                        relevant_log_file, relevance_map = run_tool_relevance_analysis_sync(
                            display_tool_calls, result_output.interpretation
                        )
                        relevant_tool_calls = get_relevant_tool_calls()
                        logger.info(f"Tool relevance analysis completed: {len(relevant_tool_calls) if relevant_tool_calls else 0}/{len(display_tool_calls)} tools deemed relevant")
                    except Exception as e:
                        logger.error(f"Tool relevance analysis failed: {e}")
                        relevant_tool_calls = None

                # Cache the results
                analysis_id = _generate_analysis_id(research_query, selected_contrast_dicts, selected_datasets)
                _cache_ai_analysis(
                    analysis_id, ri, result_output, research_query, selected_contrast_dicts,
                    results_df, selected_datasets, selected_contrasts, display_tool_calls, relevant_tool_calls
                )

        # Display unified results
        progress_placeholder.empty()
        logger.info("WORKFLOW: Displaying unified AI results")
        _display_unified_ai_results(
            ri, result_output, research_query, selected_contrast_dicts,
            results_df, selected_datasets, selected_contrasts, display_tool_calls, relevant_tool_calls
        )
        logger.info("WORKFLOW: AI analysis workflow completed successfully")

    except Exception as e:
        logger.error(f"WORKFLOW: Critical error in AI analysis workflow: {str(e)}", exc_info=True)
        st.error(f"Analysis failed: {str(e)}")

        # Log the state for debugging
        logger.error(f"WORKFLOW: Failure context - research_query: '{research_query}', selected_datasets: {len(selected_datasets)}")
        if 'selected_contrasts_for_ai' in st.session_state:
            logger.error(f"WORKFLOW: Session state had {len(st.session_state['selected_contrasts_for_ai'])} selected contrasts")
        else:
            logger.error("WORKFLOW: No 'selected_contrasts_for_ai' in session state")

        with st.expander("Error Details", expanded=False):
            import traceback
            st.code(traceback.format_exc())


def _initialise_ai_cache():
    """Initialise the AI analysis caching system."""
    if 'ai_analysis_cache' not in st.session_state:
        st.session_state['ai_analysis_cache'] = {}
    if 'current_analysis_id' not in st.session_state:
        st.session_state['current_analysis_id'] = None
    if 'show_cached_results' not in st.session_state:
        st.session_state['show_cached_results'] = False
    if 'just_ran_fresh_analysis' not in st.session_state:
        st.session_state['just_ran_fresh_analysis'] = False


def _generate_analysis_id(research_question: str, selected_contrasts: List[Dict], selected_datasets: List[str]) -> str:
    """Generate unique ID for analysis based on question, contrasts, and datasets."""
    import hashlib
    # Create a hashable representation of contrasts
    contrast_keys = []
    for contrast in selected_contrasts:
        key = f"{contrast.get('analysis_id', '')}_{contrast.get('contrast_id', '')}"
        contrast_keys.append(key)
    contrast_hash = hash(str(sorted(contrast_keys)))
    # Include datasets in the cache key to ensure cache invalidation when datasets change
    datasets_hash = hash(str(sorted(selected_datasets)))
    content = f"{research_question}_{len(selected_contrasts)}_{contrast_hash}_{datasets_hash}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _cache_ai_analysis(
    analysis_id: str,
    ri: ResultsIntegrator,
    result_output: Any,
    research_question: str,
    selected_contrast_dicts: List[Dict],
    results_df: pd.DataFrame,
    selected_datasets: List[str],
    selected_contrasts: Optional[List] = None,
    tool_calls: Optional[List[Dict]] = None,
    relevant_tool_calls: Optional[List[Dict]] = None
):
    """Cache AI analysis results."""
    st.session_state['ai_analysis_cache'][analysis_id] = {
        'result_output': result_output,
        'research_question': research_question,
        'selected_contrast_dicts': selected_contrast_dicts,
        'results_df': results_df,
        'selected_datasets': selected_datasets,
        'selected_contrasts': selected_contrasts,
        'tool_calls': tool_calls,
        'relevant_tool_calls': relevant_tool_calls,
        'cached_at': datetime.now(),
        'cache_key': analysis_id
    }
    st.session_state['current_analysis_id'] = analysis_id
    st.session_state['show_cached_results'] = True
    st.session_state['just_ran_fresh_analysis'] = True  # Flag to prevent immediate cached display


def _should_restore_cached_analysis(selected_datasets: List[str]) -> bool:
    """
    Check if we should restore a cached analysis.

    This function prevents duplicate analysis display by using the 'just_ran_fresh_analysis' flag:

    1. When a fresh analysis completes, _cache_ai_analysis() sets just_ran_fresh_analysis=True
    2. The first call to this function after fresh analysis returns False (preventing duplicate display)
    3. The flag is cleared so subsequent calls (e.g., after page refresh/rerun) return True
    4. This ensures cached results only show when restoring from previous sessions, not immediately after fresh analysis

    Flow:
    - Fresh analysis runs → flag=True → this returns False (no duplicate)
    - User downloads/refreshes → flag=False → this returns True (show cached results)
    """
    # Don't show cached results if we just ran a fresh analysis in this session
    if st.session_state.get('just_ran_fresh_analysis', False):
        # Clear the flag for next time
        st.session_state['just_ran_fresh_analysis'] = False
        return False

    # Check if cached analysis exists and was done with the same datasets
    cache_valid = (
        st.session_state.get('show_cached_results', False) and
        st.session_state.get('current_analysis_id') is not None and
        st.session_state.get('current_analysis_id') in st.session_state.get('ai_analysis_cache', {})
    )

    if cache_valid:
        # Check if cached analysis was done with same selected datasets
        analysis_id = st.session_state['current_analysis_id']
        cached_data = st.session_state['ai_analysis_cache'][analysis_id]
        cached_datasets = cached_data.get('selected_datasets', [])
        if set(cached_datasets) != set(selected_datasets):
            # Datasets changed - invalidate cache
            st.session_state['show_cached_results'] = False
            st.session_state['current_analysis_id'] = None
            return False

    return cache_valid


def _restore_and_display_cached_analysis(ri: ResultsIntegrator, results_dir: str, selected_datasets: List[str]):
    """Restore and display cached analysis results."""
    analysis_id = st.session_state['current_analysis_id']
    cached_data = st.session_state['ai_analysis_cache'][analysis_id]

    # Display the cached results
    _display_unified_ai_results(
        ri=ri,
        result_output=cached_data['result_output'],
        research_question=cached_data['research_question'],
        selected_contrast_dicts=cached_data['selected_contrast_dicts'],
        results_df=cached_data['results_df'],
        selected_datasets=cached_data['selected_datasets'],
        selected_contrasts=cached_data['selected_contrasts'],
        tool_calls=cached_data['tool_calls'],
        relevant_tool_calls=cached_data.get('relevant_tool_calls')
    )


def _display_relevance_results(ri: ResultsIntegrator, results_df, research_query: str):
    """Display the contrast relevance assessment results."""
    # Sort by relevance score
    results_df = results_df.sort_values('RelevanceScore', ascending=False)

    # Add contrast descriptions for display
    results_df['Description'] = results_df.apply(
        lambda row: ri._get_contrast_description(row['analysis_id'], row['contrast_id']),
        axis=1
    )

    # Add accession info
    results_df['Accession'] = results_df['analysis_id'].map(
        lambda aid: ri.analysis_info.get(aid, {}).get('accession', aid)
    )

    # Add organism info
    results_df['organism'] = results_df['analysis_id'].map(
        lambda aid: ri.analysis_info.get(aid, {}).get('organism', 'Unknown')
    )

    log_streamlit_event(f"Contrast relevance assessment completed: {len(results_df)} contrasts scored")
    st.success(f"Successfully assessed {len(results_df)} contrasts!")

    # Convert to consistent contrast names for AI
    selected_contrast_dicts = []
    for _, row in results_df.iterrows():
        analysis_id = row['analysis_id']
        contrast_id = row['contrast_id']

        # Find the consistent name from ri.contrast_info
        consistent_name = contrast_id
        for contrast_key, contrast_data in ri.contrast_info.items():
            if (contrast_data.get('original_name') == contrast_id and
                contrast_data.get('analysis_id') == analysis_id):
                consistent_name = contrast_data.get('name', contrast_id)
                break

        selected_contrast_dicts.append({
            'analysis_id': analysis_id,
            'contrast_id': consistent_name
        })

    st.session_state['selected_contrasts_for_ai'] = selected_contrast_dicts
    st.session_state['research_query'] = research_query

    # Display results table
    st.subheader("Contrast Relevance Scores")

    # Add Selected column based on session state
    sel = st.session_state.get("selected_contrasts_for_ai", [])
    sel_set = {(item['analysis_id'], item['contrast_id']) for item in sel}
    results_df = results_df.copy()
    results_df["Selected"] = results_df.apply(
        lambda r: (r['analysis_id'], r['contrast_id']) in sel_set, axis=1
    )

    # Configure column display
    display_columns = ['Selected', 'Accession', 'contrast_id', 'RelevanceScore', 'Description']
    if 'Run1Justification' in results_df.columns:
        display_columns.append('Run1Justification')

    st.dataframe(
        results_df[display_columns],
        use_container_width=True,
        column_config={
            "Selected": st.column_config.CheckboxColumn("Selected"),
            "RelevanceScore": st.column_config.NumberColumn(
                "Relevance Score",
                format="%.2f",
                help="AI-assessed relevance score (0-1 scale)"
            ),
            "contrast_id": st.column_config.TextColumn(
                "Contrast",
                help="Contrast identifier"
            ),
            "Description": st.column_config.TextColumn(
                "Description",
                help="Contrast description"
            ),
            "Run1Justification": st.column_config.TextColumn(
                "AI Justification",
                help="AI explanation for the relevance score"
            ),
            "Accession": st.column_config.TextColumn(
                "Dataset",
                help="Dataset accession"
            )
        }
    )

    # Provide download option
    _provide_relevance_download(results_df)





def _display_relevance_and_selection_results(
    ri: ResultsIntegrator,
    results_df: pd.DataFrame,
    selected_contrasts: List[SelectedContrast],
    research_query: str
):
    """Display the AI-selected contrasts for analysis."""

    # Sort by relevance score
    results_df = results_df.sort_values('RelevanceScore', ascending=False)

    # Add contrast descriptions for display
    results_df['Description'] = results_df.apply(
        lambda row: ri._get_contrast_description(row['analysis_id'], row['contrast_id']),
        axis=1
    )

    # Add accession and organism info
    results_df['Accession'] = results_df['analysis_id'].map(
        lambda aid: ri.analysis_info.get(aid, {}).get('accession', aid)
    )
    results_df['organism'] = results_df['analysis_id'].map(
        lambda aid: ri.analysis_info.get(aid, {}).get('organism', 'Unknown')
    )

    log_streamlit_event(f"Contrast relevance with selection completed: {len(results_df)} assessed, {len(selected_contrasts)} selected")

    st.subheader("AI-Selected Contrasts for Analysis")
    st.markdown("*Intelligently chosen subset for detailed analysis*")

    # Create DataFrame for selected contrasts
    selected_data = []
    for sc in selected_contrasts:
        # Find the relevance info
        relevance_row = results_df[
            (results_df['analysis_id'] == sc.analysis_id) &
            (results_df['contrast_id'] == sc.contrast_id)
        ]

        if not relevance_row.empty:
            selected_data.append({
                'Dataset': relevance_row.iloc[0]['Accession'],
                'Contrast': sc.contrast_id,
                'Justification': sc.selection_justification
            })

    if selected_data:
        selected_df = pd.DataFrame(selected_data)

        # Display with simple formatting
        st.dataframe(
            selected_df,
            use_container_width=True,
            column_config={
                "Dataset": st.column_config.TextColumn("Dataset", help="Dataset accession (GSE)"),
                "Contrast": st.column_config.TextColumn("Contrast", help="Contrast identifier"),
                "Justification": st.column_config.TextColumn("Justification", help="Why selected for analysis")
            }
        )

    # Provide download option for selected contrasts data
    _provide_relevance_download(results_df)


def _provide_relevance_download(results_df):
    """Provide download option for relevance results."""
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Relevance Scores as CSV",
        data=csv,
        file_name=f"contrast_relevance_scores.csv",
        mime="text/csv"
    )


@log_streamlit_function
@log_streamlit_agent
def _execute_ai_analysis(agent, prompt: str) -> Tuple[GeneAnalysisOutput, List[Dict]]:
    """Execute the AI analysis asynchronously with proper loop hygiene, timeout, and capture tool calls."""

    async def run_analysis():
        logger.info("AI EXECUTION: Starting MCP server connection")
        async with agent.run_mcp_servers():
            logger.info("AI EXECUTION: MCP servers connected successfully")

            # Start new analysis session and clear previous tool calls
            start_ai_analysis_session()
            clear_relevant_tool_log()
            logger.info("AI EXECUTION: Analysis session initialized")

            # Run main analysis with timeout
            ai_config = get_ai_agent_config()
            logger.info(f"AI EXECUTION: Starting analysis with request_limit={ai_config.request_limit}, timeout={ai_config.analysis_timeout}s")

            try:
                # Add progress tracking for long-running operations
                import time
                start_time = time.time()

                result = await asyncio.wait_for(
                    agent.run(prompt, usage_limits=UsageLimits(request_limit=ai_config.request_limit)),
                    timeout=ai_config.analysis_timeout
                )

                elapsed_time = time.time() - start_time
                logger.info(f"AI EXECUTION: Analysis completed in {elapsed_time:.2f} seconds")

            except UsageLimitExceeded as e:
                logger.error(f"AI EXECUTION: Usage limit exceeded: {e}")
                raise Exception(f"Analysis stopped: Request limit of {ai_config.request_limit} exceeded. {e}")
            except asyncio.TimeoutError:
                logger.error(f"AI EXECUTION: Analysis timeout after {ai_config.analysis_timeout}s")
                # Try to get partial tool calls before timing out
                try:
                    partial_tool_calls = get_ai_tool_logs_for_display()
                    logger.warning(f"AI EXECUTION: Partial analysis had {len(partial_tool_calls) if partial_tool_calls else 0} tool calls before timeout")
                except:
                    logger.warning("AI EXECUTION: Could not retrieve partial tool calls after timeout")
                raise Exception(f"Analysis timed out after {ai_config.analysis_timeout} seconds. The AI may have gotten stuck in a loop or encountered a complex analysis.")

            # Get tool calls from new logging system
            tool_calls = get_ai_tool_logs_for_display()
            logger.info(f"AI EXECUTION: Retrieved {len(tool_calls) if tool_calls else 0} tool calls from analysis")

            # Return the structured output directly (GeneAnalysisOutput object)
            return result.output if hasattr(result, 'output') else result, tool_calls

    try:
        logger.info("AI EXECUTION: Checking event loop status")
        # Check if there's already a running event loop
        try:
            loop = asyncio.get_running_loop()
            logger.info("AI EXECUTION: Running in existing event loop, using thread executor")
            # If we're in an existing loop, we need to run in a thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, run_analysis())
                # Add progress monitoring for thread execution
                import time
                start_time = time.time()
                try:
                    result_output, tool_calls = future.result(timeout=1520)  # Slightly longer than async timeout
                    elapsed_time = time.time() - start_time
                    logger.info(f"AI EXECUTION: Thread execution completed in {elapsed_time:.2f} seconds")
                except concurrent.futures.TimeoutError:
                    logger.error("AI EXECUTION: Thread executor timeout - analysis likely hung")
                    raise
        except RuntimeError:
            logger.info("AI EXECUTION: No running event loop, using asyncio.run")
            # No running loop, safe to use asyncio.run
            result_output, tool_calls = asyncio.run(run_analysis())

        logger.info("AI EXECUTION: Analysis execution completed successfully")
        return result_output, tool_calls

    except asyncio.TimeoutError:
        logger.warning("AI EXECUTION: Analysis timed out (asyncio.TimeoutError)")
        st.error("AI analysis timed out. The analysis may have encountered a complex query or gotten stuck. Please try again with a simpler query or fewer contrasts.")
        st.stop()
    except KeyboardInterrupt:
        logger.info("AI EXECUTION: Analysis interrupted by user")
        st.warning("Analysis stopped by user.")
        st.stop()
    except concurrent.futures.TimeoutError:
        logger.warning("AI EXECUTION: Analysis timed out in thread executor - likely hung during execution")
        st.error("AI analysis timed out during execution. This often happens when the AI gets stuck in complex analysis loops. Please try again with a simpler query or fewer contrasts.")
        st.stop()
    except Exception as e:
        logger.error(f"AI EXECUTION: Unexpected error in analysis execution: {str(e)}", exc_info=True)
        st.error(f"❌ AI analysis failed: {str(e)}")
        st.stop()


def _display_unified_ai_results(
    ri: ResultsIntegrator,
    result_output: GeneAnalysisOutput,
    research_question: str,
    selected_contrast_dicts: List[Dict],
    results_df: pd.DataFrame,
    selected_datasets: List[str],
    selected_contrasts: Optional[List] = None,
    tool_calls: Optional[List[Dict]] = None,
    relevant_tool_calls: Optional[List[Dict]] = None
):
    """Display unified AI analysis results with tabbed interface."""
    log_streamlit_event("AI analysis completed successfully")

    # Use the structured output directly
    try:
        parsed = result_output
    except Exception as e:
        st.error(f"Error processing AI output: {e}")
        st.subheader("Raw AI Response")
        st.markdown(str(result_output))
        return

    genes = set(parsed.genes)  # normalise + dedupe
    contrasts = [(item['analysis_id'], item['contrast_id']) for item in selected_contrast_dicts]

    # --- AI Interpretation (at the very top)
    st.subheader("Analysis Results")
    st.markdown("*The following text was generated by the AI agent, informed by the agent's analysis of the underlying data.*")
    st.markdown(parsed.interpretation)

    # Tool relevance analysis is now handled before this function is called
    # relevant_tool_calls is passed as a parameter

    # Display tool relevance analysis results
    if tool_calls and relevant_tool_calls is not None:
        relevant_count = len(relevant_tool_calls)
        total_count = len(tool_calls)
    elif tool_calls and relevant_tool_calls is None:
        st.warning("Tool relevance analysis failed - showing all tool logs")

    # Create tabs for organized display
    gene_tab, contrast_tab, heatmap_tab, table_tab, logs_tab, downloads_tab = st.tabs([
        "Selected Genes",
        "Selected Contrasts",
        "Heatmap",
        "Expression Data",
        "Tool Logs",
        "Download Data"
    ])

    # Prepare data for all tabs
    if ri and ri.deg_data:
        rows, missing = [], []
        for gene in genes:
            hit = False
            for aid, cid in contrasts:
                if aid in ri.deg_data and cid in ri.deg_data[aid]:
                    df = ri.deg_data[aid][cid]
                    if 'Gene' in df.columns and gene in df['Gene'].values:
                        hit = True
                        rec = df.loc[df['Gene'] == gene].iloc[0]
                        rows.append({
                            'Gene': gene,
                            'Dataset': aid,
                            'Contrast': cid,
                            'logFC': rec.get("logFC", 0),
                            'P-value': rec.get("adj.P.Val") or rec.get("P.Value", 1)
                        })
            if not hit:
                missing.append(gene)
    else:
        rows, missing = [], []

    # Tab 1: Gene List
    with gene_tab:
        st.subheader("Selected Genes")
        st.markdown("*The following genes were chosen by the agent.*")

        # Display genes in a copyable format
        sorted_genes = sorted(genes)
        genes_text = ', '.join(sorted_genes)

        st.markdown("**Gene List:**")
        st.code(genes_text, language=None)

        st.write(f"**Total:** {len(sorted_genes)} genes")

        # Download gene list
        gene_list_csv = pd.DataFrame({'Gene': sorted_genes}).to_csv(index=False)
        gene_download_key = f"download_genes_tab1_{_generate_analysis_id(research_question, selected_contrast_dicts, selected_datasets)}_{int(time.time() * 1000)}"

        st.download_button(
            label="Download Gene List (CSV)",
            data=gene_list_csv,
            file_name="selected_genes.csv",
            mime="text/csv",
            help="Download the list of selected genes",
            key=gene_download_key
        )

    # Tab 4: Contrast Selection
    with contrast_tab:
        st.subheader("Selected Contrasts")
        st.markdown("*The following contrasts were chosen by the agent.*")

        if selected_contrasts and hasattr(selected_contrasts[0], 'selection_justification'):

            # Create DataFrame for selected contrasts
            selected_data = []
            for sc in selected_contrasts:
                # Find the relevance info
                relevance_row = results_df[
                    (results_df['analysis_id'] == sc.analysis_id) &
                    (results_df['contrast_id'] == sc.contrast_id)
                ]

                if not relevance_row.empty:
                    selected_data.append({
                        'Dataset': relevance_row.iloc[0].get('Accession', sc.analysis_id),
                        'Contrast': sc.contrast_id,
                        'Justification': sc.selection_justification
                    })

            if selected_data:
                selected_df = pd.DataFrame(selected_data)
                st.dataframe(
                    selected_df,
                    use_container_width=True,
                    column_config={
                        "Dataset": st.column_config.TextColumn("Dataset", help="Dataset accession (GSE)"),
                        "Contrast": st.column_config.TextColumn("Contrast", help="Contrast identifier"),
                        "Justification": st.column_config.TextColumn("Justification", help="Why selected for analysis")
                    }
                )
        else:
            # Show basic contrast list
            st.markdown("*Contrasts analysed*")

            # Prepare contrast display data
            contrast_display_data = []
            for item in selected_contrast_dicts:
                aid = item['analysis_id']
                cid = item['contrast_id']

                # Get additional info if available
                accession = ri.analysis_info.get(aid, {}).get('accession', aid) if ri else aid
                description = ri._get_contrast_description(aid, cid) if ri else ''

                contrast_display_data.append({
                    'Dataset': accession,
                    'Contrast': cid,
                    'Description': description
                })

            if contrast_display_data:
                contrast_df = pd.DataFrame(contrast_display_data)
                st.dataframe(
                    contrast_df,
                    use_container_width=True,
                    column_config={
                        "Dataset": st.column_config.TextColumn("Dataset", help="Dataset accession (GSE)"),
                        "Contrast": st.column_config.TextColumn("Contrast", help="Contrast identifier"),
                        "Description": st.column_config.TextColumn("Description", help="Contrast description")
                    }
                )

        # Download the contrast table shown in this tab
        if selected_contrasts and hasattr(selected_contrasts[0], 'selection_justification'):
            if selected_data:
                contrast_table_csv = pd.DataFrame(selected_data).to_csv(index=False)
                contrast_download_key = f"download_contrasts_selected_tab4_{_generate_analysis_id(research_question, selected_contrast_dicts, selected_datasets)}_{int(time.time() * 1000)}"

                st.download_button(
                    label="Download Selected Contrasts (CSV)",
                    data=contrast_table_csv,
                    file_name="selected_contrasts.csv",
                    mime="text/csv",
                    help="Download the selected contrasts table",
                    key=contrast_download_key
                )
        else:
            if contrast_display_data:
                contrast_table_csv = pd.DataFrame(contrast_display_data).to_csv(index=False)
                contrast_download_key = f"download_contrasts_analysed_tab4_{_generate_analysis_id(research_question, selected_contrast_dicts, selected_datasets)}_{int(time.time() * 1000)}"

                st.download_button(
                    label="Download Analysed Contrasts (CSV)",
                    data=contrast_table_csv,
                    file_name="analysed_contrasts.csv",
                    mime="text/csv",
                    help="Download the contrasts table",
                    key=contrast_download_key
                )


    # Tab 2: Heatmap
    with heatmap_tab:
        st.subheader("Heatmap")
        st.markdown("*The heatmap reflects the underlying data. The genes and contrasts were chosen by the agent.*")

        if rows and len(genes - set(missing)) > 0:
            try:
                fig = ri.create_lfc_heatmap(
                    genes=list(genes - set(missing)),
                    contrasts=contrasts
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Could not generate heatmap for the selected genes.")
            except Exception as e:
                st.warning(f"Error generating heatmap: {str(e)}")
        else:
            st.warning("No expression data available for heatmap visualisation.")

    # Tab 3: Expression Table
    with table_tab:
        st.subheader("Expression Data")
        st.markdown("*The table reflects the underlying data. The genes and contrasts were chosen by the agent.*")

        if rows:
            lfc_df = pd.DataFrame(rows)
            st.dataframe(
                lfc_df,
                use_container_width=True,
                column_config={
                    "logFC": st.column_config.NumberColumn("Log2FC", format="%.2f"),
                    "P-value": st.column_config.NumberColumn("P-value", format="%.2e")
                }
            )

            # Download this specific table
            expression_csv = lfc_df.to_csv(index=False)
            expression_download_key = f"download_expression_tab3_{_generate_analysis_id(research_question, selected_contrast_dicts, selected_datasets)}_{int(time.time() * 1000)}"

            st.download_button(
                label="Download Expression Table (CSV)",
                data=expression_csv,
                file_name="gene_expression_data.csv",
                mime="text/csv",
                help="Download this expression data table",
                key=expression_download_key
            )

        else:
            st.info("No expression data found for the selected genes in the chosen contrasts.")

        if missing:
            st.info(f"{len(missing)} gene(s) not present in the selected contrasts: {', '.join(missing)}")


    # Tab 5: Tool Logs
    with logs_tab:
        st.subheader("Tool Logs")
        st.markdown("*The following describes the tools used by the agent. While the agent chooses which tools to use and how to use them, each tool and its output can be replicated with the provided code.*")

        if tool_calls:
            # Create sub-tabs for relevant vs full tool logs
            if relevant_tool_calls is not None:
                relevant_tab, full_tab = st.tabs(["Relevant Tool Logs", "Full Tool Logs"])

                with relevant_tab:
                    st.markdown("*Tools that contributed to the final analysis interpretation*")
                    if relevant_tool_calls:
                        _display_tool_calls_detailed(relevant_tool_calls)
                    else:
                        st.info("No tools were deemed directly relevant to the final interpretation. This may indicate the analysis was primarily exploratory.")

                with full_tab:
                    st.markdown("*Complete log of all tools used during analysis*")
                    _display_tool_calls_detailed(tool_calls)
            else:
                # Fallback to showing all tool calls if relevance analysis failed
                _display_tool_calls_detailed(tool_calls)
        else:
            st.info("No tool calls found.")

    # Tab 6: Downloads
    with downloads_tab:
        st.subheader("Download Underlying Data")
        st.markdown("*Raw data files and detailed analysis outputs.*")

        col1, col2 = st.columns(2)

        with col1:
            # Analysis report
            analysis_report = f"""
# AI Gene Analysis Report

**Research Question:** {research_question}

**Contrasts Analysed:** {len(selected_contrast_dicts)}

## Selected Genes
{', '.join(sorted(genes))}

## AI Interpretation

{parsed.interpretation}

---
*Generated by UORCA Explorer AI Assistant*
"""

            report_download_key = f"download_report_tab6_{_generate_analysis_id(research_question, selected_contrast_dicts, selected_datasets)}_{int(time.time() * 1000)}"

            st.download_button(
                label="Complete Analysis Report (Markdown)",
                data=analysis_report,
                file_name="uorca_ai_gene_analysis.md",
                mime="text/markdown",
                help="Download the complete analysis report",
                key=report_download_key
            )

            # AI agent's working dataset
            try:
                filtered_df = get_filtered_dataframe()

                if not filtered_df.empty:
                    csv_data = filtered_df.to_csv(index=False)
                    agent_data_key = f"download_agent_data_tab6_{_generate_analysis_id(research_question, selected_contrast_dicts, selected_datasets)}_{int(time.time() * 1000)}"

                    st.download_button(
                        label="AI Agent's Working Dataset (CSV)",
                        data=csv_data,
                        file_name="ai_agent_working_dataset.csv",
                        mime="text/csv",
                        help="The exact dataset the AI agent used for analysis",
                        key=agent_data_key
                    )
                else:
                    st.info("AI agent dataset not available")
            except Exception as e:
                st.info("AI agent dataset not available")

        with col2:
            # Contrast relevance scores
            if not results_df.empty:
                relevance_csv = results_df.to_csv(index=False)
                relevance_key = f"download_all_relevance_tab6_{_generate_analysis_id(research_question, selected_contrast_dicts, selected_datasets)}_{int(time.time() * 1000)}"

                st.download_button(
                    label="All Contrast Relevance Scores (CSV)",
                    data=relevance_csv,
                    file_name="contrast_relevance_scores.csv",
                    mime="text/csv",
                    help="Complete contrast relevance assessment results",
                    key=relevance_key
                )

            # Raw tool logs
            if tool_calls:
                # Create a text log of all tool calls
                log_content = "AI Tool Usage Log\n"
                log_content += "=" * 50 + "\n\n"

                for i, call in enumerate(tool_calls, 1):
                    log_content += f"Tool Call #{i}: {call.get('tool_name', 'Unknown')}\n"
                    log_content += f"Timestamp: {call.get('timestamp', 'Unknown')}\n"
                    log_content += f"Parameters: {call.get('parameters', {})}\n"
                    log_content += f"Success: {call.get('success', True)}\n"
                    if call.get('full_output'):
                        log_content += f"Output: {call.get('full_output')}\n"
                    elif call.get('output_snippet'):
                        log_content += f"Output: {call.get('output_snippet')}\n"
                    log_content += "-" * 30 + "\n\n"

                raw_log_key = f"download_raw_log_tab6_{_generate_analysis_id(research_question, selected_contrast_dicts, selected_datasets)}_{int(time.time() * 1000)}"

                st.download_button(
                    label="Raw Tool Usage Log (Text)",
                    data=log_content,
                    file_name="ai_tool_usage_log.txt",
                    mime="text/plain",
                    help="Detailed log of all AI tool usage",
                    key=raw_log_key
                )





def _get_tool_description(tool_name: str) -> str:
    """Get user-friendly description for each tool."""
    descriptions = {
        'get_most_common_genes': "Finds genes that show up as significantly changed across many different experimental conditions. This helps identify genes that are consistently important across your datasets.",

        'filter_genes_by_contrast_sets': "Compares two groups of experiments to find genes that are significant in one group but not the other. This helps identify genes that are specific to certain conditions or treatments.",

        'get_gene_contrast_stats': "Looks up detailed information about how multiple genes behave across different experimental conditions, including how much they changed and how confident we are in those changes.",

        'summarise_contrast': "Provides an overview of a specific experimental comparison, including the most important genes and overall patterns of change.",

        'calculate_gene_correlation': "Calculates how similarly genes are expressed across different experimental conditions using their average expression levels. This helps identify genes that work together in biological pathways or are co-regulated.",

        'calculate_expression_variability': "Measures how consistently genes change across different experimental conditions by calculating the variability of their fold changes. Low variability indicates reliable, consistent responses while high variability suggests context-dependent regulation."
    }
    return descriptions.get(tool_name, "This tool analyses your gene expression data to provide insights.")

def _get_tool_code_snippets(tool_name: str, parameters: dict) -> tuple:
    """Get R and Python code snippets to reproduce tool functionality."""

    if tool_name == 'get_most_common_genes':
        lfc_thresh = parameters.get('lfc_thresh', 1.0)
        p_thresh = parameters.get('p_thresh', 0.05)
        top_n = parameters.get('top_n', 10)

        r_code = f'''# Load data (download "AI Agent's Working Dataset" from Downloads tab)
data <- read.csv("ai_agent_working_dataset.csv")

# Filter for significant genes
significant_genes <- data[abs(data$logFC) >= {lfc_thresh} & data$pvalue < {p_thresh}, ]

# Count occurrences of each gene
gene_counts <- table(significant_genes$Gene)

# Get top {top_n} most common genes
top_genes <- sort(gene_counts, decreasing = TRUE)[1:{top_n}]
print(top_genes)'''

        python_code = f'''# Load data (download "AI Agent's Working Dataset" from Downloads tab)
import pandas as pd

data = pd.read_csv("ai_agent_working_dataset.csv")

# Filter for significant genes
significant_genes = data[
    (abs(data['logFC']) >= {lfc_thresh}) &
    (data['pvalue'] < {p_thresh})
]

# Count occurrences of each gene
gene_counts = significant_genes['Gene'].value_counts()

# Get top {top_n} most common genes
top_genes = gene_counts.head({top_n})
print(top_genes)'''

    elif tool_name == 'filter_genes_by_contrast_sets':
        set_a = parameters.get('set_a', [])
        set_b = parameters.get('set_b', [])
        lfc_thresh = parameters.get('lfc_thresh', 1.0)
        p_thresh = parameters.get('p_thresh', 0.05)

        set_a_str = "c(" + ", ".join([f'"{x}"' for x in set_a]) + ")"
        set_b_str = "c(" + ", ".join([f'"{x}"' for x in set_b]) + ")"
        set_a_py = str(set_a)
        set_b_py = str(set_b)

        r_code = f'''# Load data
data <- read.csv("ai_agent_working_dataset.csv")

# Define contrast sets
set_a <- {set_a_str}
set_b <- {set_b_str}

# Find genes significant in ALL contrasts of set A
sig_data_a <- data[data$contrast_id %in% set_a &
                   abs(data$logFC) >= {lfc_thresh} &
                   data$pvalue < {p_thresh}, ]

# Count how many set A contrasts each gene appears in
gene_counts_a <- table(sig_data_a$Gene)
# Keep only genes that appear in ALL set A contrasts
genes_all_a <- names(gene_counts_a[gene_counts_a == length(set_a)])

# Filter for significant genes in ANY contrast of set B
genes_b <- data[data$contrast_id %in% set_b &
                abs(data$logFC) >= {lfc_thresh} &
                data$pvalue < {p_thresh}, "Gene"]

# Find genes unique to set A (significant in ALL A, but not in any B)
unique_to_a <- setdiff(genes_all_a, unique(genes_b))
print(paste("Genes unique to set A:", length(unique_to_a)))
print(unique_to_a)'''

        python_code = f'''# Load data
import pandas as pd

data = pd.read_csv("ai_agent_working_dataset.csv")

# Define contrast sets
set_a = {set_a_py}
set_b = {set_b_py}

# Find genes significant in ALL contrasts of set A
sig_data_a = data[
    (data['contrast_id'].isin(set_a)) &
    (abs(data['logFC']) >= {lfc_thresh}) &
    (data['pvalue'] < {p_thresh})
]

# Count how many set A contrasts each gene appears in
gene_counts_a = sig_data_a.groupby('Gene')['contrast_id'].nunique()
# Keep only genes that appear in ALL set A contrasts
genes_all_a = set(gene_counts_a[gene_counts_a == len(set_a)].index)

# Filter for significant genes in ANY contrast of set B
genes_b = set(data[
    (data['contrast_id'].isin(set_b)) &
    (abs(data['logFC']) >= {lfc_thresh}) &
    (data['pvalue'] < {p_thresh})
]['Gene'])

# Find genes unique to set A (significant in ALL A, but not in any B)
unique_to_a = genes_all_a - genes_b
print(f"Genes unique to set A: {{len(unique_to_a)}}")
print(list(unique_to_a))'''

    elif tool_name == 'get_gene_contrast_stats':
        genes = parameters.get('genes', ['GENE1'])
        contrast_ids = parameters.get('contrast_ids', None)

        genes_r = f"c({', '.join([f'\"{g}\"' for g in genes])})"
        genes_py = [f'"{g}"' for g in genes]

        if contrast_ids:
            contrasts_r = f"c({', '.join([f'\"{c}\"' for c in contrast_ids])})"
            contrasts_py = [f'"{c}"' for c in contrast_ids]
            r_filter = f'data$Gene %in% {genes_r} & data$contrast_id %in% {contrasts_r}'
            py_filter = f'(data["Gene"].isin({genes_py})) & (data["contrast_id"].isin({contrasts_py}))'
        else:
            r_filter = f'data$Gene %in% {genes_r}'
            py_filter = f'data["Gene"].isin({genes_py})'

        r_code = f'''# Load data
data <- read.csv("ai_agent_working_dataset.csv")

# Filter data for specified genes and contrasts
gene_stats <- data[{r_filter}, c("Gene", "contrast_id", "logFC", "pvalue")]
print(gene_stats)'''

        python_code = f'''# Load data
import pandas as pd

data = pd.read_csv("ai_agent_working_dataset.csv")

# Filter data for specified genes and contrasts
gene_stats = data[{py_filter}][["Gene", "contrast_id", "logFC", "pvalue"]]
print(gene_stats)'''

    elif tool_name == 'summarise_contrast':
        contrast_id = parameters.get('contrast_id', 'contrast1')
        lfc_thresh = parameters.get('lfc_thresh', 1.0)
        p_thresh = parameters.get('p_thresh', 0.05)
        max_genes = parameters.get('max_genes', 10)

        r_code = f'''# Load data
data <- read.csv("ai_agent_working_dataset.csv")

# Filter for specific contrast and significance
contrast_data <- data[data$contrast_id == "{contrast_id}" &
                      abs(data$logFC) >= {lfc_thresh} &
                      data$pvalue < {p_thresh}, ]

# Summary statistics
total_degs <- nrow(contrast_data)
mean_lfc <- mean(contrast_data$logFC)
median_lfc <- median(contrast_data$logFC)

# Top genes by absolute logFC
top_genes <- contrast_data[order(-abs(contrast_data$logFC)), ][1:{max_genes},
                          c("Gene", "logFC")]

cat("Total DEGs:", total_degs, "\\n")
cat("Mean logFC:", mean_lfc, "\\n")
cat("Median logFC:", median_lfc, "\\n")
print("Top genes:")
print(top_genes)'''

        python_code = f'''# Load data
import pandas as pd

data = pd.read_csv("ai_agent_working_dataset.csv")

# Filter for specific contrast and significance
contrast_data = data[
    (data['contrast_id'] == "{contrast_id}") &
    (abs(data['logFC']) >= {lfc_thresh}) &
    (data['pvalue'] < {p_thresh})
]

# Summary statistics
total_degs = len(contrast_data)
mean_lfc = contrast_data['logFC'].mean()
median_lfc = contrast_data['logFC'].median()

# Top genes by absolute logFC
top_genes = contrast_data.reindex(
    contrast_data['logFC'].abs().sort_values(ascending=False).index
).head({max_genes})[['Gene', 'logFC']]

print(f"Total DEGs: {{total_degs}}")
print(f"Mean logFC: {{mean_lfc}}")
print(f"Median logFC: {{median_lfc}}")
print("Top genes:")
print(top_genes)'''

    elif tool_name == 'calculate_gene_correlation':
        genes = parameters.get('genes', ['GENE1', 'GENE2'])
        genes_str = "c(" + ", ".join([f'"{g}"' for g in genes]) + ")"
        genes_py = str(genes)

        r_code = f'''# Load data
data <- read.csv("ai_agent_working_dataset.csv")

# Filter for specified genes
genes_of_interest <- {genes_str}
gene_data <- data[data$Gene %in% genes_of_interest, ]

# Create matrix with genes as columns and experiments as rows
library(reshape2)
expr_matrix <- dcast(gene_data, analysis_id + contrast_id ~ Gene, value.var = "AveExpr")

# Remove ID columns for correlation
expr_matrix_clean <- expr_matrix[, !(names(expr_matrix) %in% c("analysis_id", "contrast_id"))]

# Calculate Spearman correlation
correlation_matrix <- cor(expr_matrix_clean, method = "spearman", use = "complete.obs")
print("Correlation Matrix:")
print(correlation_matrix)

# Find strong correlations (> 0.5 or < -0.5)
strong_corr <- which(abs(correlation_matrix) > 0.5 & correlation_matrix != 1, arr.ind = TRUE)
cat("\\nStrong correlations:\\n")
for(i in 1:nrow(strong_corr)) {{
  gene1 <- rownames(correlation_matrix)[strong_corr[i,1]]
  gene2 <- colnames(correlation_matrix)[strong_corr[i,2]]
  corr_val <- correlation_matrix[strong_corr[i,1], strong_corr[i,2]]
  cat(gene1, "vs", gene2, ":", round(corr_val, 4), "\\n")
}}'''

        python_code = f'''# Load data
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

data = pd.read_csv("ai_agent_working_dataset.csv")

# Filter for specified genes
genes_of_interest = {genes_py}
gene_data = data[data['Gene'].isin(genes_of_interest)]

# Create pivot table with genes as columns
pivot_data = gene_data.pivot_table(
    index=['analysis_id', 'contrast_id'],
    columns='Gene',
    values='AveExpr',
    aggfunc='mean'
)

# Remove rows with too much missing data
pivot_data = pivot_data.dropna(thresh=len(pivot_data.columns)*0.5)

# Calculate Spearman correlation
correlation_matrix = pivot_data.corr(method='spearman')
print("Correlation Matrix:")
print(correlation_matrix)

# Find strong correlations (> 0.5 or < -0.5)
print("\\nStrong correlations:")
for gene1 in correlation_matrix.columns:
    for gene2 in correlation_matrix.columns:
        if gene1 != gene2:
            corr_val = correlation_matrix.loc[gene1, gene2]
            if not pd.isna(corr_val) and abs(corr_val) > 0.5:
                print(f"{{gene1}} vs {{gene2}}: {{corr_val:.4f}}")'''

    elif tool_name == 'calculate_expression_variability':
        genes = parameters.get('genes', ['GENE1'])
        contrasts = parameters.get('contrasts', None)
        genes_str = "c(" + ", ".join([f'"{g}"' for g in genes]) + ")"
        genes_py = str(genes)

        if contrasts:
            contrasts_str = "c(" + ", ".join([f'"{c}"' for c in contrasts]) + ")"
            contrasts_py = str(contrasts)
            contrast_filter_r = f'data$contrast_id %in% {contrasts_str} &'
            contrast_filter_py = f'(data["contrast_id"].isin({contrasts_py})) &'
        else:
            contrast_filter_r = ''
            contrast_filter_py = ''

        r_code = f'''# Load data
data <- read.csv("ai_agent_working_dataset.csv")

# Filter for specified genes{' and contrasts' if contrasts else ''}
genes_of_interest <- {genes_str}
filtered_data <- data[{contrast_filter_r}data$Gene %in% genes_of_interest, ]

# Calculate variability statistics for each gene
variability_stats <- data.frame()
for(gene in genes_of_interest) {{
  gene_subset <- filtered_data[filtered_data$Gene == gene, ]

  if(nrow(gene_subset) > 0) {{
    stats <- data.frame(
      Gene = gene,
      std_dev = sd(gene_subset$logFC, na.rm = TRUE),
      mean_logFC = mean(gene_subset$logFC, na.rm = TRUE),
      median_logFC = median(gene_subset$logFC, na.rm = TRUE),
      min_logFC = min(gene_subset$logFC, na.rm = TRUE),
      max_logFC = max(gene_subset$logFC, na.rm = TRUE),
      contrast_count = nrow(gene_subset)
    )
    variability_stats <- rbind(variability_stats, stats)
  }}
}}

# Sort by standard deviation (most consistent first)
variability_stats <- variability_stats[order(variability_stats$std_dev), ]
print(variability_stats)

cat("\\nMost consistent gene:", variability_stats$Gene[1], "\\n")
cat("Most variable gene:", variability_stats$Gene[nrow(variability_stats)], "\\n")'''

        python_code = f'''# Load data
import pandas as pd
import numpy as np

data = pd.read_csv("ai_agent_working_dataset.csv")

# Filter for specified genes{' and contrasts' if contrasts else ''}
genes_of_interest = {genes_py}
filtered_data = data[{contrast_filter_py}data['Gene'].isin(genes_of_interest)]

# Calculate variability statistics for each gene
variability_stats = []
for gene in genes_of_interest:
    gene_subset = filtered_data[filtered_data['Gene'] == gene]

    if len(gene_subset) > 0:
        stats = {{
            'Gene': gene,
            'std_dev': gene_subset['logFC'].std(),
            'mean_logFC': gene_subset['logFC'].mean(),
            'median_logFC': gene_subset['logFC'].median(),
            'min_logFC': gene_subset['logFC'].min(),
            'max_logFC': gene_subset['logFC'].max(),
            'contrast_count': len(gene_subset)
        }}
        variability_stats.append(stats)

# Convert to DataFrame and sort by standard deviation
variability_df = pd.DataFrame(variability_stats)
variability_df = variability_df.sort_values('std_dev')
print(variability_df)

if len(variability_df) > 0:
    print(f"\\nMost consistent gene: {{variability_df.iloc[0]['Gene']}}")
    print(f"Most variable gene: {{variability_df.iloc[-1]['Gene']}}")'''

    else:
        r_code = "# Code snippet not available for this tool"
        python_code = "# Code snippet not available for this tool"

    return r_code, python_code

def _display_tool_calls_detailed(tool_calls: List[Dict]):
    """Display tool calls in a detailed, structured format."""
    if not tool_calls:
        st.info("No tool calls to display.")
        return

    st.markdown("### Tool Call Details")

    for i, call in enumerate(tool_calls, 1):
        # Format success indicator
        status_icon = "FAILED" if not call.get('success', True) else ""
        tool_name = call.get('tool_name', 'Unknown Tool')
        timestamp = call.get('timestamp', 'Unknown time')

        # Create expander title without SUCCESS prefix
        expander_title = f"**{tool_name}** (Call #{i})"
        if status_icon:
            expander_title = f"{status_icon} {expander_title}"

        with st.expander(expander_title, expanded=False):

            # Add tool description at the top
            st.markdown(_get_tool_description(tool_name))
            r_code, python_code = _get_tool_code_snippets(tool_name, call.get('parameters', {}))

            code_tab1, code_tab2 = st.tabs(["R Code", "Python Code"])
            with code_tab1:
                st.code(r_code, language='r')
            with code_tab2:
                st.code(python_code, language='python')

            # Create two-column layout (25% parameters, 75% results)
            param_col, result_col = st.columns([1, 3])

            # Left column: Parameters
            with param_col:
                st.markdown("#### Parameters")
                if call.get('parameters'):
                    # Display parameters in a formatted way
                    for param_name, param_value in call['parameters'].items():
                        if isinstance(param_value, (list, dict)):
                            st.markdown(f"**{param_name}:**")
                            st.json(param_value)
                        else:
                            st.markdown(f"**{param_name}:** `{param_value}`")
                else:
                    st.write("*No parameters*")

            # Right column: Results
            with result_col:
                st.markdown("#### Results")
                if call.get('success', True):
                    output_snippet = call.get('output_snippet')
                    if output_snippet and output_snippet.strip():
                        # Get tool name and output for custom formatting
                        tool_name = call.get('tool_name', '')

                        # Use full_output for parsing if available, otherwise use snippet
                        full_output = call.get('full_output')
                        if full_output is not None:
                            output_for_parsing = full_output
                            is_truncated = "truncated" in output_snippet
                        else:
                            output_for_parsing = output_snippet
                            is_truncated = "truncated" in output_snippet



                        # Clean output for parsing (remove truncation text if present)
                        if is_truncated and full_output is None:
                            clean_output = output_snippet.split('[truncated')[0]
                        else:
                            clean_output = output_for_parsing

                        try:
                            # Parse Python literal output (not JSON)
                            import ast
                            if isinstance(call.get('output_snippet'), (dict, list)):
                                parsed_output = call.get('output_snippet')
                            elif clean_output.strip().startswith(('{', '[')):
                                # Use ast.literal_eval for Python literals with single quotes
                                parsed_output = ast.literal_eval(clean_output)
                            else:
                                parsed_output = None

                            # Custom formatting based on tool type
                            if tool_name == 'get_most_common_genes' and parsed_output:
                                # Display as table
                                if isinstance(parsed_output, list):
                                    df_data = []
                                    for item in parsed_output:
                                        if isinstance(item, dict) and 'gene' in item and 'count' in item:
                                            df_data.append({'Gene': item['gene'], 'Count': item['count']})
                                    if df_data:
                                        import pandas as pd
                                        df = pd.DataFrame(df_data)
                                        st.dataframe(df, use_container_width=True)
                                    else:
                                        st.code(clean_output)
                                else:
                                    st.code(clean_output)

                            elif tool_name == 'filter_genes_by_contrast_sets' and parsed_output:
                                # Extract and display just the gene list
                                if isinstance(parsed_output, dict) and 'genes' in parsed_output:
                                    genes = parsed_output['genes']
                                    if genes:
                                        # Display genes as wrapped text
                                        gene_text = ', '.join(genes)
                                        st.code(gene_text, language = 'r')
                                        st.caption(f"Total genes: {len(genes)}")
                                    else:
                                        st.write("No genes found")
                                else:
                                    st.code(clean_output)

                            elif tool_name == 'get_gene_contrast_stats':
                                # Display as dataframe or show empty message
                                if parsed_output and isinstance(parsed_output, list) and len(parsed_output) > 0:
                                    df_data = []
                                    for item in parsed_output:
                                        if isinstance(item, dict):
                                            df_data.append({
                                                'Gene': item.get('Gene', ''),
                                                'Contrast': item.get('contrast_id', ''),
                                                'logFC': item.get('logFC', ''),
                                                'P-value': item.get('pvalue', '')
                                            })
                                    if df_data:
                                        import pandas as pd
                                        df = pd.DataFrame(df_data)
                                        st.dataframe(df, use_container_width=True)
                                    else:
                                        st.info("No gene statistics found for the specified parameters")
                                elif parsed_output == [] or (isinstance(parsed_output, list) and len(parsed_output) == 0):
                                    st.info("No gene statistics found - these genes may not be significantly expressed in any contrasts")
                                else:
                                    st.info("No gene statistics available")

                            elif tool_name == 'summarise_contrast' and parsed_output:
                                # Display total DEGs as text and top genes as table
                                if isinstance(parsed_output, dict):
                                    total_degs = parsed_output.get('total_DEGs', 0)
                                    st.markdown(f"**Total DEGs:** {total_degs}")

                                    top_genes = parsed_output.get('top_genes', [])
                                    if top_genes:
                                        st.markdown("**Top Genes by Absolute LogFC:**")
                                        df_data = []
                                        for gene_info in top_genes:
                                            if isinstance(gene_info, dict) and 'gene' in gene_info and 'logFC' in gene_info:
                                                df_data.append({
                                                    'Gene': gene_info['gene'],
                                                    'LogFC': round(gene_info['logFC'], 4)
                                                })
                                        if df_data:
                                            import pandas as pd
                                            df = pd.DataFrame(df_data)
                                            st.dataframe(df, use_container_width=True)
                                else:
                                    st.text_area("Summary:", value=str(parsed_output), height=100, disabled=True, key=f"summary_output_{call.get('call_id', hash(str(parsed_output)))}")

                            elif tool_name == 'calculate_expression_variability' and parsed_output:
                                # Display as table with genes and variability statistics
                                if isinstance(parsed_output, dict) and 'variability_stats' in parsed_output:
                                    variability_stats = parsed_output['variability_stats']
                                    if variability_stats:
                                        df_data = []
                                        for stat in variability_stats:
                                            if isinstance(stat, dict) and 'gene' in stat:
                                                df_data.append({
                                                    'Gene': stat.get('gene', ''),
                                                    'Std_Dev': stat.get('std_dev', 'N/A'),
                                                    'Mean_LogFC': stat.get('mean_logFC', 'N/A'),
                                                    'Median_LogFC': stat.get('median_logFC', 'N/A'),
                                                    'Min_LogFC': stat.get('min_logFC', 'N/A'),
                                                    'Max_LogFC': stat.get('max_logFC', 'N/A'),
                                                    'Contrast_Count': stat.get('contrast_count', 0)
                                                })
                                        if df_data:
                                            import pandas as pd
                                            df = pd.DataFrame(df_data)
                                            st.dataframe(df, use_container_width=True)

                                        # Show summary if available
                                        summary = parsed_output.get('summary', {})
                                        if summary:
                                            genes_with_data = summary.get('genes_with_data', 0)
                                            total_requested = summary.get('total_genes_requested', 0)
                                    else:
                                        st.info("No variability statistics available")
                                else:
                                    st.json(parsed_output)

                            elif tool_name == 'calculate_gene_correlation' and parsed_output:
                                # Display correlation matrix as table
                                if isinstance(parsed_output, dict) and 'correlation_matrix' in parsed_output:
                                    correlation_matrix = parsed_output['correlation_matrix']
                                    genes_analyzed = parsed_output.get('genes_analyzed', [])
                                    sample_size = parsed_output.get('sample_size', 0)

                                    if correlation_matrix and genes_analyzed:
                                        # Convert correlation matrix to DataFrame
                                        import pandas as pd
                                        df_data = []
                                        for gene1 in genes_analyzed:
                                            row_data = {'Gene': gene1}
                                            for gene2 in genes_analyzed:
                                                corr_val = correlation_matrix.get(gene1, {}).get(gene2)
                                                row_data[gene2] = corr_val if corr_val is not None else 'N/A'
                                            df_data.append(row_data)

                                        if df_data:
                                            df = pd.DataFrame(df_data)
                                            st.markdown("**Gene Correlation Matrix (Spearman):**")
                                            st.dataframe(df, use_container_width=True)
                                    else:
                                        st.info("No correlation data available")
                                else:
                                    st.json(parsed_output)

                            else:
                                # Default display for other tools or unparseable output
                                if parsed_output:
                                    st.json(parsed_output)
                                else:
                                    st.code(clean_output)



                        except Exception as e:
                            # Parsing error
                                st.warning(f"Could not parse output: {e}")

                                # Try JSON parsing as fallback
                                try:
                                    import json as json_module
                                    parsed_json = json_module.loads(clean_output.replace("'", '"'))
                                    st.json(parsed_json)
                                except:
                                    # Display as formatted text
                                    if clean_output.strip().startswith(('{', '[')):
                                        st.code(clean_output, language='json')
                                    else:
                                        st.text_area("Raw Output:", value=clean_output, height=100, disabled=True, key=f"raw_output_{call.get('call_id', hash(clean_output))}")
                    elif not output_snippet or not output_snippet.strip():
                        st.info("Tool completed successfully but returned no output")
                    else:
                        st.write("*No output*")
                else:
                    st.error(f"**Error:** {call.get('error', 'Unknown error')}")

            # Metadata (full width below columns)
            if call.get('analysis_id'):
                st.caption(f"Analysis ID: {call['analysis_id']}")


def _display_raw_log_file():
    """Display the raw log file contents."""
    st.markdown("### Raw Log File Contents")
    st.markdown("*Complete JSON log file with all tool call data*")

    raw_contents = read_log_file_contents()

    if raw_contents:
        # Add download button for raw log
        st.download_button(
            label="Download Raw Log File",
            data=raw_contents,
            file_name=f"ai_tool_calls_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # Display in code block
        st.code(raw_contents, language='json')
    else:
        st.error("Could not read log file contents.")
