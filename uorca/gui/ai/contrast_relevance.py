from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Tuple
import asyncio
import json
import pandas as pd
import statistics
import sys
import os
import logging
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:
    # dotenv not available, continue without it
    def load_dotenv():
        pass

# Import validation functions
from uorca.gui.components.helpers import get_valid_contrasts_with_data

from uorca.gui.components.helpers import log_streamlit_agent
from .config_loader import get_contrast_relevance_with_selection_config
from pydantic_ai import Agent
from uorca.ai_provider import get_model


# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Add path to import from dataset_identification module
sys.path.insert(0, str(Path(__file__).parent.parent / "dataset_identification"))

def call_openai_json(prompt: str, schema: Dict[str, Any], name: str) -> dict:
    """Call LLM with JSON schema enforcement using the configured provider."""
    agent = Agent(get_model(), output_type=str, system_prompt="You must respond with valid JSON matching the requested schema. Output ONLY the JSON, no other text.")
    result = agent.run_sync(prompt)
    return json.loads(result.output)

# Define prompts directory for this module
PROMPT_DIR = Path(__file__).parent / "prompts"
logger = logging.getLogger(__name__)

def load_prompt(fname: str) -> str:
    """Load prompt from the reporting prompts directory."""
    path = PROMPT_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text().strip()

class ContrastAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_id: str
    contrast_id: str
    RelevanceScore: float
    Justification: str

class ContrastAssessments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessments: List[ContrastAssessment]

class ContrastCategory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str  # "primary", "control", "comparative", "supportive"
    justification: str

class SelectedContrast(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_id: str
    contrast_id: str
    RelevanceScore: float
    category: ContrastCategory
    selection_justification: str

class ContrastSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    research_query: str
    selection_strategy: str
    max_contrasts: int
    selected_contrasts: List[SelectedContrast]
    selection_summary: str

class ContrastAssessmentWithSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessments: List[ContrastAssessment]  # All contrasts with relevance scores
    selection: ContrastSelection  # Intelligent subset selection

CONTRAST_ASSESS_SCHEMA = ContrastAssessments.model_json_schema()
CONTRAST_ASSESS_WITH_SELECTION_SCHEMA = ContrastAssessmentWithSelection.model_json_schema()

@log_streamlit_agent
async def assess_contrast_subbatch(
    sub_df: pd.DataFrame,
    query: str,
    schema: Dict[str, Any],
    name: str,
    rep: int,
    batch_idx: int,
    total_batches: int,
    sem: asyncio.Semaphore
) -> List[ContrastAssessment]:
    async with sem:
        print(f"⏳ Contrast relevance rep {rep+1}, batch {batch_idx+1}/{total_batches}")
        sys_prompt = load_prompt("assess_and_select_contrasts.txt")
        prompt = (
            f"{sys_prompt}\n"
            f"Research query: \"{query}\"\n"
            f"Contrasts to assess (batch {batch_idx+1}):\n"
            f"{sub_df.to_json(orient='records')}"
        )
        # Log the contrast relevance assessment prompt
        logger.info(f"CONTRAST RELEVANCE PROMPT: Research query: '{query}' | Batch {batch_idx+1}/{total_batches} | Rep {rep+1} | Contrasts: {len(sub_df)} | Prompt length: {len(prompt)} chars")
        logger.debug(f"Full contrast relevance prompt:\n{prompt}")
        data = await asyncio.to_thread(call_openai_json, prompt, schema, name)
        return [ContrastAssessment.model_validate(a) for a in data['assessments']]

@log_streamlit_agent
async def repeated_contrast_relevance(
    df: pd.DataFrame,
    query: str,
    repeats: int,
    batch_size: int,
    openai_api_jobs: int
) -> pd.DataFrame:
    print(
        f"📊 Starting contrast relevance scoring: "
        f"{repeats} repetitions, batch size {batch_size}, "
        f"parallel API jobs: {openai_api_jobs}"
    )
    sem = asyncio.Semaphore(openai_api_jobs)
    tasks = []
    batches = [df.iloc[i:i+batch_size] for i in range(0, len(df), batch_size)]
    total_batches = len(batches)

    for rep in range(repeats):
        for idx, sub in enumerate(batches):
            tasks.append(
                assess_contrast_subbatch(
                    sub, query, CONTRAST_ASSESS_SCHEMA, "contrast_assessments",
                    rep, idx, total_batches, sem
                )
            )

    all_results = await asyncio.gather(*tasks)

    # Flatten and aggregate
    coll: Dict[tuple, Dict[str, Any]] = {}
    for result in all_results:
        for a in result:
            key = (a.analysis_id, a.contrast_id)
            entry = coll.setdefault(key, {'scores': [], 'justifications': []})
            entry['scores'].append(a.RelevanceScore)
            entry['justifications'].append(a.Justification)

    records: List[Dict[str, Any]] = []
    for (analysis_id, contrast_id), v in coll.items():
        rec = {
            'analysis_id': analysis_id,
            'contrast_id': contrast_id,
            'RelevanceScore': round(statistics.mean(v['scores']), 2)
        }
        for i, (score, just) in enumerate(zip(v['scores'], v['justifications']), 1):
            rec[f'Run{i}Score'] = score
            rec[f'Run{i}Justification'] = just
        records.append(rec)

    return pd.DataFrame(records)

@log_streamlit_agent
def run_contrast_relevance(
    ri,  # ResultsIntegrator instance
    query: str,
    repeats: int = 3,
    batch_size: int = 50,
    parallel_jobs: int = 4
) -> pd.DataFrame:
    """
    Run contrast relevance assessment using LLM calls.

    Parameters:
    -----------
    ri : ResultsIntegrator
        Instance of ResultsIntegrator with loaded data
    query : str
        Research question to assess contrast relevance against
    repeats : int
        Number of repetitions for each contrast assessment
    batch_size : int
        Number of contrasts to assess in each API call
    parallel_jobs : int
        Number of parallel API calls

    Returns:
    --------
    pd.DataFrame
        DataFrame with contrast relevance scores and justifications
    """
    # Use proper validation logic to get valid contrasts only
    valid_contrasts = get_valid_contrasts_with_data(ri)

    contrast_list = []
    for contrast in valid_contrasts:
        analysis_id = contrast['analysis_id']
        accession = contrast['accession']

        # Get dataset metadata
        dataset_meta = getattr(ri, "dataset_info", {}).get(analysis_id, {})
        title = dataset_meta.get("title", "")
        summary = dataset_meta.get("summary", "")
        design = dataset_meta.get("design", "")

        # Get organism info
        analysis_info = ri.analysis_info.get(analysis_id, {})
        organism = analysis_info.get('organism', 'Unknown')

        contrast_list.append({
            'analysis_id': analysis_id,
            'contrast_id': contrast['contrast_name'],  # Use consistent name
            'accession': accession,
            'organism': organism,
            'description': contrast['description'],
            'title': title,
            'summary': summary,
            'design': design
        })

    if not contrast_list:
        return pd.DataFrame()

    df_contrasts = pd.DataFrame(contrast_list)

    # Run the async assessment
    return asyncio.run(
        repeated_contrast_relevance(
            df_contrasts,
            query,
            repeats,
            batch_size,
            parallel_jobs
        )
    )

@log_streamlit_agent
async def assess_and_select_contrasts(
    sub_df: pd.DataFrame,
    query: str,
    schema: Dict[str, Any],
    name: str,
    rep: int,
    batch_idx: int,
    total_batches: int,
    sem: asyncio.Semaphore
) -> ContrastAssessmentWithSelection:
    async with sem:
        print(f"⏳ Contrast relevance + selection rep {rep+1}, batch {batch_idx+1}/{total_batches}")
        sys_prompt = load_prompt("assess_and_select_contrasts.txt")
        prompt = (
            f"{sys_prompt}\n"
            f"Research query: \"{query}\"\n"
            f"Contrasts to assess and select from (batch {batch_idx+1}):\n"
            f"{sub_df.to_json(orient='records')}"
        )
        # Log the contrast relevance with selection prompt
        logger.info(f"CONTRAST RELEVANCE WITH SELECTION PROMPT: Research query: '{query}' | Batch {batch_idx+1}/{total_batches} | Rep {rep+1} | Contrasts: {len(sub_df)} | Prompt length: {len(prompt)} chars")
        logger.debug(f"Full contrast relevance with selection prompt:\n{prompt}")
        data = await asyncio.to_thread(call_openai_json, prompt, schema, name)
        return ContrastAssessmentWithSelection.model_validate(data)

@log_streamlit_agent
async def repeated_contrast_relevance_with_selection(
    df: pd.DataFrame,
    query: str,
    repeats: int,
    batch_size: int,
    openai_api_jobs: int
) -> Tuple[pd.DataFrame, List[SelectedContrast]]:
    """
    Run contrast relevance assessment with intelligent selection.

    Returns:
        Tuple of (relevance_df, selected_contrasts_list)
    """
    print(
        f"📊 Starting contrast relevance + selection: "
        f"{repeats} repetitions, batch size {batch_size}, "
        f"parallel API jobs: {openai_api_jobs}"
    )
    sem = asyncio.Semaphore(openai_api_jobs)

    if len(df) <= batch_size:
        # Single batch - can do selection directly
        task = assess_and_select_contrasts(
            df, query, CONTRAST_ASSESS_WITH_SELECTION_SCHEMA,
            "contrast_assessment_with_selection", 0, 0, 1, sem
        )
        result = await task

        # Convert assessments to DataFrame
        relevance_records = []
        for assessment in result.assessments:
            relevance_records.append({
                'analysis_id': assessment.analysis_id,
                'contrast_id': assessment.contrast_id,
                'RelevanceScore': assessment.RelevanceScore,
                'Justification': assessment.Justification
            })

        relevance_df = pd.DataFrame(relevance_records)
        selected_contrasts = result.selection.selected_contrasts

        return relevance_df, selected_contrasts
    else:
        # Multiple batches - fall back to original approach for now
        # (Could be enhanced later to do selection across batches)
        relevance_df = await repeated_contrast_relevance(df, query, repeats, batch_size, openai_api_jobs)
        return relevance_df, []

@log_streamlit_agent
def run_contrast_relevance_with_selection(
    ri,  # ResultsIntegrator instance
    query: str,
    repeats: int = 1,  # Reduced since we're doing more complex processing
    batch_size: int = 100,  # Larger batch size for better selection
    parallel_jobs: int = 2
) -> Tuple[pd.DataFrame, List[SelectedContrast]]:
    """
    Run contrast relevance assessment with intelligent selection.

    Returns:
        Tuple of (relevance_df, selected_contrasts_list)
    """
    # Build contrast list (same as before)
    # Use proper validation logic to get valid contrasts only
    valid_contrasts = get_valid_contrasts_with_data(ri)

    contrast_list = []
    for contrast in valid_contrasts:
        analysis_id = contrast['analysis_id']
        accession = contrast['accession']

        # Get dataset metadata
        dataset_meta = getattr(ri, "dataset_info", {}).get(analysis_id, {})
        title = dataset_meta.get("title", "")
        summary = dataset_meta.get("summary", "")
        design = dataset_meta.get("design", "")

        # Get organism info
        analysis_info = ri.analysis_info.get(analysis_id, {})
        organism = analysis_info.get('organism', 'Unknown')

        contrast_list.append({
            'analysis_id': analysis_id,
            'contrast_id': contrast['contrast_name'],  # Use consistent name
            'accession': accession,
            'organism': organism,
            'description': contrast['description'],
            'title': title,
            'summary': summary,
            'design': design
        })

    if not contrast_list:
        return pd.DataFrame(), []

    df_contrasts = pd.DataFrame(contrast_list)

    # Run the async assessment with selection
    return asyncio.run(
        repeated_contrast_relevance_with_selection(
            df_contrasts,
            query,
            repeats,
            batch_size,
            parallel_jobs
        )
    )
