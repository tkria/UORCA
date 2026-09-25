#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Agent Factory for UORCA Analysis
-----------------------------------
Factory functions for creating AI agents that connect to the UORCA MCP server
for analyzing RNA-seq results using structured tools.
"""

import os
import logging
from typing import Optional
from pathlib import Path
import streamlit as st

from uorca.gui.components.helpers import log_streamlit_function
from uorca.gui.components.helpers.ai_agent_tool_logger import (
    start_ai_analysis_session,
    clear_ai_tool_logs,
    get_ai_tool_logger
)
from .gene_schema import GeneAnalysisOutput
from .config_loader import get_ai_agent_config, get_mcp_server_config
from uorca.ai_provider import get_model

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

logger = logging.getLogger(__name__)

def load_system_prompt() -> str:
    """Load the system prompt from file."""
    prompt_file = Path(__file__).parent / "prompts" / "ai_agent_analysis.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"System prompt file not found: {prompt_file}")
    return prompt_file.read_text().strip()

@st.cache_resource
def create_uorca_agent(selected_contrasts_key: str = "") -> Optional[Agent]:
    """Create an agent connected to the UORCA MCP server.

    Args:
        selected_contrasts_key: String representation of selected contrasts for cache invalidation
    """

    # Load configuration
    ai_config = get_ai_agent_config()
    mcp_config = get_mcp_server_config()

    # Determine model: fall back to get_model() when config has empty value
    model = ai_config.model
    if not model:
        model = get_model()

    try:
        server_script = Path(__file__).parent.parent / "mcp_server" / "server_core.py"

        # Create fresh environment with current selected contrasts
        server_env = os.environ.copy()
        logger.info(f"Creating MCP server with selected contrasts key: {selected_contrasts_key[:100]}...")

        server = MCPServerStdio(
            command="uv",
            args=["run", str(server_script), "server"],
            env=server_env,
            timeout = mcp_config.startup_timeout
        )

        agent = Agent(
            model=model,
            model_settings={"temperature": ai_config.temperature},
            mcp_servers=[server],
            system_prompt=load_system_prompt(),
            output_type=GeneAnalysisOutput,
        )

        # Initialize file-based tool logging system
        tool_logger = get_ai_tool_logger()
        logger.info("AI agent created with file-based tool logging initialized")

        return agent
    except Exception as e:
        logger.error(f"Failed to create UORCA MCP agent: {e}")
        raise RuntimeError(f"Agent creation failed: {e}")



# Convenience function for Streamlit apps
def get_uorca_agent(selected_contrasts_key: str = "") -> Optional[Agent]:
    """Return a cached UORCA agent instance for Streamlit.

    Args:
        selected_contrasts_key: String representation of selected contrasts for cache invalidation

    Returns:
        Agent instance or None if API key is not available
    """
    try:
        agent = create_uorca_agent(selected_contrasts_key)
        if agent is None:
            logger.info("UORCA agent not created - OpenAI API key not available")
            return None
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize UORCA MCP agent: {e}")
        st.error(f"Failed to initialize UORCA MCP agent: {e}")
        return None
