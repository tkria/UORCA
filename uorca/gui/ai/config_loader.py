#!/usr/bin/env python3
"""
Configuration Loader for UORCA AI Assistant
===========================================

This module handles loading and managing configuration settings for the AI assistant,
including model parameters, thresholds, and other settings.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)




@dataclass
class AIAgentConfig:
    """Configuration for the main AI agent."""
    model: str = ""  # Empty = use get_model() default
    temperature: float = 1.0  # Temperature set to 1 for deterministic responses
    request_limit: int = 50  # Allow sufficient tool calls for complex analyses
    analysis_timeout: int = 300  # 5 minutes timeout to prevent indefinite hanging


@dataclass
class ContrastRelevanceWithSelectionConfig:
    """Configuration for contrast relevance with intelligent selection."""
    model: str = ""  # Empty = use get_model() default
    temperature: float = 0.1
    repeats: int = 3
    batch_size: int = 100
    parallel_jobs: int = 4


@dataclass
class UISettings:
    """UI-related settings."""
    default_lfc_threshold: float = 1.0
    default_p_threshold: float = 0.05
    max_genes_display: int = 50


@dataclass
class MCPServerConfig:
    """MCP server configuration."""
    startup_timeout: int = 180  # Keep MCP startup at 3 minutes
    max_retries: int = 3


class AIAssistantConfigLoader:
    """Configuration loader for AI Assistant settings."""

    # uorca/gui/ai/config_loader.py -> uorca/config/ai_assistant_config.json
    DEFAULT_CONFIG_PATH = (
        Path(__file__).parent.parent.parent / "config" / "ai_assistant_config.json"
    )

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config_data: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self._config_data = json.load(f)
                logger.info(f"Loaded AI assistant configuration from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found at {self.config_path}, using defaults")
                self._config_data = {}
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_path}: {e}")
            logger.info("Using default configuration values")
            self._config_data = {}

    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert dataclass objects back to dict format
            config_dict = {
                "ai_agent": {
                    "model": self.ai_agent.model,
                    "temperature": self.ai_agent.temperature,
                    "request_limit": self.ai_agent.request_limit,
                    "analysis_timeout": self.ai_agent.analysis_timeout
                },

                "contrast_relevance_with_selection": {
                    "model": self.contrast_relevance_with_selection.model,
                    "temperature": self.contrast_relevance_with_selection.temperature,
                    "repeats": self.contrast_relevance_with_selection.repeats,
                    "batch_size": self.contrast_relevance_with_selection.batch_size,
                    "parallel_jobs": self.contrast_relevance_with_selection.parallel_jobs
                },
                "ui_settings": {
                    "default_lfc_threshold": self.ui_settings.default_lfc_threshold,
                    "default_p_threshold": self.ui_settings.default_p_threshold,
                    "max_genes_display": self.ui_settings.max_genes_display
                },
                "mcp_server": {
                    "startup_timeout": self.mcp_server.startup_timeout,
                    "max_retries": self.mcp_server.max_retries
                }
            }

            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)

            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {self.config_path}: {e}")

    @property
    def ai_agent(self) -> AIAgentConfig:
        """Get AI agent configuration."""
        config_data = self._config_data.get("ai_agent", {})
        return AIAgentConfig(
            model=config_data.get("model", ""),  # Empty string = use get_model() default
            temperature=config_data.get("temperature", 1.0),
            request_limit=config_data.get("request_limit", 50),
            analysis_timeout=config_data.get("analysis_timeout", 300)
        )

    @property
    def contrast_relevance_with_selection(self) -> ContrastRelevanceWithSelectionConfig:
        """Get contrast relevance with selection configuration."""
        config_data = self._config_data.get("contrast_relevance_with_selection", {})
        return ContrastRelevanceWithSelectionConfig(
            model=config_data.get("model", ""),  # Empty string = use get_model() default
            temperature=config_data.get("temperature", 0.1),
            repeats=config_data.get("repeats", 3),
            batch_size=config_data.get("batch_size", 100),
            parallel_jobs=config_data.get("parallel_jobs", 4)
        )

    @property
    def ui_settings(self) -> UISettings:
        """Get UI settings."""
        config_data = self._config_data.get("ui_settings", {})
        return UISettings(
            default_lfc_threshold=config_data.get("default_lfc_threshold", 1.0),
            default_p_threshold=config_data.get("default_p_threshold", 0.05),
            max_genes_display=config_data.get("max_genes_display", 50)
        )

    @property
    def mcp_server(self) -> MCPServerConfig:
        """Get MCP server configuration."""
        config_data = self._config_data.get("mcp_server", {})
        return MCPServerConfig(
            startup_timeout=config_data.get("startup_timeout", 180),
            max_retries=config_data.get("max_retries", 3)
        )

    def update_ai_agent(self, **kwargs) -> None:
        """Update AI agent configuration."""
        current = self.ai_agent
        for key, value in kwargs.items():
            if hasattr(current, key):
                setattr(current, key, value)

        if "ai_agent" not in self._config_data:
            self._config_data["ai_agent"] = {}

        for key, value in kwargs.items():
            if hasattr(current, key):
                self._config_data["ai_agent"][key] = value

    def get_config_summary(self) -> str:
        """Get a human-readable summary of the current configuration."""
        summary = []
        summary.append("AI Assistant Configuration Summary:")
        summary.append("=" * 40)
        summary.append(f"AI Agent Model: {self.ai_agent.model}")
        summary.append(f"AI Agent Temperature: {self.ai_agent.temperature}")
        summary.append(f"AI Agent Request Limit: {self.ai_agent.request_limit}")
        summary.append(f"AI Agent Analysis Timeout: {self.ai_agent.analysis_timeout}s")
        summary.append("")

        summary.append(f"Default LFC Threshold: {self.ui_settings.default_lfc_threshold}")
        summary.append(f"Default P-value Threshold: {self.ui_settings.default_p_threshold}")

        return "\n".join(summary)


# Global configuration instance
_config_instance: Optional[AIAssistantConfigLoader] = None


def get_ai_config() -> AIAssistantConfigLoader:
    """Get the global AI assistant configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AIAssistantConfigLoader()
    return _config_instance


def reload_config() -> AIAssistantConfigLoader:
    """Reload the configuration from file."""
    global _config_instance
    _config_instance = AIAssistantConfigLoader()
    return _config_instance


# Convenience functions for common use cases
def get_ai_agent_config() -> AIAgentConfig:
    """Get AI agent configuration."""
    return get_ai_config().ai_agent


def get_contrast_relevance_with_selection_config() -> ContrastRelevanceWithSelectionConfig:
    """Get contrast relevance with selection configuration."""
    return get_ai_config().contrast_relevance_with_selection


def get_ui_settings() -> UISettings:
    """Get UI settings."""
    return get_ai_config().ui_settings


def get_mcp_server_config() -> MCPServerConfig:
    """Get MCP server configuration."""
    return get_ai_config().mcp_server
