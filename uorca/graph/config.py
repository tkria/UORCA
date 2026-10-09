"""Configuration models for UORCA workflow."""

from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class GEOConfig(BaseModel):
    """Configuration for GEO dataset mode."""

    accession: str = Field(..., description="GEO series accession (e.g., GSE12345)")

    @field_validator("accession")
    @classmethod
    def validate_accession(cls, v: str) -> str:
        if not v.upper().startswith("GSE"):
            raise ValueError("GEO accession must start with 'GSE'")
        return v.upper()


class UserDataConfig(BaseModel):
    """Configuration for user-provided data mode."""

    fastq_dir: Path = Field(
        ...,
        description="Directory containing paired-end FASTQ files",
    )
    metadata_path: Path = Field(..., description="Path to metadata CSV file")
    organism: Optional[str] = Field(
        None,
        description="Organism name (e.g., 'Homo sapiens'). If not provided, LLM will infer.",
    )
    description: Optional[str] = Field(None, description="Dataset description for context")

    @field_validator("fastq_dir", "metadata_path", mode="before")
    @classmethod
    def convert_to_path(cls, v: Union[str, Path]) -> Path:
        return Path(v) if isinstance(v, str) else v


class WorkflowConfig(BaseModel):
    """Main workflow configuration."""

    mode: Literal["geo", "user"] = Field(..., description="Data source mode")

    # Mode-specific configs
    geo: Optional[GEOConfig] = None
    user_data: Optional[UserDataConfig] = None

    # Common settings
    output_dir: Path = Field(..., description="Directory for analysis outputs")
    resource_dir: Path = Field(..., description="Directory containing Kallisto indices")

    research_question: Optional[str] = None
    cleanup: bool = True
    max_retries: int = 5

    @field_validator("output_dir", "resource_dir", mode="before")
    @classmethod
    def convert_paths(cls, v: Union[str, Path]) -> Path:
        return Path(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def validate_mode_config(self) -> "WorkflowConfig":
        """Validate that the correct config is provided for the mode."""
        if self.mode == "geo" and not self.geo:
            raise ValueError("GEO config required when mode='geo'")
        if self.mode == "user" and not self.user_data:
            raise ValueError("user_data config required when mode='user'")
        return self


def load_config(config_path: Union[str, Path]) -> WorkflowConfig:
    """Load and validate workflow configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    return WorkflowConfig(**raw_config)
