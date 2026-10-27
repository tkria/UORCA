"""
AI Gene Analysis Schema
======================

Pydantic models for structured AI gene analysis output in UORCA Explorer.
These schemas ensure that AI agent responses conform to expected formats
and provide type safety for downstream processing.
"""

from typing import List
from pydantic import BaseModel, Field


class GeneFilters(BaseModel):
    """
    Filters used for gene selection analysis.

    Attributes:
        lfc_thresh: Log2 fold change threshold (must be >= 0)
        p_thresh: Adjusted p-value (BH) threshold (must be > 0 and <= 1)
    """
    lfc_thresh: float = Field(..., ge=0, description="Log2 fold change threshold")
    p_thresh: float = Field(..., gt=0, le=1, description="Adjusted p-value (BH) threshold")


class GeneAnalysisOutput(BaseModel):
    """
    Structured output from AI gene analysis.

    Attributes:
        genes: List of gene symbols (1-50 genes)
        filters: Filter parameters used in analysis
        interpretation: AI-generated biological interpretation
    """
    genes: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of gene symbols identified by analysis"
    )
    filters: GeneFilters = Field(..., description="Filter parameters used")
    interpretation: str = Field(
        ...,
        min_length=10,
        description="Biological interpretation of the results"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "genes": ["MYCN", "ALK", "PHOX2B"],
                "filters": {
                    "lfc_thresh": 1.0,
                    "p_thresh": 0.05
                },
                "interpretation": "These genes represent key oncogenic drivers in neuroblastoma development."
            }
        }
