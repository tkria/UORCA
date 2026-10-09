"""Graph assembly and diagram export."""

from pydantic_graph import Graph

from .nodes.contrasts import ContrastDesignNode
from .nodes.edger import PrepareEdgeRNode, RunDEAnalysisNode
from .nodes.evaluate import EvaluateNode
from .nodes.geo_extract import GEOExtractNode
from .nodes.kallisto import KallistoQuantifyNode
from .nodes.metadata import MetadataAnalysisNode
from .nodes.reflect import ReflectNode
from .nodes.user_extract import UserDataExtractNode

# Assemble the workflow graph
workflow_graph = Graph(
    nodes=[
        GEOExtractNode,
        UserDataExtractNode,  # Alternative entry point for user-provided data
        MetadataAnalysisNode,
        ContrastDesignNode,
        KallistoQuantifyNode,
        PrepareEdgeRNode,
        RunDEAnalysisNode,
        EvaluateNode,
        ReflectNode,
    ]
)


def export_mermaid_diagram() -> str:
    """Export Mermaid diagram for documentation."""
    return workflow_graph.mermaid_code()


def save_diagram(output_path: str = "docs/workflow_diagram.md") -> None:
    """Save Mermaid diagram to file."""
    diagram = export_mermaid_diagram()
    with open(output_path, "w") as f:
        f.write("# UORCA Workflow Graph\n\n")
        f.write("```mermaid\n")
        f.write(diagram)
        f.write("\n```\n")
