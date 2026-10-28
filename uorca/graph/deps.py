"""External services injected into graph nodes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


@dataclass
class WorkflowDeps:
    """Injected dependencies for the RNA-seq workflow graph."""

    # File system paths
    output_dir: Path
    resource_dir: Path  # Kallisto indices, tx2gene files

    # NCBI credentials
    entrez_email: str
    entrez_api_key: Optional[str] = None

    # API clients
    openai_client: Any = None  # Deprecated — agents use ai_provider.get_model()
    http_client: Optional[httpx.AsyncClient] = None

    # Configuration
    cleanup_after_success: bool = True
    docker_image: str = "kevingchen/uorca:0.1.0"
    max_retries: int = 5

    # Set per contrast-design call so the agent's output validator can check
    # expressions against the R level names (raw level -> R name).
    group_level_names: Optional[Dict[str, str]] = None
    # Last issues the contrast output validator found, so the node can report
    # them if the model runs out of retries.
    contrast_issues: Optional[List[str]] = None
