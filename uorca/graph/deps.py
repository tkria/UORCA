"""External services injected into graph nodes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
