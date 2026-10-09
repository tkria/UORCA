"""Entry point for running the workflow graph."""

import asyncio
import os
from pathlib import Path
from typing import Optional, Union

import httpx

from .config import WorkflowConfig, load_config
from .deps import WorkflowDeps
from .graph import workflow_graph
from .nodes.geo_extract import GEOExtractNode
from .nodes.user_extract import UserDataExtractNode
from .state import WorkflowState


async def run_workflow(
    # Required
    output_dir: Path,
    resource_dir: Path,
    geo_accession: str,  # GEO accession (e.g., 'GSE12345')
    # Optional configuration
    entrez_email: Optional[str] = None,
    entrez_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    research_question: Optional[str] = None,
    cleanup: bool = True,
    max_retries: int = 5,
) -> WorkflowState:
    """
    Run the UORCA RNA-seq analysis workflow on a GEO dataset.

    Args:
        output_dir: Directory for analysis outputs
        resource_dir: Directory containing Kallisto indices
        geo_accession: GEO accession (e.g., 'GSE12345')

        entrez_email: Email for NCBI API (required, uses env var if not provided)
        entrez_api_key: Optional NCBI API key for faster queries
        openai_api_key: OpenAI API key (uses env var if not provided)
        research_question: Optional context for AI-driven analysis
        cleanup: Whether to cleanup intermediate files on success
        max_retries: Maximum reflection/retry attempts

    Returns:
        WorkflowState with analysis results

    Example:
        state = await run_workflow(
            output_dir=Path("./results"),
            resource_dir=Path("./kallisto_indices"),
            geo_accession="GSE12345"
        )
    """
    # Validate NCBI credentials
    entrez_email = entrez_email or os.environ.get("ENTREZ_EMAIL")
    if not entrez_email:
        raise ValueError("entrez_email required (or set ENTREZ_EMAIL env var)")

    # Initialize state
    state = WorkflowState(
        dataset_id=geo_accession,
        research_question=research_question,
    )

    # Initialize dependencies
    async with httpx.AsyncClient(timeout=300) as http_client:
        openai_client = None  # Agents use get_model() from ai_provider

        deps = WorkflowDeps(
            openai_client=openai_client,
            http_client=http_client,
            output_dir=Path(output_dir),
            resource_dir=Path(resource_dir),
            entrez_email=entrez_email,
            entrez_api_key=entrez_api_key or os.environ.get("ENTREZ_API_KEY"),
            cleanup_after_success=cleanup,
            max_retries=max_retries,
        )

        # Run the graph starting with GEOExtractNode
        await workflow_graph.run(
            GEOExtractNode(),
            state=state,
            deps=deps,
        )

        return state


def run_workflow_sync(**kwargs) -> WorkflowState:
    """Synchronous wrapper for run_workflow."""
    return asyncio.run(run_workflow(**kwargs))


async def run_workflow_from_config(
    config_path: Union[str, Path],
    entrez_email: Optional[str] = None,
    entrez_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> WorkflowState:
    """
    Run the UORCA workflow from a configuration file.

    Supports both GEO datasets and user-provided data.

    Args:
        config_path: Path to YAML configuration file
        entrez_email: NCBI email (required for GEO mode, uses env var if not provided)
        entrez_api_key: Optional NCBI API key
        openai_api_key: OpenAI API key (uses env var if not provided)

    Returns:
        WorkflowState with analysis results

    Example:
        state = await run_workflow_from_config("uorca_config.yaml")
    """
    config = load_config(config_path)

    # Validate NCBI credentials for GEO mode
    if config.mode == "geo":
        entrez_email = entrez_email or os.environ.get("ENTREZ_EMAIL")
        if not entrez_email:
            raise ValueError("entrez_email required for GEO mode (or set ENTREZ_EMAIL env var)")

    # Determine dataset ID
    if config.mode == "geo":
        assert config.geo is not None  # Validated by config
        dataset_id = config.geo.accession
    else:
        assert config.user_data is not None  # Validated by config
        dataset_id = f"user_{config.user_data.fastq_dir.name}"

    # Initialize state
    state = WorkflowState(
        dataset_id=dataset_id,
        research_question=config.research_question,
        data_source=config.mode,
    )

    # Store user data config for retry support
    if config.mode == "user" and config.user_data is not None:
        state.user_fastq_dir = str(config.user_data.fastq_dir)
        state.user_metadata_path = str(config.user_data.metadata_path)
        state.user_organism = config.user_data.organism
        state.user_description = config.user_data.description

    # Create output directory structure
    run_output_dir = config.output_dir / dataset_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dependencies
    async with httpx.AsyncClient(timeout=300) as http_client:
        openai_client = None  # Agents use get_model() from ai_provider

        deps = WorkflowDeps(
            openai_client=openai_client,
            http_client=http_client,
            output_dir=run_output_dir,
            resource_dir=config.resource_dir,
            entrez_email=entrez_email or "",
            entrez_api_key=entrez_api_key or os.environ.get("ENTREZ_API_KEY"),
            cleanup_after_success=config.cleanup,
            max_retries=config.max_retries,
        )

        # Select entry node based on mode
        if config.mode == "geo":
            start_node = GEOExtractNode()
        else:
            assert config.user_data is not None
            start_node = UserDataExtractNode(
                fastq_dir=str(config.user_data.fastq_dir),
                metadata_path=str(config.user_data.metadata_path),
                organism=config.user_data.organism,
                description=config.user_data.description,
            )

        # Run the graph
        await workflow_graph.run(
            start_node,
            state=state,
            deps=deps,
        )

        return state


def run_workflow_from_config_sync(config_path: Union[str, Path], **kwargs) -> WorkflowState:
    """Synchronous wrapper for run_workflow_from_config."""
    return asyncio.run(run_workflow_from_config(config_path, **kwargs))


def main():
    """CLI entry point for the graph-based workflow."""
    import argparse
    import json
    import logging
    import sys

    parser = argparse.ArgumentParser(
        description="UORCA Graph-based RNA-seq Analysis Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GEO dataset mode:
  python -m uorca.graph.runner --accession GSE12345 --output_dir ./results --resource_dir ./kallisto_indices

  # User-provided data mode (via config file):
  python -m uorca.graph.runner --config my_config.yaml
        """
    )

    # Create mutually exclusive group for mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--accession",
        help="GEO accession ID (e.g., GSE12345)"
    )
    mode_group.add_argument(
        "--config",
        help="Path to YAML config file for user-provided data"
    )
    parser.add_argument(
        "--output_dir",
        help="Directory for analysis outputs (required for --accession mode)"
    )
    parser.add_argument(
        "--resource_dir",
        help="Directory containing Kallisto indices and tx2gene files (required for --accession mode)"
    )
    parser.add_argument(
        "--research_question",
        default=None,
        help="Optional research question for AI-guided analysis"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Cleanup intermediate files (FASTQ/SRA) after successful analysis"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum retry attempts on failure (default: 5)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    from uorca.shared.workflow_logging import setup_logging

    # Handle config mode (user-provided data)
    if args.config:
        # Load config to get output directory for logging
        config = load_config(args.config)
        run_output_dir = config.output_dir / f"user_{config.user_data.fastq_dir.name}" if config.user_data else config.output_dir
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_dir = run_output_dir / "logs"
        log_level = logging.DEBUG if args.verbose else logging.INFO
        log_file = setup_logging(log_dir=log_dir, level=log_level, run_id="user_data")

        logger = logging.getLogger(__name__)

        logger.info("=" * 60)
        logger.info("UORCA Graph-based Workflow (User Data Mode)")
        logger.info("=" * 60)
        if log_file and log_file.exists():
            logger.info(f"Log file: {log_file}")
        logger.info(f"Config file: {args.config}")
        logger.info(f"Output directory: {run_output_dir}")
        logger.info(f"Resource directory: {config.resource_dir}")
        logger.info(f"Cleanup: {config.cleanup}")
        logger.info(f"Max retries: {config.max_retries}")
        if config.research_question:
            logger.info(f"Research question: {config.research_question}")
        logger.info("=" * 60)

        try:
            state = run_workflow_from_config_sync(args.config)

            # Report results
            logger.info("=" * 60)
            if state.analysis_success:
                logger.info("Analysis completed successfully!")
                logger.info(f"Results: {state.deg_results_path}")
            else:
                logger.error("Analysis failed!")
                logger.error(f"Diagnostics: {state.analysis_diagnostics}")
                sys.exit(1)

            # Output final state summary as JSON
            summary = {
                "success": state.analysis_success,
                "dataset_id": state.dataset_id,
                "organism": state.organism,
                "results_path": state.deg_results_path,
                "checkpoints": {k: v.value for k, v in state.checkpoints.items()},
                "diagnostics": state.analysis_diagnostics,
            }
            logger.info(f"Summary: {json.dumps(summary, indent=2)}")

        except Exception as e:
            logger.exception(f"Workflow failed with error: {e}")
            sys.exit(1)

    # Handle accession mode (GEO datasets)
    else:
        # Validate required args for accession mode
        if not args.output_dir or not args.resource_dir:
            parser.error("--output_dir and --resource_dir are required when using --accession")

        # Create accession-specific output directory (matches old master.py behavior)
        run_output_dir = Path(args.output_dir) / args.accession
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Setup file-based logging in accession output directory
        log_dir = run_output_dir / "logs"
        log_level = logging.DEBUG if args.verbose else logging.INFO
        log_file = setup_logging(log_dir=log_dir, level=log_level, run_id=args.accession)

        logger = logging.getLogger(__name__)

        logger.info("=" * 60)
        logger.info("UORCA Graph-based Workflow")
        logger.info("=" * 60)
        if log_file and log_file.exists():
            logger.info(f"Log file: {log_file}")
        logger.info(f"Accession: {args.accession}")
        logger.info(f"Output directory: {run_output_dir}")
        logger.info(f"Resource directory: {args.resource_dir}")
        logger.info(f"Cleanup: {args.cleanup}")
        logger.info(f"Max retries: {args.max_retries}")
        if args.research_question:
            logger.info(f"Research question: {args.research_question}")
        logger.info("=" * 60)

        try:
            state = run_workflow_sync(
                output_dir=run_output_dir,
                resource_dir=Path(args.resource_dir),
                geo_accession=args.accession,
                research_question=args.research_question,
                cleanup=args.cleanup,
                max_retries=args.max_retries,
            )

            # Report results
            logger.info("=" * 60)
            if state.analysis_success:
                logger.info("Analysis completed successfully!")
                logger.info(f"Results: {state.deg_results_path}")
            else:
                logger.error("Analysis failed!")
                logger.error(f"Diagnostics: {state.analysis_diagnostics}")

                # Write analysis_info.json even on failure (for the results explorer)
                try:
                    from .nodes.evaluate import write_analysis_info
                    from .deps import WorkflowDeps
                    deps = WorkflowDeps(
                        openai_client=None,
                        http_client=None,
                        output_dir=run_output_dir,
                        resource_dir=Path(args.resource_dir),
                        entrez_email="",
                    )
                    state.analysis_success = False
                    write_analysis_info(state, deps)
                except Exception:
                    pass

                # Cleanup FASTQ/SRA on failure too
                if args.cleanup:
                    from .nodes.evaluate import cleanup_fastq_and_sra
                    logger.info("Cleaning up FASTQ/SRA files from failed analysis...")
                    cleanup_fastq_and_sra(run_output_dir)

                sys.exit(1)

            # Output final state summary as JSON for programmatic consumption
            summary = {
                "success": state.analysis_success,
                "dataset_id": state.dataset_id,
                "organism": state.organism,
                "results_path": state.deg_results_path,
                "checkpoints": {k: v.value for k, v in state.checkpoints.items()},
                "diagnostics": state.analysis_diagnostics,
            }
            logger.info(f"Summary: {json.dumps(summary, indent=2)}")

        except Exception as e:
            logger.exception(f"Workflow failed with error: {e}")

            # Write analysis_info.json on exception
            try:
                from .nodes.evaluate import write_analysis_info
                from .deps import WorkflowDeps
                deps = WorkflowDeps(
                    openai_client=None,
                    http_client=None,
                    output_dir=run_output_dir,
                    resource_dir=Path(args.resource_dir),
                    entrez_email="",
                )
                state.analysis_success = False
                state.analysis_diagnostics = str(e)
                write_analysis_info(state, deps)
            except Exception:
                pass

            # Cleanup FASTQ/SRA on exception too
            if args.cleanup:
                try:
                    from .nodes.evaluate import cleanup_fastq_and_sra
                    logger.info("Cleaning up FASTQ/SRA files after error...")
                    cleanup_fastq_and_sra(run_output_dir)
                except Exception:
                    pass

            sys.exit(1)


if __name__ == "__main__":
    main()
