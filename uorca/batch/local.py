"""
Local Batch Processor

This module provides local parallel processing capabilities for UORCA datasets
using Docker containers. It implements storage-aware queueing and proper job
cancellation while respecting resource constraints.
"""


import os
import psutil
import signal
import subprocess
import time
import threading
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from pathlib import Path
from typing import Dict, List, Any
import multiprocessing as mp
from dotenv import load_dotenv, find_dotenv

from .base import BatchProcessor


def run_single_dataset_local(accession: str, output_dir: str, resource_dir: str,
                           cleanup: bool, status_dir: str, docker_image: str = "kevingchen/uorca:0.1.0",
                           container_tmpfs_gb: int = 20) -> Dict[str, Any]:
    """
    Run a single dataset analysis locally using Docker.

    This function is designed to be run in a separate process. It creates a temporary
    directory on the host filesystem ({output_dir}/{accession}/tmp/) and bind-mounts it
    to the container to bypass Docker's virtual disk limits. This allows processing of
    large datasets regardless of Docker Desktop's disk allocation.

    Args:
        accession: Dataset accession ID
        output_dir: Output directory on host
        resource_dir: Resource directory for Kallisto indices on host
        cleanup: Whether to cleanup intermediate files (FASTQ/SRA after analysis)
        status_dir: Directory for status tracking on host
        docker_image: Docker image to use
        container_tmpfs_gb: Size in GB for in-memory tmpfs (for small temp files)

    Returns:
        Dictionary with job results including container_id

    Note:
        The temp directory ({output_dir}/{accession}/tmp/) is automatically created
        and cleaned up (even on failure). Temporary files from fasterq-dump, pigz, etc.
        write to this bind-mounted directory on the host, not to Docker's virtual disk.
    """
    import subprocess
    import json
    import shutil
    from datetime import datetime
    from pathlib import Path
    import re

    process_id = os.getpid()
    start_time = datetime.now()

    # Setup paths (before try block so temp_dir is available in finally)
    output_path = Path(output_dir).absolute()
    resource_path = Path(resource_dir).absolute()

    # Find project root (where this batch processor is located)
    current_file = Path(__file__).absolute()
    project_root = current_file.parent.parent.parent  # Go up from uorca/batch/local.py to project root

    # Create directories if they don't exist
    output_path.mkdir(parents=True, exist_ok=True)
    resource_path.mkdir(parents=True, exist_ok=True)

    # Create temp directory for large intermediate files (bind-mounted to bypass Docker disk limits)
    temp_dir = output_path / accession / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Update status to running
    status_file = Path(status_dir) / f"{accession}_status.json"
    if status_file.exists():
        with open(status_file, 'r') as f:
            status = json.load(f)
        status.update({
            'state': 'running',
            'started_time': start_time.isoformat(),
            'process_id': process_id
        })
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

    try:
        # Docker paths (inside container)
        docker_output_dir = "/workspace/output"
        docker_resource_dir = "/workspace/resources"

        # Build Docker command
        cmd = [
            'docker', 'run', '--rm',
        ]

        # Storage configuration to prevent host disk exhaustion
        if container_tmpfs_gb and container_tmpfs_gb > 0:
            cmd += ['--tmpfs', f'/tmp:rw,size={int(container_tmpfs_gb)}g']
        cmd += ['--shm-size=4g']  # Increase shared memory

        # Mount volumes
        cmd += [
            '-v', f'{output_path}:{docker_output_dir}',
            '-v', f'{resource_path}:{docker_resource_dir}',
            '-v', f'{project_root}:/workspace/src',
            '-v', f'{temp_dir}:/workspace/temp',  # Bind-mount temp directory
            '-v', '/workspace/src/.venv',  # Anonymous volume to hide host .venv
            '--workdir', '/workspace/src',  # Mount source code so we use local changes
        ]

        # Pass environment variables
        cmd += [
            '-e', 'UV_NO_SYNC=1',
            '-e', 'PATH=/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            '-e', 'VIRTUAL_ENV=/workspace/.venv',
            '-e', 'TMPDIR=/workspace/temp',    # Standard Unix temp dir
            '-e', 'TEMP=/workspace/temp',      # Alternative name
            '-e', 'TMP=/workspace/temp',       # Windows compatibility
            '-e', f'ENTREZ_EMAIL={os.getenv("ENTREZ_EMAIL", "")}',
            '-e', f'OPENAI_API_KEY={os.getenv("OPENAI_API_KEY", "")}',
            '-e', f'ENTREZ_API_KEY={os.getenv("ENTREZ_API_KEY", "")}',
        ]

        # Use the specified Docker image and run command
        # Don't use 'uv run' - the container's venv is already in PATH via env vars
        cmd += [
            docker_image,
            'python', '-m', 'uorca.graph.runner',
            '--accession', accession,
            '--output_dir', docker_output_dir,
            '--resource_dir', docker_resource_dir
        ]

        if cleanup:
            cmd.append('--cleanup')

        # Run the analysis
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=21600  # 6 hours timeout
        )

        end_time = datetime.now()
        success = result.returncode == 0

        # Extract container ID if available (for cleanup purposes)
        container_id = None
        if result.stderr:
            # Try to extract container ID from Docker output
            import re
            container_match = re.search(r'[a-f0-9]{12,}', result.stderr)
            if container_match:
                container_id = container_match.group()

        # Update final status
        final_status = {
            'state': 'completed' if success else 'failed',
            'completed_time': end_time.isoformat(),
            'runtime_seconds': (end_time - start_time).total_seconds(),
            'return_code': result.returncode,
            'success': success,
            'stdout': result.stdout[-2000:] if result.stdout else '',  # Last 2000 chars
            'stderr': result.stderr[-2000:] if result.stderr else '',   # Last 2000 chars
            'container_id': container_id
        }

        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            status.update(final_status)
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)

        return {
            'accession': accession,
            'success': success,
            'return_code': result.returncode,
            'runtime': (end_time - start_time).total_seconds(),
            'stdout': result.stdout,
            'stderr': result.stderr,
            'container_id': container_id
        }

    except subprocess.TimeoutExpired:
        # Handle timeout
        final_status = {
            'state': 'timeout',
            'completed_time': datetime.now().isoformat(),
            'error': 'Job exceeded 6 hour timeout',
            'return_code': -1
        }

        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            status.update(final_status)
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)

        return {
            'accession': accession,
            'success': False,
            'error': 'timeout',
            'runtime': 21600
        }

    except Exception as e:
        # Handle other errors
        final_status = {
            'state': 'failed',
            'completed_time': datetime.now().isoformat(),
            'error': str(e),
            'return_code': -1
        }

        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            status.update(final_status)
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)

        return {
            'accession': accession,
            'success': False,
            'error': str(e),
            'runtime': 0
        }

    finally:
        # Clean up temp directory regardless of success/failure
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                # Note: Logger not available in this process, so no logging here
            except Exception as cleanup_error:
                # Cleanup failure is non-critical, just skip
                pass


class LocalBatchProcessor(BatchProcessor):
    """
    Local parallel batch processor for UORCA datasets using Docker.

    This processor handles parallel execution of dataset analyses on the local machine
    using Docker containers, with storage-aware queueing and proper job cancellation.
    """

    def __init__(self):
        """Initialize the local batch processor."""
        super().__init__()
        self.executor = None
        self.running_jobs = {}  # accession -> {'future': Future, 'storage_gb': float}
        self.job_queue = []     # List of datasets waiting to run
        self.current_storage_used = 0.0
        self.job_counter = 0
        self.cancelled = False
        self.monitor_thread = None
        self.lock = threading.Lock()

    def validate_csv_file(self, input_path: str) -> tuple[pd.DataFrame, str]:
        """
        Validate input and prepare dataset information.
        Override base class to handle local-specific storage defaults.

        Args:
            input_path: Either CSV file path or directory containing CSV and metadata

        Returns:
            Tuple of (validated DataFrame, research_question)

        Raises:
            FileNotFoundError: If input doesn't exist or CSV not found
            ValueError: If required columns are missing
        """
        # Call parent method for basic validation and research question extraction
        df, research_question = super().validate_csv_file(input_path)

        # Apply local-specific storage handling
        if 'SafeStorageGB' in df.columns and (df['SafeStorageGB'] == 0).any():
            # Override zero storage values with local default for datasets without size info
            zero_storage_mask = df['SafeStorageGB'] == 0
            if zero_storage_mask.any():
                df.loc[zero_storage_mask, 'SafeStorageGB'] = 30.0
                print("Warning: Using default value of 30GB per dataset for entries without size information")

        return df, research_question

    @property
    def default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for local batch processing."""
        # Use 75% of available CPU cores, minimum 1
        max_workers = max(1, int(mp.cpu_count() * 0.75))

        # Default storage limit: min(50 GB, 75% of free disk space)
        # Note: Final value is recomputed against the actual output directory in submit_datasets
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            max_storage_gb = int(min(50, free_gb * 0.75))
        except Exception:
            # Fallback to a safe default if disk stats are unavailable
            max_storage_gb = 50

        return {
            'max_workers': max_workers,
            'max_storage_gb': max_storage_gb,
            'cleanup': True,  # Default to cleanup
            'resource_dir': './data/kallisto_indices/',
            'check_interval': 600,  # 10 minutes
            'timeout_hours': 6,
            'docker_image': 'kevingchen/uorca:0.1.0',
            'container_tmpfs_gb': 20
        }

    def load_environment_variables(self):
        """Load environment variables from .env file."""
        # Try to find and load .env file
        project_root = Path(__file__).resolve().parent.parent.parent
        env_file = project_root / ".env"

        if env_file.exists():
            load_dotenv(env_file)
            print(f"Loaded environment variables from {env_file}")
        else:
            # Try to find .env in current directory or parent directories
            load_dotenv(find_dotenv())

    def check_environment_requirements(self) -> List[str]:
        """
        Check if all required tools are available.

        Note: Environment variable validation is now handled at the CLI level.

        Returns:
            List of missing requirements (empty if all satisfied)
        """
        missing = []

        # Check for Docker
        try:
            result = subprocess.run(['docker', '--version'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                missing.append("Docker (not available or not running)")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            missing.append("Docker (not installed or not in PATH)")

        return missing

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            if not self.cancelled:
                print(f"\nReceived interrupt signal. Cancelling running jobs...")
                self.cancelled = True
                self.cancel_all_jobs()
                # Exit immediately to prevent reentrant calls
                os._exit(1)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def cancel_all_jobs(self):
        """Cancel all running jobs and Docker containers."""
        try:
            with self.lock:
                # Cancel all futures
                for accession, job_info in self.running_jobs.items():
                    future = job_info['future']
                    if not future.done():
                        print(f"  Cancelling {accession}...")
                        future.cancel()

                # Try to kill any running Docker containers
                self._kill_docker_containers()

                # Clear the queue
                self.job_queue.clear()

            # Shutdown executor forcefully
            if self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None

            print("All jobs cancelled.")
        except Exception as e:
            print(f"Error during cancellation: {e}")

    def _kill_docker_containers(self):
        """Kill any running Docker containers for this batch."""
        try:
            # Get all running containers
            result = subprocess.run([
                'docker', 'ps', '--format', '{{.ID}} {{.Image}}'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line and 'kevingchen/uorca' in line:
                        container_id = line.split()[0]
                        print(f"  Killing Docker container: {container_id}")
                        subprocess.run(['docker', 'kill', container_id],
                                     capture_output=True, timeout=10)
        except Exception as e:
            print(f"  Warning: Could not kill Docker containers: {e}")

    def submit_next_job_if_possible(self, params: Dict[str, Any], status_dir: Path):
        """Submit the next job from queue if storage allows."""
        with self.lock:
            if not self.job_queue or self.cancelled:
                return

            # Find the largest job that fits in remaining storage
            max_storage = params['max_storage_gb']
            available_storage = max_storage - self.current_storage_used

            # Respect max_workers: do not submit beyond allowed parallelism
            if len(self.running_jobs) >= params['max_workers']:
                return

            for i, job_info in enumerate(self.job_queue):
                row = job_info['row']
                storage_needed = row['SafeStorageGB']

                if storage_needed <= available_storage:
                    # Submit this job
                    accession = row['Accession']
                    job_id = f"local_{self.job_counter:04d}"
                    self.job_counter += 1

                    # Create job status file
                    self.create_job_status_file(
                        status_dir=status_dir,
                        accession=accession,
                        job_id=job_id,
                        storage_gb=storage_needed,
                        max_workers=params['max_workers'],
                        docker_image=params['docker_image'],
                        container_tmpfs_gb=params.get('container_tmpfs_gb', 20)
                    )

                    # Submit job (check executor is available)
                    if not self.executor:
                        print(f"  ❌ Cannot submit {accession}: executor not available")
                        return

                    future = self.executor.submit(
                        run_single_dataset_local,
                        accession=accession,
                        output_dir=job_info['output_dir'],
                        resource_dir=params['resource_dir'],
                        cleanup=params['cleanup'],
                        status_dir=str(status_dir),
                        docker_image=params['docker_image'],
                        container_tmpfs_gb=params.get('container_tmpfs_gb', 20)
                    )

                    # Track the running job
                    self.running_jobs[accession] = {
                        'future': future,
                        'storage_gb': storage_needed
                    }
                    self.current_storage_used += storage_needed

                    # Remove from queue
                    self.job_queue.pop(i)

                    print(f"  Started {accession} (job {job_id}) - Storage: {self.current_storage_used:.1f}/{max_storage:.1f} GB")
                    print(f"  Queue: {len(self.job_queue)} datasets waiting")
                    break

    def monitor_jobs(self, params: Dict[str, Any], status_dir: Path):
        """Monitor running jobs and submit new ones as they complete."""
        check_interval = params['check_interval']

        while not self.cancelled and (self.running_jobs or self.job_queue):
            time.sleep(check_interval)

            if self.cancelled:
                break

            completed_jobs = []

            with self.lock:
                for accession, job_info in list(self.running_jobs.items()):
                    future = job_info['future']

                    if future.done():
                        completed_jobs.append(accession)
                        storage_freed = job_info['storage_gb']
                        self.current_storage_used -= storage_freed

                        try:
                            result = future.result()
                            if result['success']:
                                print(f"  {accession} completed successfully ({result['runtime']:.0f}s)")
                            else:
                                error_msg = result.get('error', 'Unknown error')
                                if 'stderr' in result and result['stderr']:
                                    print(f"  {accession} failed: {error_msg}")
                                    print(f"      stderr: {result['stderr'][:200]}...")
                                else:
                                    print(f"  {accession} failed: {error_msg}")
                        except Exception as e:
                            print(f"  {accession} failed with exception: {e}")

                        # Remove from running jobs
                        del self.running_jobs[accession]

            # Try to submit next job if any completed
            if completed_jobs and not self.cancelled:
                self.submit_next_job_if_possible(params, status_dir)

            # Show current status
            if not self.cancelled:
                running_count = len(self.running_jobs)
                queue_count = len(self.job_queue)
                if running_count > 0 or queue_count > 0:
                    print(f"  Running: {running_count}, Queued: {queue_count}, Storage: {self.current_storage_used:.1f}/{params['max_storage_gb']:.1f} GB")

    def submit_datasets(self, input_path: str, output_dir: str, **kwargs) -> int:
        """
        Submit datasets for local batch processing.

        Args:
            input_path: Either CSV file with dataset information or directory containing CSV and metadata
            output_dir: Output directory for results
            **kwargs: Additional parameters

        Returns:
            Number of jobs submitted
        """
        # Setup signal handlers
        self.setup_signal_handlers()

        # Merge default parameters with provided kwargs
        params = {**self.default_parameters, **kwargs}

        # Validate inputs and extract research question
        df, research_question = self.validate_csv_file(input_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # If max_storage_gb not explicitly provided, compute from the output directory's filesystem
        if 'max_storage_gb' not in kwargs or kwargs.get('max_storage_gb') is None:
            try:
                disk = psutil.disk_usage(str(output_path))
                free_gb = disk.free / (1024**3)
                params['max_storage_gb'] = int(min(50, free_gb * 0.75))
                print(f"Auto-configured max_storage_gb to {params['max_storage_gb']} GB (min(50 GB, 75% of free space on {output_path.resolve()}))")
            except Exception as e:
                print(f"Warning: Could not determine disk space for {output_path}: {e}. Using default max_storage_gb={params['max_storage_gb']} GB")

        # Save research question if available
        self.save_research_question(output_dir, research_question)

        # Check tool requirements (environment variables checked at CLI level)
        missing_reqs = self.check_environment_requirements()
        if missing_reqs:
            raise EnvironmentError(f"Missing requirements: {missing_reqs}")

        # Test Docker image availability
        try:
            print(f"Checking Docker image: {params['docker_image']}")
            result = subprocess.run([
                'docker', 'image', 'inspect', params['docker_image']
            ], capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                print(f"Pulling Docker image: {params['docker_image']}")
                pull_result = subprocess.run([
                    'docker', 'pull', params['docker_image']
                ], timeout=300)  # 5 minute timeout for pull

                if pull_result.returncode != 0:
                    raise EnvironmentError(f"Failed to pull Docker image: {params['docker_image']}")
                else:
                    print(f"Successfully pulled Docker image: {params['docker_image']}")
            else:
                print(f"Docker image {params['docker_image']} is available")

        except subprocess.TimeoutExpired:
            raise EnvironmentError("Docker operations timed out. Check Docker daemon status.")
        except Exception as e:
            raise EnvironmentError(f"Docker error: {e}")

        # Setup job tracking
        status_dir = self.setup_job_tracking(output_dir)
        logs_dir = output_path / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Sort datasets by size (largest first for better resource utilization)
        df_sorted = df.sort_values('SafeStorageGB', ascending=False)

        # Display scheduling information
        self.display_scheduling_order(df_sorted, params['max_storage_gb'])

        # Filter out datasets that exceed storage limit individually
        datasets_to_process = []
        for _, row in df_sorted.iterrows():
            accession = row['Accession']
            storage_needed = row['SafeStorageGB']

            if storage_needed > params['max_storage_gb']:
                print(f"Skipping {accession}: exceeds storage limit ({storage_needed:.2f} > {params['max_storage_gb']:.2f} GB)")
                continue

            datasets_to_process.append({
                'row': row,
                'output_dir': output_dir
            })

        if not datasets_to_process:
            print("No datasets to process after storage filtering.")
            return 0

        total_storage_needed = sum(job['row']['SafeStorageGB'] for job in datasets_to_process)

        print(f"\nStarting local parallel processing...")
        print(f"   Docker image: {params['docker_image']}")
        print(f"   Max workers: {params['max_workers']}")
        print(f"   Total datasets: {len(datasets_to_process)}")
        print(f"   Total storage needed: {total_storage_needed:.2f} GB")
        print(f"   Storage limit: {params['max_storage_gb']:.2f} GB")
        print(f"   Cleanup enabled: {params['cleanup']}")

        # Initialize the queue with all datasets (largest first)
        self.job_queue = datasets_to_process.copy()

        # Initialize process pool executor
        self.executor = ProcessPoolExecutor(max_workers=params['max_workers'])

        # Initialize counters
        initial_submitted = 0

        try:
            # Submit initial jobs up to storage limit
            while self.job_queue and not self.cancelled:
                self.submit_next_job_if_possible(params, status_dir)
                current_submitted = len(self.running_jobs)
                if current_submitted == initial_submitted:
                    break  # No more jobs could be submitted due to storage
                initial_submitted = current_submitted

            if self.running_jobs:
                print(f"\nInitial jobs submitted: {len(self.running_jobs)}")
                print(f"Jobs in queue: {len(self.job_queue)}")
                print(f"Job logs will be written to: {logs_dir}")
                print(f"Job status tracking: {status_dir}")
                print(f"\nMonitoring job progress (checking every {params['check_interval']//60} minutes)...")
                print("   Press Ctrl+C to cancel all jobs")

                # Start monitoring in a separate thread
                self.monitor_thread = threading.Thread(
                    target=self.monitor_jobs,
                    args=(params, status_dir)
                )
                self.monitor_thread.start()

                # Wait for monitoring to complete
                if self.monitor_thread:
                    self.monitor_thread.join()

            else:
                print("No jobs could be submitted due to storage constraints.")

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            self.cancelled = True
        finally:
            # Clean up executor
            if self.executor and not self.cancelled:
                self.executor.shutdown(wait=True)
                self.executor = None
            elif self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None

        total_submitted = initial_submitted + len([job for job in datasets_to_process if job not in self.job_queue])
        return total_submitted

    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check the status of a specific local job.

        Args:
            job_id: Local job ID

        Returns:
            Dictionary containing job status information
        """
        return {
            'job_id': job_id,
            'state': 'UNKNOWN',
            'system': 'local',
            'found': False,
            'note': 'Use job status files for local job tracking'
        }

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a local job.

        Args:
            job_id: Local job ID

        Returns:
            True if job was successfully cancelled, False otherwise
        """
        # Try to cancel futures if they exist
        cancelled_count = 0

        with self.lock:
            for accession, job_info in list(self.running_jobs.items()):
                future = job_info['future']
                if future and not future.done():
                    if future.cancel():
                        cancelled_count += 1
                        del self.running_jobs[accession]
                        print(f"Cancelled job for {accession}")

        return cancelled_count > 0

    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get information about available local resources.

        Returns:
            Dictionary containing resource information
        """
        # Get system information
        cpu_count = mp.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Get load average if available (Unix systems)
        load_avg = None
        try:
            load_avg = os.getloadavg()
        except AttributeError:
            pass  # Windows doesn't have getloadavg

        # Check Docker status
        docker_status = "unknown"
        try:
            result = subprocess.run(['docker', 'info'],
                                  capture_output=True, text=True, timeout=10)
            docker_status = "running" if result.returncode == 0 else "error"
        except:
            docker_status = "not available"

        return {
            'system': 'local',
            'cpu_cores': cpu_count,
            'cpu_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent_used': memory.percent,
            'disk_total_gb': disk.total / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'disk_percent_used': (disk.used / disk.total) * 100,
            'load_average': load_avg,
            'docker_status': docker_status,
            'active_jobs': len(self.running_jobs),
            'queued_jobs': len(self.job_queue),
            'current_storage_used_gb': self.current_storage_used
        }

    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """
        List currently active local jobs.

        Returns:
            List of job information dictionaries
        """
        jobs = []

        with self.lock:
            for accession, job_info in self.running_jobs.items():
                future = job_info['future']
                jobs.append({
                    'accession': accession,
                    'state': 'running',
                    'system': 'local',
                    'storage_gb': job_info['storage_gb']
                })

            # Add queued jobs
            for job_info in self.job_queue:
                row = job_info['row']
                jobs.append({
                    'accession': row['Accession'],
                    'state': 'queued',
                    'system': 'local',
                    'storage_gb': row['SafeStorageGB']
                })

        return jobs

    def cleanup(self):
        """Clean up resources and shutdown executor."""
        self.cancelled = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None
        self.running_jobs.clear()
        self.job_queue.clear()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
