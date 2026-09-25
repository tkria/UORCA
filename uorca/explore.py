"""
UORCA Explorer CLI Wrapper

This module provides a simplified CLI wrapper for the UORCA Explorer Streamlit application
that runs directly without container dependency.
"""

import sys
import os
import socket
import subprocess
from pathlib import Path
from dotenv import load_dotenv, find_dotenv


def is_port_available(host, port):
    """
    Check if a port is available on the given host.

    Args:
        host: Host address to check
        port: Port number to check

    Returns:
        bool: True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0
    except Exception:
        return False


def find_available_port(host, start_port, max_attempts=10):
    """
    Find an available port starting from start_port.

    Args:
        host: Host address to check
        start_port: Starting port number
        max_attempts: Maximum number of ports to try

    Returns:
        int: Available port number, or None if none found
    """
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(host, port):
            return port
    return None


def attempt_conservative_port_cleanup(host, port):
    """
    Conservatively attempt to terminate only Python processes using the specified port.
    Only kills processes if there's exactly one Python process using the port.

    Args:
        host: Host address
        port: Port number to clean up

    Returns:
        bool: True if cleanup was attempted and successful, False otherwise
    """
    try:
        import signal
        import time

        # Try to find processes using the port on Unix-like systems
        if os.name == 'posix':
            try:
                # Use lsof to find processes using the port, filtering for Python
                result = subprocess.run(['lsof', '-i', f':{port}'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    python_processes = []

                    for line in lines:
                        if 'python' in line.lower() or 'streamlit' in line.lower():
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    pid = int(parts[1])
                                    python_processes.append(pid)
                                except ValueError:
                                    continue

                    if len(python_processes) == 1:
                        pid = python_processes[0]
                        try:
                            os.kill(pid, signal.SIGTERM)
                            print(f"Terminated Python process {pid} using port {port}")
                            time.sleep(2)
                            return True
                        except (ProcessLookupError, PermissionError) as e:
                            print(f"Could not terminate process {pid}: {e}")
                    elif len(python_processes) > 1:
                        print(f"Found {len(python_processes)} Python processes using port {port}. Not terminating for safety.")
                        print("Please manually stop the processes or use a different port.")
                    else:
                        print(f"No Python processes found using port {port}.")

            except (subprocess.TimeoutExpired, FileNotFoundError):
                # lsof not available or timed out
                pass

        # On Windows, be similarly conservative
        elif os.name == 'nt':
            try:
                # Find processes using the port
                result = subprocess.run(['netstat', '-ano'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    pids = []
                    for line in lines:
                        if f':{port} ' in line and 'LISTENING' in line:
                            parts = line.split()
                            if len(parts) >= 5:
                                try:
                                    pid = int(parts[-1])
                                    pids.append(pid)
                                except ValueError:
                                    pass

                    # Check if any of these PIDs are Python processes
                    python_pids = []
                    for pid in pids:
                        try:
                            tasklist_result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV'],
                                                           capture_output=True, text=True, timeout=3)
                            if 'python' in tasklist_result.stdout.lower() or 'streamlit' in tasklist_result.stdout.lower():
                                python_pids.append(pid)
                        except subprocess.TimeoutExpired:
                            pass

                    if len(python_pids) == 1:
                        pid = python_pids[0]
                        try:
                            subprocess.run(['taskkill', '/F', '/PID', str(pid)],
                                         capture_output=True, timeout=5)
                            print(f"Terminated Python process {pid} using port {port}")
                            time.sleep(1)
                            return True
                        except subprocess.TimeoutExpired:
                            print(f"Could not terminate process {pid}")
                    elif len(python_pids) > 1:
                        print(f"Found {len(python_pids)} Python processes using port {port}. Not terminating for safety.")

            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    except Exception as e:
        print(f"Port cleanup attempt failed: {e}")

    return False


def validate_dataset_success(results_dir):
    """
    Validate that the results directory contains successful UORCA analyses.

    Args:
        results_dir: Path to UORCA results directory

    Returns:
        tuple: (is_valid, error_message, success_count)
    """
    try:
        results_path = Path(results_dir).resolve()

        # Look for GSE*/metadata/analysis_info.json files
        analysis_info_files = list(results_path.glob("*/metadata/analysis_info.json"))

        if not analysis_info_files:
            return False, "No analysis_info.json files found in GSE*/metadata/ subdirectories", 0

        success_count = 0
        total_files = len(analysis_info_files)

        for info_file in analysis_info_files:
            try:
                import json
                with open(info_file, 'r') as f:
                    data = json.load(f)
                    if data.get('analysis_success') is True and data.get('unique_groups') is not None:
                        success_count += 1
            except (json.JSONDecodeError, IOError) as e:
                # Skip files that can't be read or parsed
                continue

        if success_count == 0:
            return False, f"No successful analyses found. Found {total_files} analysis files but none have 'analysis_success': true", 0

        return True, f"Found {success_count} successful analyses out of {total_files} total", success_count

    except Exception as e:
        return False, f"Error validating dataset: {e}", 0


def main(results_dir=None, port=8501, host="127.0.0.1", headless=False):
    """
    Launch UORCA Explorer Streamlit application directly.

    Args:
        results_dir: Path to UORCA results directory
        port: Port number for the web application
        host: Host address to bind to
        headless: Run in headless mode (no browser auto-open)
    """

    # Load environment variables from .env file
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Try to find .env file automatically
        load_dotenv(find_dotenv())

    # Check for OpenAI API key (required for AI features in UORCA Explorer)
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found.")
        print("AI-powered analysis features will be disabled in UORCA Explorer.")
        print("To enable AI features, add OPENAI_API_KEY=your_key to your .env file.")
        print("")

    # Validate results directory if provided
    if results_dir:
        results_path = Path(results_dir).resolve()
        if not results_path.exists():
            print(f"Error: Results directory does not exist: {results_dir}")
            sys.exit(1)
        if not results_path.is_dir():
            print(f"Error: Results path is not a directory: {results_dir}")
            sys.exit(1)

        # Validate that the directory contains successful UORCA analyses
        is_valid, validation_message, success_count = validate_dataset_success(results_dir)

        if not is_valid:
            print("""Please provide a directory with successful UORCA analyses. \n\nIf you believe that the directory is correct, please run: cat GSE*/metadata/analysis_info.json | grep '"analysis_success": true' - this checks for successful analyses.""")
            sys.exit(1)
        else:
            # Suppress success message to keep CLI output clean
            pass

        # Count analysis folders using same logic as ResultsIntegrator
        analysis_folders = []

        # Check if the results directory itself contains RNAseqAnalysis
        rnaseq_dir = results_path / "RNAseqAnalysis"
        if rnaseq_dir.is_dir():
            analysis_folders.append(results_path.name)

        # If no analysis folders found in current directory, look for subdirectories
        if not analysis_folders:
            for item in results_path.iterdir():
                if item.is_dir():
                    # Check if it contains an RNAseqAnalysis subdirectory
                    rnaseq_subdir = item / "RNAseqAnalysis"
                    if rnaseq_subdir.is_dir():
                        analysis_folders.append(item.name)

        # Validation complete - directory contains UORCA results
        pass

    # Set up environment for Streamlit
    env = os.environ.copy()

    # Ensure .env variables are available to the subprocess
    # Re-load to make sure all variables are in the current environment
    if env_file.exists():
        load_dotenv(env_file, override=False)  # Don't override existing env vars

    # Copy all current environment variables to subprocess environment
    env.update(os.environ)

    # Configure UORCA Explorer
    if results_dir:
        env['UORCA_DEFAULT_RESULTS_DIR'] = str(Path(results_dir).resolve())

    # Configure Streamlit
    env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    env['STREAMLIT_SERVER_HEADLESS'] = 'false'
    env['STREAMLIT_LOGGER_LEVEL'] = 'ERROR'
    env['STREAMLIT_CLIENT_SHOW_ERROR_DETAILS'] = 'false'
    env['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    env['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    # Ensure browser opens to 127.0.0.1 regardless of server binding address
    env['STREAMLIT_BROWSER_SERVER_ADDRESS'] = '127.0.0.1'

    # Get path to app.py
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    explorer_script = current_dir / "gui" / "app.py"

    if not explorer_script.exists():
        print(f"Error: Could not find app.py at {explorer_script}")
        sys.exit(1)

    # Check if the requested port is available
    if not is_port_available(host, port):
        print(f"Port {port} is already in use. Finding an available port...")
        available_port = find_available_port(host, port)
        if available_port is None:
            print(f"Error: Could not find an available port starting from {port}")
            print("Please try a different port or manually stop the existing service.")
            sys.exit(1)
        else:
            print(f"Using port {available_port} instead of {port}")
            port = available_port

    # Build streamlit command for direct execution
    cmd = [
        'uv', 'run', 'streamlit', 'run',
        str(explorer_script),
        '--server.port', str(port),
        '--server.address', host,
        '--server.headless', str(not headless).lower(),
        '--browser.serverAddress', '127.0.0.1',
        '--logger.level', 'error'
    ]

    print("=" * 50)
    print("UORCA Explorer")
    print("=" * 50)

    # Determine which URL(s) to show the user
    if host == "0.0.0.0":
        # When binding to all interfaces, show localhost (most common) and mention network access
        primary_url = f"http://127.0.0.1:{port}"
        print(f"Starting UORCA Explorer on all network interfaces")
        if results_dir:
            print(f"Using results directory: {results_dir}")
        print("")
        print(f"Please navigate to: {primary_url}")
        print("(Also accessible from other devices on your network via your machine's IP)")
    else:
        # Show the specific host they requested
        primary_url = f"http://{host}:{port}"
        print(f"Starting UORCA Explorer on {primary_url}")
        if results_dir:
            print(f"Using results directory: {results_dir}")
        print("")
        print(f"Please navigate to {primary_url} in your browser")
    print("")
    print("Press Ctrl+C to stop the application")
    print("=" * 50)

    try:
        # Change to project root for proper imports
        os.chdir(project_root)

        # Run streamlit directly with selective output filtering
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)

        # Filter out unwanted Streamlit messages while preserving important ones
        for line in process.stdout:
            line = line.strip()
            # Skip common Streamlit noise
            if any(skip_phrase in line.lower() for skip_phrase in [
                'welcome to streamlit',
                'if you\'d like to receive helpful onboarding emails',
                'please enter your email address',
                'you can find our privacy policy',
                'this open source library collects usage statistics',
                'telemetry data is stored',
                'if you\'d like to opt out',
                'creating that file if necessary',
                'you can now view your streamlit app',
                'for better performance, install the watchdog'
            ]):
                continue
            # Show important messages
            if line and not line.startswith('  '):
                print(f"  {line}")

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except subprocess.CalledProcessError as e:
        print(f"\nError: Failed to start UORCA Explorer: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: Could not find 'uv' command.")
        print("Make sure you're in the UORCA environment with uv installed.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nStopping UORCA Explorer...")

        # Attempt conservative cleanup of the port we were using
        print("Cleaning up...")
        attempt_conservative_port_cleanup(host, port)

        print("Thanks for using UORCA Explorer! Hopefully you found the analyses to be well orca-strated!")
        sys.exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Launch UORCA Explorer Streamlit application'
    )
    parser.add_argument('results_dir', nargs='?',
                       help='Path to UORCA results directory')
    parser.add_argument('--port', type=int, default=8501,
                       help='Port number for the web application')
    parser.add_argument('--host', default='127.0.0.1',
                       help='Host address to bind to')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no browser auto-open)')

    args = parser.parse_args()
    main(args.results_dir, args.port, args.host, args.headless)
