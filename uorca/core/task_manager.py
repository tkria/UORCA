"""
Background task management for UORCA Streamlit app.

Uses threading for non-blocking execution and SQLite for persistence.
"""

import logging
import sqlite3
import threading
import time
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List
import json

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskManager:
    """
    Manages background tasks with SQLite persistence.

    Thread-safe singleton for managing long-running tasks in Streamlit.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: str = None):
        """Initialize task manager with SQLite database."""
        if self._initialized:
            return

        if db_path is None:
            # Default to user's home directory for persistence
            home = Path.home()
            uorca_dir = home / ".uorca"
            uorca_dir.mkdir(exist_ok=True)
            db_path = str(uorca_dir / "tasks.db")

        self.db_path = db_path
        self.tasks: Dict[str, threading.Thread] = {}
        self.task_states: Dict[str, Dict[str, Any]] = {}
        self._setup_database()
        self._load_existing_tasks()
        self._initialized = True

    def _setup_database(self):
        """Create tasks table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                progress REAL DEFAULT 0.0,
                progress_message TEXT,
                result_path TEXT,
                error_message TEXT,
                parameters TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _load_existing_tasks(self):
        """Load task states from database on startup."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE status IN ('pending', 'running')")
        rows = cursor.fetchall()
        conn.close()

        # Mark interrupted tasks as failed
        if rows:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            for row in rows:
                task_id = row[0]
                cursor.execute("""
                    UPDATE tasks
                    SET status = ?, error_message = ?, completed_at = ?
                    WHERE task_id = ?
                """, (TaskStatus.FAILED.value,
                      "Task interrupted by app restart",
                      datetime.now().isoformat(),
                      task_id))
            conn.commit()
            conn.close()

    def submit_task(
        self,
        task_id: str,
        task_type: str,
        task_func: Callable,
        parameters: Dict[str, Any],
        on_progress: Optional[Callable[[float, str], None]] = None,
        on_complete: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ) -> str:
        """
        Submit a new background task.

        Args:
            task_id: Unique identifier for this task
            task_type: Type of task (e.g., "identify", "run")
            task_func: Function to execute in background
            parameters: Parameters to pass to task_func
            on_progress: Optional callback for progress updates
            on_complete: Optional callback when task completes
            on_error: Optional callback on error

        Returns:
            task_id
        """
        # Store in database
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO tasks (task_id, task_type, status, created_at, parameters)
            VALUES (?, ?, ?, ?, ?)
        """, (task_id, task_type, TaskStatus.PENDING.value,
              datetime.now().isoformat(), json.dumps(parameters)))
        conn.commit()
        conn.close()

        # Initialize task state
        self.task_states[task_id] = {
            "status": TaskStatus.PENDING,
            "progress": 0.0,
            "progress_message": "Waiting to start...",
            "result": None,
            "error": None
        }

        # Create and start thread
        def task_wrapper():
            try:
                # Update status to running
                self._update_task_status(task_id, TaskStatus.RUNNING, 0.0, "Starting...")

                # Execute task
                result = task_func(
                    **parameters,
                    progress_callback=lambda p, m: self._on_progress(task_id, p, m, on_progress)
                )

                # Mark complete
                self._update_task_status(task_id, TaskStatus.COMPLETED, 1.0, "Complete", result)
                if on_complete:
                    on_complete(result)

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                logger.exception("Task %s failed", task_id)
                self._update_task_status(task_id, TaskStatus.FAILED, error=error_msg)
                if on_error:
                    on_error(e)

        thread = threading.Thread(target=task_wrapper, daemon=True)
        self.tasks[task_id] = thread
        thread.start()

        return task_id

    def _update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: float = None,
        progress_message: str = None,
        result: Any = None,
        error: str = None
    ):
        """Update task status in memory and database."""
        # Update memory
        if task_id in self.task_states:
            self.task_states[task_id]["status"] = status
            if progress is not None:
                self.task_states[task_id]["progress"] = progress
            if progress_message is not None:
                self.task_states[task_id]["progress_message"] = progress_message
            if result is not None:
                self.task_states[task_id]["result"] = result
            if error is not None:
                self.task_states[task_id]["error"] = error

        # Update database
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        updates = ["status = ?"]
        values = [status.value]

        if status == TaskStatus.RUNNING and progress == 0.0:
            updates.append("started_at = ?")
            values.append(datetime.now().isoformat())

        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            updates.append("completed_at = ?")
            values.append(datetime.now().isoformat())

        if progress is not None:
            updates.append("progress = ?")
            values.append(progress)

        if progress_message is not None:
            updates.append("progress_message = ?")
            values.append(progress_message)

        if result is not None:
            updates.append("result_path = ?")
            values.append(str(result) if isinstance(result, Path) else result)

        if error is not None:
            updates.append("error_message = ?")
            values.append(error)

        values.append(task_id)

        cursor.execute(f"""
            UPDATE tasks SET {', '.join(updates)}
            WHERE task_id = ?
        """, values)
        conn.commit()
        conn.close()

    def _on_progress(self, task_id: str, progress: float, message: str,
                     user_callback: Optional[Callable] = None):
        """Handle progress updates."""
        self._update_task_status(task_id, TaskStatus.RUNNING, progress, message)
        if user_callback:
            user_callback(progress, message)

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current status of a task (always fetches from database for freshness)."""
        # Always check database first to avoid stale cache issues
        # This is especially important in Streamlit where reruns can cause cache staleness
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            status = {
                "status": TaskStatus(row[2]),
                "progress": row[6] or 0.0,
                "progress_message": row[7] or "",
                "result": row[8],
                "error": row[9]
            }
            # Update in-memory cache with fresh data
            if task_id in self.task_states:
                self.task_states[task_id].update(status)
            return status

        # Fallback to in-memory cache if not in database
        if task_id in self.task_states:
            return self.task_states[task_id].copy()

        return None

    def list_tasks(self, task_type: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent tasks, optionally filtered by type."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()

        if task_type:
            cursor.execute("""
                SELECT * FROM tasks
                WHERE task_type = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (task_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM tasks
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        tasks = []
        for row in rows:
            tasks.append({
                "task_id": row[0],
                "task_type": row[1],
                "status": row[2],
                "created_at": row[3],
                "started_at": row[4],
                "completed_at": row[5],
                "progress": row[6] or 0.0,
                "progress_message": row[7] or "",
                "result_path": row[8],
                "error_message": row[9],
                "parameters": json.loads(row[10]) if row[10] else {}
            })

        return tasks

    def cancel_task(self, task_id: str) -> bool:
        """
        Attempt to cancel a running task.

        Note: Python threading doesn't support true cancellation,
        so this just marks the task as cancelled. The task function
        should check progress_callback return value to support cancellation.
        """
        if task_id in self.task_states:
            if self.task_states[task_id]["status"] == TaskStatus.RUNNING:
                self._update_task_status(task_id, TaskStatus.CANCELLED)
                return True
        return False
