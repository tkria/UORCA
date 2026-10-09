"""Unit tests for TaskManager."""

import pytest
import tempfile
from pathlib import Path
import time
import threading

from uorca.core.task_manager import TaskManager, TaskStatus


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    # Reset singleton for each test
    TaskManager._instance = None

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.mark.xfail(
    strict=True,
    reason="Known defect: results round-trip through the result_path TEXT column and come back "
    "stringified ('10' != 10). Remove this marker when TaskManager preserves result types.",
)
def test_task_submission(temp_db):
    """Test basic task submission."""
    manager = TaskManager(db_path=temp_db)

    def simple_task(value, progress_callback=None):
        if progress_callback:
            progress_callback(0.5, "Halfway")
        time.sleep(0.1)
        return value * 2

    task_id = manager.submit_task(
        task_id="test_1",
        task_type="test",
        task_func=simple_task,
        parameters={"value": 5}
    )

    assert task_id == "test_1"

    # Wait for completion
    time.sleep(0.3)

    status = manager.get_task_status(task_id)
    assert status["status"] == TaskStatus.COMPLETED
    assert status["result"] == 10


def test_progress_tracking(temp_db):
    """Test progress callback mechanism."""
    manager = TaskManager(db_path=temp_db)

    progress_updates = []

    def tracking_task(progress_callback=None):
        for i in range(5):
            if progress_callback:
                progress_callback(i / 4, f"Step {i+1}/5")
            time.sleep(0.05)
        return "done"

    def on_progress(progress, message):
        progress_updates.append((progress, message))

    task_id = manager.submit_task(
        task_id="test_progress",
        task_type="test",
        task_func=tracking_task,
        parameters={},
        on_progress=on_progress
    )

    time.sleep(0.4)

    assert len(progress_updates) >= 4
    assert progress_updates[-1][0] >= 0.75


def test_error_handling(temp_db):
    """Test task error handling."""
    manager = TaskManager(db_path=temp_db)

    def failing_task(progress_callback=None):
        raise ValueError("Test error")

    task_id = manager.submit_task(
        task_id="test_error",
        task_type="test",
        task_func=failing_task,
        parameters={}
    )

    time.sleep(0.2)

    status = manager.get_task_status(task_id)
    assert status["status"] == TaskStatus.FAILED
    assert "ValueError: Test error" in status["error"]


def test_concurrent_tasks(temp_db):
    """Test multiple concurrent tasks."""
    manager = TaskManager(db_path=temp_db)

    def concurrent_task(task_num, progress_callback=None):
        time.sleep(0.1)
        return f"task_{task_num}_done"

    task_ids = []
    for i in range(5):
        task_id = manager.submit_task(
            task_id=f"concurrent_{i}",
            task_type="test",
            task_func=concurrent_task,
            parameters={"task_num": i}
        )
        task_ids.append(task_id)

    time.sleep(0.3)

    for task_id in task_ids:
        status = manager.get_task_status(task_id)
        assert status["status"] == TaskStatus.COMPLETED


def test_persistence(temp_db):
    """Test SQLite persistence across manager instances."""
    # Create and run task
    manager1 = TaskManager(db_path=temp_db)

    def persistent_task(progress_callback=None):
        return "persisted"

    task_id = manager1.submit_task(
        task_id="persist_test",
        task_type="test",
        task_func=persistent_task,
        parameters={}
    )

    time.sleep(0.2)

    # Reset singleton and create new manager instance
    TaskManager._instance = None
    manager2 = TaskManager(db_path=temp_db)

    # Should be able to retrieve task from database
    tasks = manager2.list_tasks(task_type="test")
    assert any(t["task_id"] == "persist_test" for t in tasks)


def test_list_tasks_filtering(temp_db):
    """Test list_tasks with type filtering."""
    manager = TaskManager(db_path=temp_db)

    def quick_task(progress_callback=None):
        return "done"

    # Submit tasks of different types
    manager.submit_task("identify_1", "identify", quick_task, {})
    manager.submit_task("pipeline_1", "pipeline", quick_task, {})
    manager.submit_task("identify_2", "identify", quick_task, {})

    time.sleep(0.2)

    # Filter by type
    identify_tasks = manager.list_tasks(task_type="identify")
    assert len(identify_tasks) == 2
    assert all(t["task_type"] == "identify" for t in identify_tasks)

    pipeline_tasks = manager.list_tasks(task_type="pipeline")
    assert len(pipeline_tasks) == 1
    assert pipeline_tasks[0]["task_type"] == "pipeline"


def test_list_tasks_limit(temp_db):
    """Test list_tasks respects limit parameter."""
    manager = TaskManager(db_path=temp_db)

    def quick_task(progress_callback=None):
        return "done"

    # Submit many tasks
    for i in range(20):
        manager.submit_task(f"task_{i}", "test", quick_task, {})

    time.sleep(0.5)

    # Test limit
    tasks = manager.list_tasks(limit=5)
    assert len(tasks) <= 5


def test_cancel_task(temp_db):
    """Test task cancellation."""
    manager = TaskManager(db_path=temp_db)

    def long_running_task(progress_callback=None):
        time.sleep(5)
        return "completed"

    task_id = manager.submit_task(
        task_id="long_task",
        task_type="test",
        task_func=long_running_task,
        parameters={}
    )

    time.sleep(0.1)  # Let it start

    # Cancel the task
    result = manager.cancel_task(task_id)
    assert result is True

    status = manager.get_task_status(task_id)
    assert status["status"] == TaskStatus.CANCELLED


def test_cancel_nonexistent_task(temp_db):
    """Test cancelling a non-existent task."""
    manager = TaskManager(db_path=temp_db)

    result = manager.cancel_task("nonexistent")
    assert result is False


def test_get_nonexistent_task(temp_db):
    """Test getting status of non-existent task."""
    manager = TaskManager(db_path=temp_db)

    status = manager.get_task_status("nonexistent")
    assert status is None


def test_task_timestamps(temp_db):
    """Test that timestamps are recorded correctly."""
    manager = TaskManager(db_path=temp_db)

    def quick_task(progress_callback=None):
        time.sleep(0.1)
        return "done"

    task_id = manager.submit_task(
        task_id="timestamp_test",
        task_type="test",
        task_func=quick_task,
        parameters={}
    )

    time.sleep(0.3)

    tasks = manager.list_tasks()
    task = next((t for t in tasks if t["task_id"] == "timestamp_test"), None)

    assert task is not None
    assert task["created_at"] is not None
    assert task["started_at"] is not None
    assert task["completed_at"] is not None


def test_interrupted_tasks_marked_failed(temp_db):
    """Test that interrupted tasks are marked as failed on restart."""
    # Create manager and submit running task
    manager1 = TaskManager(db_path=temp_db)

    def long_task(progress_callback=None):
        time.sleep(10)
        return "done"

    task_id = manager1.submit_task(
        task_id="interrupted_test",
        task_type="test",
        task_func=long_task,
        parameters={}
    )

    time.sleep(0.1)  # Let it start

    # Simulate restart by creating new manager instance
    TaskManager._instance = None
    manager2 = TaskManager(db_path=temp_db)

    # Check that the interrupted task is marked as failed
    tasks = manager2.list_tasks()
    interrupted_task = next((t for t in tasks if t["task_id"] == "interrupted_test"), None)

    assert interrupted_task is not None
    assert interrupted_task["status"] == TaskStatus.FAILED.value
    assert "interrupted by app restart" in interrupted_task["error_message"]


def test_singleton_pattern(temp_db):
    """Test that TaskManager is a singleton."""
    manager1 = TaskManager(db_path=temp_db)
    manager2 = TaskManager()  # Should return same instance

    assert manager1 is manager2


def test_on_complete_callback(temp_db):
    """Test on_complete callback is invoked."""
    manager = TaskManager(db_path=temp_db)

    completed_results = []

    def task_with_result(progress_callback=None):
        return "result_value"

    def on_complete(result):
        completed_results.append(result)

    manager.submit_task(
        task_id="callback_test",
        task_type="test",
        task_func=task_with_result,
        parameters={},
        on_complete=on_complete
    )

    time.sleep(0.2)

    assert len(completed_results) == 1
    assert completed_results[0] == "result_value"


def test_on_error_callback(temp_db):
    """Test on_error callback is invoked."""
    manager = TaskManager(db_path=temp_db)

    error_exceptions = []

    def failing_task(progress_callback=None):
        raise RuntimeError("Task failed")

    def on_error(exception):
        error_exceptions.append(exception)

    manager.submit_task(
        task_id="error_callback_test",
        task_type="test",
        task_func=failing_task,
        parameters={},
        on_error=on_error
    )

    time.sleep(0.2)

    assert len(error_exceptions) == 1
    assert isinstance(error_exceptions[0], RuntimeError)
    assert str(error_exceptions[0]) == "Task failed"
