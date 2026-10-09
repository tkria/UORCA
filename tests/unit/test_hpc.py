import pytest


def test_parse_squeue_output():
    from uorca.gui.hpc.slurm import parse_squeue_output
    raw = """60687927|temp_submit_GSE155237.sba|RUNNING|3:52|6:00:00|example_partition|(None)
60687928|temp_submit_GSE309698.sba|RUNNING|3:52|6:00:00|example_partition|(None)
60687940|temp_submit_GSE326313.sba|PENDING|0:00|6:00:00|example_partition|(Resources)"""
    jobs = parse_squeue_output(raw)
    assert len(jobs) == 3
    assert jobs[0]["job_id"] == "60687927"
    assert jobs[0]["state"] == "RUNNING"
    assert jobs[2]["state"] == "PENDING"
    assert jobs[2]["reason"] == "(Resources)"


def test_parse_squeue_empty():
    from uorca.gui.hpc.slurm import parse_squeue_output
    jobs = parse_squeue_output("")
    assert jobs == []


def test_parse_sacct_output():
    from uorca.gui.hpc.slurm import parse_sacct_output
    raw = """60687927|temp_submit_GSE155237|COMPLETED|0:0|2026-04-16T13:10:00|2026-04-16T14:52:00|01:42:00|4096K
60687928|temp_submit_GSE309698|FAILED|1:0|2026-04-16T13:10:00|2026-04-16T14:30:00|01:20:00|8192K"""
    jobs = parse_sacct_output(raw)
    assert len(jobs) == 2
    assert jobs[0]["state"] == "COMPLETED"
    assert jobs[1]["state"] == "FAILED"
