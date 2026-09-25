"""Test that submit_user_data_job writes a sbatch script whose filename is
unique per user-config invocation, so concurrent or sequential submissions
don't clobber each other."""

from pathlib import Path

import yaml


def _write_user_config(tmp_path, fastq_dir_name, metadata_name, shared_output_dir):
    fastq_dir = tmp_path / fastq_dir_name
    fastq_dir.mkdir()
    metadata = tmp_path / metadata_name
    metadata.write_text("sample_id\nS1\n")

    cfg_path = tmp_path / f"{fastq_dir_name}.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "mode": "user",
        "user_data": {
            "fastq_dir": str(fastq_dir),
            "metadata_path": str(metadata),
        },
        # Both configs share the same output_dir to expose any filename collision.
        "output_dir": str(shared_output_dir),
        "resource_dir": str(tmp_path / "res"),
    }))
    return cfg_path


def test_unique_sbatch_filename_per_dataset(monkeypatch, tmp_path):
    from uorca.batch.slurm import SlurmBatchProcessor

    shared_out = tmp_path / "shared_out"
    cfg_a = _write_user_config(tmp_path, "fq_a", "meta_a.csv", shared_out)
    cfg_b = _write_user_config(tmp_path, "fq_b", "meta_b.csv", shared_out)

    processor = SlurmBatchProcessor.__new__(SlurmBatchProcessor)

    fake_default_parameters = {
        "partition": "test", "constraint": "",
        "cpus_per_task": 1, "memory": "4G", "time_limit": "1:00:00",
    }
    # default_parameters is a read-only property — patch it on the class.
    monkeypatch.setattr(
        SlurmBatchProcessor,
        "default_parameters",
        property(lambda self: fake_default_parameters),
    )

    written: list[Path] = []

    def fake_run(cmd, **kw):
        # Capture the sbatch path before unlink runs.
        written.append(Path(cmd[1]))
        class R:
            returncode = 0
            stdout = "Submitted batch job 12345"
            stderr = ""
        return R()

    monkeypatch.setattr("uorca.batch.slurm.subprocess.run", fake_run)

    processor.submit_user_data_job(str(cfg_a))
    processor.submit_user_data_job(str(cfg_b))

    assert len(written) == 2
    assert written[0] != written[1], (
        f"Expected unique sbatch filenames, got: {written}"
    )
