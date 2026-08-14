from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "p6_launcher.py"


def test_wrapper_unlinks_lock_before_publishing_terminal_status(
    tmp_path, monkeypatch
):
    # Regression (launcher timing race): the wrapper used to publish
    # status=complete and only then unlink .launch.lock, so a poller could see
    # a terminal status while the lock still existed (a duplicate launcher
    # would then refuse a run that had already finished).  The terminal status
    # must be written only after the lock is gone.  Recorded at each status
    # write, the lock-existence sequence must be [True, False]: present for the
    # running snapshot, absent for the terminal snapshot.
    import p6_job_wrapper

    run_directory = tmp_path / "durable"
    run_directory.mkdir()
    lock_path = run_directory / ".launch.lock"
    lock_path.write_text("held\n")

    real_atomic_write = p6_job_wrapper.atomic_write_json
    lock_seen_at_write = []

    def recording_write(path, payload):
        lock_seen_at_write.append(lock_path.exists())
        return real_atomic_write(path, payload)

    monkeypatch.setattr(p6_job_wrapper, "atomic_write_json", recording_write)
    monkeypatch.setattr(
        p6_job_wrapper.subprocess,
        "run",
        lambda command, check=False: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p6_job_wrapper.py",
            "--run-dir",
            str(run_directory),
            "--lock-path",
            str(lock_path),
            "--",
            "python",
            "-c",
            "pass",
        ],
    )

    exit_code = p6_job_wrapper.main()
    assert exit_code == 0
    status = json.loads((run_directory / "launcher_status.json").read_text())
    assert status["status"] == "complete"
    assert not lock_path.exists()
    assert lock_seen_at_write == [True, False]
    assert (run_directory / "LAUNCHER_COMPLETE").is_file()


def test_wrapper_lock_unlink_failure_still_publishes_terminal_status(
    tmp_path, monkeypatch
):
    # Hardening: the wrapper unlinks .launch.lock before publishing the
    # terminal status.  A non-ENOENT unlink failure (e.g. PermissionError)
    # must NEVER suppress the status publish -- the status is the only way a
    # poller learns the outcome.  Old code published first, so it survived
    # such a failure; the reordered code must too.
    import p6_job_wrapper

    run_directory = tmp_path / "durable"
    run_directory.mkdir()
    lock_path = run_directory / ".launch.lock"
    lock_path.write_text("held\n")

    real_unlink = Path.unlink

    def raising_unlink(path, *args, **kwargs):
        if path.name == ".launch.lock":
            raise PermissionError("simulated lock unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(p6_job_wrapper.Path, "unlink", raising_unlink)
    monkeypatch.setattr(
        p6_job_wrapper.subprocess,
        "run",
        lambda command, check=False: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "p6_job_wrapper.py",
            "--run-dir",
            str(run_directory),
            "--lock-path",
            str(lock_path),
            "--",
            "python",
            "-c",
            "pass",
        ],
    )

    assert p6_job_wrapper.main() == 0

    status_path = run_directory / "launcher_status.json"
    assert status_path.is_file()
    status = json.loads(status_path.read_text())
    assert status["status"] == "complete"
    assert status["exit_code"] == 0


def wait_for_status(run_directory, expected, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status_path = run_directory / "launcher_status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            if status.get("status") == expected:
                return status
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for launcher status {expected}")


def launch(run_directory, command, resume=False):
    arguments = [
        sys.executable,
        str(LAUNCHER),
        "--run-dir",
        str(run_directory),
    ]
    if resume:
        arguments.append("--resume")
    arguments.extend(["--", *command])
    return subprocess.run(
        arguments,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )


def test_launcher_detaches_records_logs_and_refuses_duplicate(tmp_path):
    run_directory = tmp_path / "durable"
    command = [sys.executable, "-c", "import time; time.sleep(0.6)"]
    launched = launch(run_directory, command)
    assert launched.returncode == 0
    launch_record = json.loads((run_directory / "launch.json").read_text())
    assert launch_record["resolved_command"] == command
    assert int(launch_record["wrapper_pid"]) > 0

    duplicate = launch(run_directory, command, resume=True)
    assert duplicate.returncode != 0
    assert "already active" in duplicate.stderr or "lock" in duplicate.stderr

    status = wait_for_status(run_directory, "complete")
    assert status["exit_code"] == 0
    assert (run_directory / "LAUNCHER_COMPLETE").is_file()
    assert Path(launch_record["stdout_path"]).is_file()
    assert Path(launch_record["stderr_path"]).is_file()
    assert not (run_directory / ".launch.lock").exists()

    completed_resume = launch(run_directory, command, resume=True)
    assert completed_resume.returncode != 0
    assert "completed run" in completed_resume.stderr


def test_launcher_records_failure_without_overwriting_logs(tmp_path):
    run_directory = tmp_path / "failure"
    command = [sys.executable, "-c", "raise SystemExit(9)"]
    launched = launch(run_directory, command)
    assert launched.returncode == 0

    status = wait_for_status(run_directory, "failed")
    assert status["exit_code"] == 9
    assert (run_directory / "LAUNCHER_FAILED").is_file()
    assert not (run_directory / "LAUNCHER_COMPLETE").exists()


def test_launcher_rejects_p6_train_logdir_mismatch_before_creating_run(tmp_path):
    run_directory = tmp_path / "guarded"
    other_directory = tmp_path / "wrong"
    result = launch(
        run_directory,
        [
            sys.executable,
            str(PROJECT_ROOT / "p6_train.py"),
            f"logdir={other_directory}",
        ],
    )

    assert result.returncode != 0
    assert "differs from the Hydra logdir" in result.stderr
    assert not run_directory.exists()
