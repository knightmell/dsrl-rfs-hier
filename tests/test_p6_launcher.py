from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "p6_launcher.py"


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
