"""Detached job wrapper used by p6_launcher.py."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
from pathlib import Path

from p6_runtime import atomic_write_json


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--lock-path", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Detached wrapper requires a command")

    run_directory = Path(arguments.run_dir).resolve()
    lock_path = Path(arguments.lock_path).resolve()
    status_path = run_directory / "launcher_status.json"
    disk = shutil.disk_usage(run_directory)
    status = {
        "status": "running",
        "wrapper_pid": os.getpid(),
        "command": command,
        "start_time_utc": _timestamp(),
        "end_time_utc": None,
        "exit_code": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "disk_free_bytes_at_start": disk.free,
    }
    atomic_write_json(status_path, status)
    try:
        completed = subprocess.run(command, check=False)
        status["exit_code"] = int(completed.returncode)
        if completed.returncode == 0:
            status["status"] = "complete"
            marker = run_directory / "LAUNCHER_COMPLETE"
        elif completed.returncode == 75:
            status["status"] = "interrupted"
            marker = run_directory / "LAUNCHER_INTERRUPTED"
        else:
            status["status"] = "failed"
            marker = run_directory / "LAUNCHER_FAILED"
        marker.write_text(f"exit_code={completed.returncode}\n")
        return int(completed.returncode)
    except BaseException as error:
        status["status"] = "wrapper_failed"
        status["wrapper_failure"] = f"{type(error).__name__}: {error}"
        status["exit_code"] = 127
        (run_directory / "LAUNCHER_FAILED").write_text(
            status["wrapper_failure"] + "\n"
        )
        return 127
    finally:
        status["end_time_utc"] = _timestamp()
        # Remove the launcher lock BEFORE publishing the terminal status.  The
        # launcher treats a complete/interrupted/failed status while the lock
        # still exists as an in-flight run; a poller that sees status=complete
        # must also be able to rely on the lock being gone (duplicate-launch
        # safety and the test that asserts the pair).
        try:
            lock_path.unlink()
        except OSError:
            # The status publish is the only way a poller learns the outcome,
            # so a lock-cleanup failure must never suppress it.  FileNotFoundError
            # (already unlinked) and any other OSError are both safe to ignore:
            # the lock lives in the launcher's domain and stale-lock recovery
            # exists for the crash case.
            pass
        atomic_write_json(status_path, status)


if __name__ == "__main__":
    raise SystemExit(main())
