"""Durable, duplicate-safe launcher for P6 production commands."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

from p6_runtime import atomic_write_json


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _timestamp_token() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _validate_p6_command_run_directory(
    command: list[str],
    run_directory: Path,
) -> None:
    if not any(Path(token).name == "p6_train.py" for token in command):
        return
    logdir_values = [
        token.split("=", maxsplit=1)[1]
        for token in command
        if token.startswith("logdir=")
    ]
    if len(logdir_values) != 1:
        raise ValueError(
            "P6 production commands require exactly one explicit logdir= override"
        )
    command_run_directory = Path(logdir_values[0]).expanduser()
    if not command_run_directory.is_absolute():
        command_run_directory = Path(__file__).resolve().parent / command_run_directory
    command_run_directory = command_run_directory.resolve()
    if command_run_directory != run_directory:
        raise ValueError(
            "Launcher --run-dir differs from the Hydra logdir override: "
            f"{run_directory} != {command_run_directory}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Launcher requires a command after --")

    run_directory = Path(arguments.run_dir).expanduser().resolve()
    _validate_p6_command_run_directory(command, run_directory)
    if arguments.resume:
        if not run_directory.is_dir():
            raise FileNotFoundError(f"Resume run directory is missing: {run_directory}")
        if (run_directory / "COMPLETE").exists() or (
            run_directory / "LAUNCHER_COMPLETE"
        ).exists():
            raise FileExistsError("Refusing to resume a completed run")
    else:
        try:
            run_directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to overwrite existing run directory {run_directory}"
            ) from error

    lock_path = run_directory / ".launch.lock"
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        status_path = run_directory / "launcher_status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            pid = int(status.get("wrapper_pid", -1))
            if pid > 0 and _process_alive(pid):
                raise RuntimeError(f"P6 run is already active with PID {pid}") from error
        raise RuntimeError(
            f"Stale or active P6 launcher lock exists: {lock_path}"
        ) from error
    os.close(lock_descriptor)

    token = _timestamp_token()
    stdout_path = run_directory / f"stdout_{token}.log"
    stderr_path = run_directory / f"stderr_{token}.log"
    wrapper_command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("p6_job_wrapper.py")),
        "--run-dir",
        str(run_directory),
        "--lock-path",
        str(lock_path),
        "--",
        *command,
    ]
    try:
        stdout_file = stdout_path.open("xb")
        stderr_file = stderr_path.open("xb")
        process = subprocess.Popen(
            wrapper_command,
            cwd=str(Path(__file__).resolve().parent),
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
            close_fds=True,
        )
        stdout_file.close()
        stderr_file.close()
        atomic_write_json(
            run_directory / "launch.json",
            {
                "wrapper_pid": process.pid,
                "resolved_command": command,
                "wrapper_command": wrapper_command,
                "run_directory": str(run_directory),
                "resume": bool(arguments.resume),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "launch_time_utc": dt.datetime.now(
                    dt.timezone.utc
                ).isoformat(),
            },
        )
    except BaseException:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        raise
    print(
        json.dumps(
            {
                "wrapper_pid": process.pid,
                "run_directory": str(run_directory),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
