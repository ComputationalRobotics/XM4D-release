from __future__ import annotations

import atexit
import datetime as _dt
import sys
import time
from pathlib import Path

_ACTIVE_LOGGER: dict[str, object] | None = None


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


def install_script_logger(stage: str, output_dir: str | Path) -> Path:
    """Tee stdout/stderr to <output_dir>/logs/<stage>.log and record runtime."""
    out_dir = Path(output_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    start = time.perf_counter()
    state = {"status": 0, "finished": False, "paused_at": None, "paused_seconds": 0.0}

    print(f"[{stage}] log: {log_path}", file=original_stdout, flush=True)
    sys.stdout = _Tee(original_stdout, log_file)
    sys.stderr = _Tee(original_stderr, log_file)

    print(f"stage: {stage}")
    print(f"log_file: {log_path}")
    print(f"start_time: {_dt.datetime.now().isoformat(timespec='seconds')}")
    print(f"command: {' '.join(sys.argv)}")
    print()

    old_excepthook = sys.excepthook

    def _excepthook(exc_type, exc, tb):
        state["status"] = 1
        old_excepthook(exc_type, exc, tb)

    sys.excepthook = _excepthook

    # sys.exit(n) raises SystemExit, which bypasses the excepthook: record its status too.
    old_exit = sys.exit

    def _exit(code=None):
        if code not in (None, 0):
            state["status"] = code if isinstance(code, int) else 1
        old_exit(code)

    sys.exit = _exit

    def _finish() -> None:
        if state["finished"]:
            return
        if state["paused_at"] is not None:
            state["paused_seconds"] += time.perf_counter() - float(state["paused_at"])
            state["paused_at"] = None
        state["finished"] = True
        runtime = time.perf_counter() - start - float(state["paused_seconds"])
        print()
        print(f"end_time: {_dt.datetime.now().isoformat(timespec='seconds')}")
        print(f"runtime_seconds: {runtime:.3f}")
        print(f"paused_seconds: {float(state['paused_seconds']):.3f}")
        print(f"exit_status: {state['status']}")
        print(f"[{stage}] runtime: {runtime:.3f}s")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

    atexit.register(_finish)
    global _ACTIVE_LOGGER
    _ACTIVE_LOGGER = {
        "finish": _finish,
        "state": state,
    }
    return log_path


def pause_script_logger(reason: str = "") -> None:
    if _ACTIVE_LOGGER is None:
        return
    state = _ACTIVE_LOGGER["state"]
    if state["finished"] or state["paused_at"] is not None:
        return
    state["paused_at"] = time.perf_counter()
    print(f"[logger] pause runtime timer{': ' + reason if reason else ''}")


def resume_script_logger(reason: str = "") -> None:
    if _ACTIVE_LOGGER is None:
        return
    state = _ACTIVE_LOGGER["state"]
    if state["finished"] or state["paused_at"] is None:
        return
    paused = time.perf_counter() - float(state["paused_at"])
    state["paused_seconds"] += paused
    state["paused_at"] = None
    print(f"[logger] resume runtime timer after {paused:.3f}s{': ' + reason if reason else ''}")


def finish_script_logger() -> None:
    if _ACTIVE_LOGGER is None:
        return
    _ACTIVE_LOGGER["finish"]()
