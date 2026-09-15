"""Single-instance lock + persisted serve state.

``SingleInstance`` is a flock-held pidfile: an advisory ``LOCK_EX | LOCK_NB`` on an fd we keep
open for the daemon's whole life. flock (not a bare port-bind) avoids TIME_WAIT races and is
released automatically if the daemon dies, so a crashed daemon never wedges its own restart.

``ServeStateStore`` persists ``{model, port, pid, args, starttime, logPath}`` as JSON on every
lifecycle change, so a restarted daemon can re-adopt a still-running serve. ``starttime`` +
``args`` are what make re-adoption PID-reuse-safe and config-exact."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


class AlreadyRunning(RuntimeError):
    """Another daemon holds the single-instance lock."""


class SingleInstance:
    def __init__(self, path: str) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self._lock(fd)
        except OSError as exc:
            os.close(fd)
            raise AlreadyRunning(f"another ft daemon holds {self.path}") from exc
        pid_bytes = f"{os.getpid()}\n".encode()
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, pid_bytes)
        # never truncate below the locked range (Windows byte-range locks die with their bytes)
        os.ftruncate(fd, len(pid_bytes))
        os.fsync(fd)
        self._fd = fd

    @staticmethod
    def _lock(fd: int) -> None:
        try:
            import fcntl
        except ImportError:
            # Windows: no fcntl — lock the pidfile's first byte with msvcrt instead.
            import msvcrt

            if os.lseek(fd, 0, os.SEEK_END) == 0:
                os.write(fd, b" ")  # give the lock a byte to hold
                os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)  # closing drops the flock
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> "SingleInstance":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


@dataclass
class ServeState:
    model: str
    port: int
    pid: int
    args: list[str]
    starttime: int | None = None
    log_path: str | None = None


class ServeStateStore:
    def __init__(self, path: str) -> None:
        self.path = path

    def save(self, state: ServeState) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(asdict(state), fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)  # atomic swap — never a torn/partial state file

    def clear(self) -> None:
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        except OSError:  # pragma: no cover - defensive
            pass

    def load(self) -> ServeState | None:
        try:
            with open(self.path) as fh:
                doc = json.load(fh)
        except (FileNotFoundError, ValueError):
            return None
        except OSError:  # pragma: no cover - defensive
            return None
        try:
            return ServeState(
                model=doc["model"],
                port=int(doc["port"]),
                pid=int(doc["pid"]),
                args=list(doc.get("args") or []),
                starttime=doc.get("starttime"),
                log_path=doc.get("log_path"),
            )
        except (KeyError, TypeError, ValueError):
            return None
