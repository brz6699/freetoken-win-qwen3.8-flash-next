"""EXP-ONLY raw-output recorder (independent of FreeToken's own "clean" verdict).

Writes one JSONL line per generation request to a file under a configurable
directory (default ``G:\\freetoken_raw_log``) so the model's RAW emitted tokens can
be inspected later without trusting any self-assessment done inside the engine.

Design constraints:
* MUST NEVER raise into the request path. Every public entry point is wrapped and
  degrades to a silent no-op on any error (bad disk, missing drive, full file, ...).
* Non-fatal and cheap: a single lazily-opened append handle per process, a threading
  lock for in-process threads, and the whole line written in one ``write`` call so
  concurrent workers (multiprocessing spawn) do not tear a record.
* Off by default only when explicitly disabled via ``FREETOKEN_RAW_LOG=0``.

Environment:
* ``FREETOKEN_RAW_LOG_DIR``  target directory (default ``G:\\freetoken_raw_log``)
* ``FREETOKEN_RAW_LOG``      set to ``0``/``false``/``off`` to disable entirely
* ``FREETOKEN_RAW_LOG_FILE`` file basename (default ``freetoken_raw.jsonl``)
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_DEFAULT_DIR = "G:\\freetoken_raw_log"
_DEFAULT_FILE = "freetoken_raw.jsonl"

_lock = threading.Lock()
_handle: Any = None
_disabled = False      # set True if logging is turned off / init failed
_notified = False      # emit at most one stderr note per process
_path: str | None = None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no", ""}


def _resolve_path() -> str | None:
    global _path
    if not _env_flag("FREETOKEN_RAW_LOG", True):
        return None
    directory = os.environ.get("FREETOKEN_RAW_LOG_DIR", _DEFAULT_DIR)
    filename = os.environ.get("FREETOKEN_RAW_LOG_FILE", _DEFAULT_FILE)
    return os.path.join(directory, filename)


def _notify(message: str) -> None:
    global _notified
    if _notified:
        return
    _notified = True
    try:
        print(f"[raw_log] {message}", flush=True)
    except Exception:
        pass


def _open_handle(path: str) -> Any:
    # Ensure the directory exists; open in append mode with utf-8 so CJK in the raw
    # output is human-readable and concurrent multi-process appends share O_APPEND.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "a", encoding="utf-8", newline="\n")


def _sampling_dict(sampling: Any) -> dict[str, Any]:
    """Pull the sampling fields we care about without depending on the dataclass shape."""
    if sampling is None:
        return {}
    out: dict[str, Any] = {}
    for key in ("temperature", "top_k", "top_p", "ignore_eos", "max_tokens"):
        if hasattr(sampling, key):
            out[key] = getattr(sampling, key)
    if hasattr(sampling, "stop_strs"):
        out["stop_strs"] = list(sampling.stop_strs or [])
    if hasattr(sampling, "is_greedy"):
        try:
            out["is_greedy"] = bool(sampling.is_greedy)
        except Exception:
            pass
    return out


def record_generation(
    uid: int,
    *,
    stream: bool,
    raw: str,
    reasoning: str | None = None,
    content: str | None = None,
    finish_reason: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
    matched_stop: str | None = None,
    sampling: Any = None,
    error: str | None = None,
) -> None:
    """Append one raw-output record. Never raises; any failure disables logging for
    the rest of this process and (once) notes it on stderr."""
    global _handle, _disabled, _path
    if _disabled:
        return
    try:
        if _path is None:
            _path = _resolve_path()
            if _path is None:
                _disabled = True
                return
        with _lock:
            if _handle is None:
                _handle = _open_handle(_path)
            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pid": os.getpid(),
                "uid": uid,
                "stream": stream,
                "raw": raw,
                "reasoning": reasoning,
                "content": content,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "matched_stop": matched_stop,
                "sampling": _sampling_dict(sampling),
                "error": error,
            }
            # One write call keeps the record contiguous across concurrent processes.
            _handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            _handle.flush()
    except Exception as exc:  # never break generation on a logging failure
        _disabled = True
        _notify(f"disabled after error ({type(exc).__name__}: {exc}); path={_path}")


def note_startup() -> None:
    """Optional: emit where (if) logging is active so the operator can confirm the
    target path from the server console at boot. Safe to call multiple times."""
    try:
        path = _resolve_path()
        if path is None:
            _notify("disabled via FREETOKEN_RAW_LOG=0")
        else:
            _notify(f"recording raw model output -> {path}")
    except Exception:
        pass
