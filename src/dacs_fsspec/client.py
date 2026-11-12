"""The seam between this package and the DACS Python client.

Everything the filesystem needs from DACS is expressed by two small structural
protocols that mirror the *synchronous* API of ``py_dacs_client``:

* :class:`DacsClientProtocol` -- the subset of
  ``py_dacs_client.dacs_client.DACSCacheClient`` used here.
* :class:`DacsHandleProtocol` -- the subset of ``DACSCacheFileHandle``.

A real ``DACSCacheClient`` satisfies them as-is, so there is no wrapper layer
in the hot path, and tests can substitute an in-memory fake.  The signature
compatibility test introspects the real classes to detect drift.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from .errors import DacsUnavailableError

logger = logging.getLogger("dacs_fsspec")

__all__ = [
    "WRITE_MODES",
    "ClientFactory",
    "DacsClientProtocol",
    "DacsHandleProtocol",
    "StatProtocol",
    "default_client_factory",
    "get_client",
    "normalize_write_mode",
    "reset_clients",
]

#: Friendly option value -> ``py_dacs_client.WriteMode`` value.
WRITE_MODES: dict[str, str] = {
    "disabled": "write_disabled",
    "sync": "write_sync",
    "blob_only": "write_blob_only",
    "cache_only": "write_cache_only",
    "cache_blob_async": "write_cache_blob_async",
}

#: Directory entry type the client reports for directories (``DT_DIR``).
DT_DIR = 4


class StatProtocol(Protocol):
    """Fields of the client's ``Stat64`` structure that are relied upon."""

    st_mode: int
    st_size: int
    st_mtime: int


class DacsHandleProtocol(Protocol):
    """Subset of ``py_dacs_client.dacs_client.DACSCacheFileHandle``."""

    def read_sync(self, buf: Any, size: int, offset: int) -> int: ...

    def write_sync(self, buf: Any, size: int, offset: int) -> int: ...


@runtime_checkable
class DacsClientProtocol(Protocol):
    """Subset of ``py_dacs_client.dacs_client.DACSCacheClient``.

    ``blob_path`` arguments are ``az://<host>/<container>/<blob>`` URLs.
    Buffers are ctypes objects exposing writable memory.
    """

    def get_file_stat_sync(self, blob_path: str) -> Any: ...

    def list_dir_sync(self, blob_path: str) -> list[dict[str, Any]]: ...

    def get_file_handle_sync(self, blob_path: str, flags: int) -> Any: ...

    def release_file_handle_sync(self, file_handle: Any) -> None: ...


#: ``factory(write_mode, consistent_mode) -> client``.  ``write_mode`` is one of
#: the ``WriteMode`` *values* (for example ``"write_sync"``).
ClientFactory = Callable[[str, bool], DacsClientProtocol]


def normalize_write_mode(mode: Any) -> str:
    """Accept friendly names, ``WriteMode`` values or ``WriteMode`` members."""
    if mode is None or mode is False:
        return WRITE_MODES["disabled"]
    value = getattr(mode, "value", mode)
    if not isinstance(value, str):
        raise ValueError(f"Invalid write_mode {mode!r}")
    key = value.strip().lower()
    if key in WRITE_MODES:
        return WRITE_MODES[key]
    if key in WRITE_MODES.values():
        return key
    choices = ", ".join(repr(k) for k in WRITE_MODES)
    raise ValueError(f"Invalid write_mode {mode!r}; expected one of {choices}")


def default_client_factory(write_mode: str, consistent_mode: bool) -> DacsClientProtocol:
    """Construct the real ``DACSCacheClient`` (imports ``py_dacs_client`` lazily)."""
    try:
        from py_dacs_client import WriteMode
        from py_dacs_client.dacs_client import DACSCacheClient
    except Exception as exc:  # ImportError, or OSError from loading the native library
        raise DacsUnavailableError(
            "The DACS client could not be imported "
            f"({type(exc).__name__}: {exc}). Install it with 'pip install dacs-client' "
            "(Linux x86_64/aarch64, glibc >= 2.35)."
        ) from exc
    try:
        client = DACSCacheClient(write_mode=WriteMode(write_mode), consistent_mode=consistent_mode)
    except Exception as exc:
        raise DacsUnavailableError(f"The DACS client could not be initialised ({type(exc).__name__}: {exc})") from exc
    return client  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Per-process client registry
# ---------------------------------------------------------------------------
# The native client keeps process-global state (it writes one configuration
# file per PID and owns threads and sockets), so clients are created lazily,
# at most once per (process, factory, configuration), and never travel across
# fork() or pickle boundaries.

_Key = tuple[Any, str, bool]
_clients: dict[_Key, DacsClientProtocol] = {}
_owner_pid: int | None = None
_lock = threading.Lock()


def _reset_after_fork() -> None:
    global _lock, _owner_pid
    _clients.clear()
    _owner_pid = None
    _lock = threading.Lock()


if hasattr(os, "register_at_fork"):  # pragma: no branch
    os.register_at_fork(after_in_child=_reset_after_fork)


def get_client(
    factory: ClientFactory | None = None,
    write_mode: str = "write_disabled",
    consistent_mode: bool = False,
) -> DacsClientProtocol:
    """Return this process's client for the given configuration, creating it on first use."""
    global _owner_pid
    resolved: ClientFactory = factory or default_client_factory
    key: _Key = (resolved, write_mode, consistent_mode)
    pid = os.getpid()
    with _lock:
        if _owner_pid != pid:
            _clients.clear()
            _owner_pid = pid
        client = _clients.get(key)
        if client is None:
            if factory is None and any(k[0] is default_client_factory for k in _clients):
                logger.warning(
                    "A second DACS client configuration (write_mode=%s, consistent_mode=%s) is being "
                    "created in this process. The native client keeps one configuration per process, "
                    "so the most recently created one wins.",
                    write_mode,
                    consistent_mode,
                )
            client = resolved(write_mode, consistent_mode)
            _clients[key] = client
        return client


def reset_clients(shutdown: bool = False) -> None:
    """Forget cached clients (optionally calling ``shutdown_sync`` on each)."""
    with _lock:
        clients = list(_clients.values())
        _clients.clear()
    if shutdown:
        for client in clients:
            stop = getattr(client, "shutdown_sync", None)
            if callable(stop):
                try:
                    stop()
                except Exception:  # pragma: no cover - best effort
                    logger.debug("DACS client shutdown failed", exc_info=True)
