"""The fsspec filesystem for DACS."""

from __future__ import annotations

import contextlib
import ctypes
import datetime
import errno
import importlib.util
import logging
import os
import time
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import isfilelike

from .backends import AdlfsBackend, ClientMetadataBackend, DirectReader, MetadataBackend
from .client import WRITE_MODES, ClientFactory, DacsClientProtocol, get_client, normalize_write_mode
from .errors import DacsFallbackWarning, DacsUnavailableError, is_unavailable, translate_error
from .paths import PROTOCOL, BlobPath, strip_protocol

logger = logging.getLogger("dacs_fsspec")

__all__ = ["DacsFile", "DacsFileSystem"]

T = TypeVar("T")

FALLBACK_ENV = "DACS_FSSPEC_FALLBACK"
_TRUTHY = ("1", "true", "yes", "on")
_WRITE_DISABLED = WRITE_MODES["disabled"]
# Overwriting an existing blob needs the client's "consistent mode", which the
# client only honours for these write modes.
_CONSISTENT_CAPABLE = (WRITE_MODES["sync"], WRITE_MODES["blob_only"])


def _unsupported(operation: str, hint: str = "") -> NotImplementedError:
    message = f"{operation} is not supported: the DACS client has no such operation."
    if hint:
        message = f"{message} {hint}"
    return NotImplementedError(message)


class DacsFileSystem(AbstractFileSystem):
    """Read (and optionally write) Azure blobs through the DACS cache.

    Paths are ``dacs://<account>/<container>/<blob>``; a complete blob hostname
    may stand in for the account (``dacs://acct.blob.core.usgovcloudapi.net/...``).

    Parameters
    ----------
    write_mode:
        ``"disabled"`` (default), ``"sync"``, ``"blob_only"``, ``"cache_only"``
        or ``"cache_blob_async"``; the client's ``WriteMode`` values and enum
        members are accepted too.
    consistent_mode:
        Enable the client's consistent mode, which is what allows an existing
        blob to be overwritten.  Only effective with ``"sync"`` and
        ``"blob_only"``.
    metadata:
        ``"dacs"`` (default) answers ``ls``/``info`` with the client's own
        ``stat``/``list_dir``; ``"adlfs"`` asks Azure Blob Storage directly; or
        pass any :class:`~dacs_fsspec.backends.MetadataBackend`.
    fallback:
        If true, reads bypass DACS and go straight to Azure Blob Storage when
        the client is missing, misconfigured or unreachable, after a single
        :class:`~dacs_fsspec.DacsFallbackWarning`.  ``None`` (default) reads
        the ``DACS_FSSPEC_FALLBACK`` environment variable and is otherwise
        off.  May also be an object implementing both
        :class:`~dacs_fsspec.backends.DirectReader` and ``MetadataBackend``.
    fallback_retry_seconds:
        While in fallback, try DACS again after this many seconds.  ``None``
        or ``0`` stays in fallback for the life of the process.
    blob_options:
        ``storage_options`` for adlfs when it is used for metadata or
        fallback.  Defaults to credentials mirroring the DACS client's.
    client_factory:
        ``factory(write_mode_value, consistent_mode) -> client``.  Defaults to
        constructing ``py_dacs_client.dacs_client.DACSCacheClient``.  Must be
        picklable if the filesystem is sent to worker processes.
    block_size, cache_type:
        Defaults for the buffered file object (read-ahead, 8 MiB).
    """

    protocol = PROTOCOL
    root_marker = ""
    DEFAULT_BLOCK_SIZE = 8 * 2**20
    #: Chunk size for whole-object transfers (``get_file``, ``cat_file``).
    TRANSFER_CHUNK_SIZE = 32 * 2**20

    def __init__(
        self,
        write_mode: Any = "disabled",
        consistent_mode: bool = False,
        metadata: str | MetadataBackend = "dacs",
        fallback: Any = None,
        fallback_retry_seconds: float | None = 60.0,
        blob_options: dict[str, Any] | None = None,
        client_factory: ClientFactory | None = None,
        block_size: int | None = None,
        cache_type: str = "readahead",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.write_mode = normalize_write_mode(write_mode)
        self.consistent_mode = bool(consistent_mode)
        if self.consistent_mode and self.write_mode not in _CONSISTENT_CAPABLE:
            raise ValueError("consistent_mode=True requires write_mode 'sync' or 'blob_only'")
        self.block_size = int(block_size) if block_size else self.DEFAULT_BLOCK_SIZE
        self.cache_type = cache_type
        self.blob_options = dict(blob_options) if blob_options else None
        self._client_factory = client_factory
        self._client: DacsClientProtocol | None = None
        self._client_pid: int | None = None

        if fallback is None:
            fallback = os.environ.get(FALLBACK_ENV, "").strip().lower() in _TRUTHY
        self._direct: Any | None = None
        if isinstance(fallback, bool):
            self.fallback_enabled = fallback
            if fallback:
                self._require_adlfs("fallback=True")
        else:
            if not (isinstance(fallback, DirectReader) and isinstance(fallback, MetadataBackend)):
                raise TypeError("fallback must be a bool or implement read_range(), info() and ls()")
            self.fallback_enabled = True
            self._direct = fallback
        self.fallback_retry_seconds = fallback_retry_seconds or None

        self._metadata: MetadataBackend
        if isinstance(metadata, str):
            if metadata == "dacs":
                self._metadata = ClientMetadataBackend(self._get_client)
            elif metadata == "adlfs":
                if self._direct is None:
                    self._require_adlfs("metadata='adlfs'")
                self._metadata = self._get_direct()
            else:
                raise ValueError(f"metadata must be 'dacs', 'adlfs' or a MetadataBackend, got {metadata!r}")
        elif isinstance(metadata, MetadataBackend):
            self._metadata = metadata
        else:
            raise TypeError("metadata must be 'dacs', 'adlfs' or implement info() and ls()")
        self._metadata_via_client = isinstance(self._metadata, ClientMetadataBackend)

        self._fallback_active = False
        self._fallback_until: float | None = None
        self._fallback_reason: str | None = None
        self._warned = False

    # ------------------------------------------------------------------
    # fsspec plumbing
    # ------------------------------------------------------------------
    @classmethod
    def _strip_protocol(cls, path: Any) -> Any:
        if isinstance(path, list):
            return [cls._strip_protocol(p) for p in path]
        return strip_protocol(os.fspath(path))

    def unstrip_protocol(self, name: str) -> str:
        return BlobPath.parse(name).dacs_url

    @property
    def fsid(self) -> str:
        return PROTOCOL

    # ------------------------------------------------------------------
    # Client and fallback management
    # ------------------------------------------------------------------
    def _get_client(self) -> DacsClientProtocol:
        """This process's DACS client, created on first use (never pickled or forked)."""
        pid = os.getpid()
        if self._client is None or self._client_pid != pid:
            try:
                self._client = get_client(self._client_factory, self.write_mode, self.consistent_mode)
            except DacsUnavailableError:
                raise
            except Exception as exc:
                raise DacsUnavailableError(f"The DACS client factory failed ({type(exc).__name__}: {exc})") from exc
            self._client_pid = pid
        return self._client

    def _get_direct(self) -> Any:
        if self._direct is None:
            self._direct = AdlfsBackend(self.blob_options)
        return self._direct

    @staticmethod
    def _require_adlfs(purpose: str) -> None:
        # Fail when the filesystem is created, not in the middle of an outage.
        if importlib.util.find_spec("adlfs") is None:
            raise ImportError(f"{purpose} needs adlfs: pip install 'dacs-fsspec[adlfs]'")

    def _in_fallback(self) -> bool:
        if not self._fallback_active:
            return False
        # Once the retry window has passed, give DACS another chance.
        return self._fallback_until is None or time.monotonic() < self._fallback_until

    def _enter_fallback(self, exc: BaseException) -> None:
        self._fallback_reason = f"{type(exc).__name__}: {exc}"
        retry = self.fallback_retry_seconds
        self._fallback_until = time.monotonic() + retry if retry else None
        self._fallback_active = True
        if not self._warned:
            self._warned = True
            message = (
                "DACS is unavailable; reading directly from Azure Blob Storage instead "
                f"(no cache acceleration). Reason: {self._fallback_reason}"
            )
            # One user-facing signal; the log record is for services that
            # collect logs but filter warnings.
            logger.info(message)
            warnings.warn(message, DacsFallbackWarning, stacklevel=4)

    def _leave_fallback(self) -> None:
        if self._fallback_active:
            self._fallback_active = False
            self._fallback_until = None
            logger.info("DACS is reachable again; resuming cached reads")

    def _guarded(self, path: str, via_dacs: Callable[[], T], direct: Callable[[], T]) -> T:
        """Run ``via_dacs``; if DACS itself is unavailable and fallback is on, run ``direct``."""
        if self.fallback_enabled and self._in_fallback():
            return direct()
        try:
            result = via_dacs()
        except Exception as exc:
            error = translate_error(exc, path)
            if is_unavailable(error):
                if self.fallback_enabled:
                    self._enter_fallback(error)
                    return direct()
                if isinstance(error, DacsUnavailableError):
                    raise DacsUnavailableError(
                        f"{error} Pass fallback=True (or set {FALLBACK_ENV}=1) to read directly "
                        "from Azure Blob Storage when DACS is unavailable."
                    ) from exc
            if error is exc:
                raise
            raise error from exc
        self._leave_fallback()
        return result

    def status(self) -> dict[str, Any]:
        """Describe how this filesystem is currently serving requests."""
        return {
            "reads": "azure-blob-direct" if self._in_fallback() else "dacs",
            "fallback_enabled": self.fallback_enabled,
            "fallback_reason": self._fallback_reason if self._fallback_active else None,
            "metadata": "dacs" if self._metadata_via_client else type(self._metadata).__name__,
            "write_mode": self.write_mode,
            "consistent_mode": self.consistent_mode,
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def _blob(self, path: str, operation: str = "This operation") -> BlobPath:
        blob = BlobPath.parse(path)
        if not blob.key:
            raise IsADirectoryError(
                errno.EISDIR,
                f"{operation} needs a blob path (dacs://<account>/<container>/<blob>)",
                blob.path,
            )
        return blob

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        blob = BlobPath.parse(path)
        if self._metadata_via_client:
            return self._guarded(blob.path, lambda: self._metadata.info(blob), lambda: self._get_direct().info(blob))
        return self._metadata.info(blob)

    def ls(self, path: str, detail: bool = True, **kwargs: Any) -> Any:
        blob = BlobPath.parse(path)
        try:
            if self._metadata_via_client:
                entries: list[dict[str, Any]] = self._guarded(
                    blob.path, lambda: self._metadata.ls(blob), lambda: self._get_direct().ls(blob)
                )
            else:
                entries = self._metadata.ls(blob)
        except NotADirectoryError:
            entries = [self.info(blob.path)]
        if not entries and not (blob.is_root or blob.is_account):
            # Blob namespaces have no empty directories: an empty listing means
            # the prefix does not exist.  info() raises FileNotFoundError then.
            self.info(blob.path)
        if detail:
            return entries
        return [entry["name"] for entry in entries]

    def exists(self, path: str, **kwargs: Any) -> bool:
        """True if the path exists.  Unlike the fsspec default, errors other
        than "not found" (authentication, connectivity) propagate."""
        try:
            self.info(path, **kwargs)
        except (FileNotFoundError, NotADirectoryError):
            return False
        return True

    def modified(self, path: str) -> datetime.datetime:
        info = self.info(path)
        if info.get("mtime") is None:
            raise NotImplementedError(f"No modification time is available for {path!r}")
        return datetime.datetime.fromtimestamp(float(info["mtime"]), tz=datetime.timezone.utc)

    def created(self, path: str) -> datetime.datetime:
        raise _unsupported("created()", "Use modified() instead.")

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def _pread(self, handle: Any, start: int, end: int) -> bytes:
        """Read ``[start, end)`` from an open client handle."""
        length = end - start
        if length <= 0:
            return b""
        out = bytearray(length)
        got = 0
        while got < length:
            window = (ctypes.c_char * (length - got)).from_buffer(out, got)
            try:
                n = int(handle.read_sync(window, length - got, start + got))
            finally:
                del window  # release the buffer export before touching ``out`` again
            if n < 0:
                raise OSError(errno.EIO, f"DACS read failed with code {n}")
            if n == 0:
                break  # the blob is shorter than its reported size
            got += n
        if got == length:
            return bytes(out)
        return bytes(out[:got])

    def _read_range(self, blob: BlobPath, start: int, end: int) -> bytes:
        if end <= start:
            return b""

        def via_dacs() -> bytes:
            client = self._get_client()
            handle = client.get_file_handle_sync(blob.client_url, os.O_RDONLY)
            try:
                return self._pread(handle, start, end)
            finally:
                client.release_file_handle_sync(handle)

        return self._guarded(blob.path, via_dacs, lambda: bytes(self._get_direct().read_range(blob, start, end)))

    def _file_size(self, path: str) -> int:
        info = self.info(path)
        if info["type"] != "file":
            raise IsADirectoryError(errno.EISDIR, "Is a directory", info["name"])
        return int(info["size"])

    def cat_file(self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any) -> bytes:
        blob = self._blob(path, "cat_file()")
        size = self._file_size(blob.path)
        first = 0 if start is None else start
        last = size if end is None else end
        if first < 0:
            first = max(0, size + first)
        if last < 0:
            last = max(0, size + last)
        return self._read_range(blob, min(first, size), min(last, size))

    def get_file(
        self,
        rpath: str,
        lpath: Any,
        callback: Any = DEFAULT_CALLBACK,
        outfile: Any = None,
        **kwargs: Any,
    ) -> None:
        """Download one blob through the cache, streaming in large chunks."""
        if outfile is None and isfilelike(lpath):
            outfile = lpath
        if outfile is None and self.isdir(rpath):
            os.makedirs(lpath, exist_ok=True)
            return
        with contextlib.ExitStack() as stack:
            source = stack.enter_context(self.open(rpath, "rb", cache_type="none"))
            target = outfile if outfile is not None else stack.enter_context(open(lpath, "wb"))
            callback.set_size(getattr(source, "size", None))
            while True:
                chunk = source.read(self.TRANSFER_CHUNK_SIZE)
                if not chunk:
                    break
                target.write(chunk)
                callback.relative_update(len(chunk))

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: Any = None,
        autocommit: bool = True,
        cache_options: dict[str, Any] | None = None,
        cache_type: str | None = None,
        size: int | None = None,
        **kwargs: Any,
    ) -> DacsFile:
        return DacsFile(
            self,
            path,
            mode=mode,
            block_size=block_size or self.block_size,
            autocommit=autocommit,
            cache_type=cache_type or self.cache_type,
            cache_options=cache_options,
            size=size,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Writes: only what the client can do
    # ------------------------------------------------------------------
    def _check_writable(self, path: str) -> None:
        if self.write_mode == _WRITE_DISABLED:
            raise OSError(
                errno.EROFS,
                "Writes are disabled. Create the filesystem with write_mode='sync', 'blob_only', "
                "'cache_only' or 'cache_blob_async'",
                path,
            )

    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        """No-op: blob "directories" are implied by blob names (containers are not created)."""

    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        """No-op: blob "directories" are implied by blob names (containers are not created)."""

    def rmdir(self, path: str) -> None:
        raise _unsupported("rmdir()")

    def rm_file(self, path: str) -> None:
        raise _unsupported("Deleting blobs", "Delete through Azure Blob Storage (for example adlfs) instead.")

    def _rm(self, path: str) -> None:
        self.rm_file(path)

    def rm(self, path: Any, recursive: bool = False, maxdepth: int | None = None) -> None:
        self.rm_file(path)

    def cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        raise _unsupported("Remote-to-remote copy", "Download with get() and upload with put().")

    def copy(self, path1: Any, path2: Any, *args: Any, **kwargs: Any) -> None:
        self.cp_file(path1, path2)

    def mv(self, path1: Any, path2: Any, *args: Any, **kwargs: Any) -> None:
        raise _unsupported("Moving or renaming blobs")

    def touch(self, path: str, truncate: bool = True, **kwargs: Any) -> None:
        raise _unsupported("touch()")

    def sign(self, path: str, expiration: int = 100, **kwargs: Any) -> str:
        raise _unsupported("Generating signed URLs")


class DacsFile(AbstractBufferedFile):
    """Buffered file over a DACS client handle.

    Read mode keeps one native handle open and serves fsspec's read-ahead
    cache with positional reads.  Write mode streams sequential chunks to a
    native handle; the blob is committed when the file is closed.
    """

    def __init__(
        self,
        fs: DacsFileSystem,
        path: str,
        mode: str = "rb",
        block_size: Any = "default",
        autocommit: bool = True,
        cache_type: str = "readahead",
        cache_options: dict[str, Any] | None = None,
        size: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._handle: Any = None
        self._handle_pid: int | None = None
        self._handle_client: Any = None
        self._keepalive: list[Any] = []
        self.closed = True  # so __del__ is harmless if validation below raises
        self._blob = fs._blob(path, "open()")
        if mode == "ab":
            raise NotImplementedError(
                "Append mode is not supported by the DACS client; write a new blob with mode='wb'"
            )
        if mode not in ("rb", "wb", "xb"):
            raise NotImplementedError(f"File mode {mode!r} is not supported; use 'rb', 'wb' or 'xb'")
        if mode in ("wb", "xb"):
            if not autocommit:
                raise NotImplementedError("Transactions (autocommit=False) are not supported by the DACS client")
            fs._check_writable(self._blob.path)
        elif mode == "rb" and size is None:
            size = fs._file_size(self._blob.path)
        super().__init__(
            fs,
            self._blob.path,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            cache_type=cache_type,
            cache_options=cache_options,
            size=size,
            **kwargs,
        )

    # -- handle management ---------------------------------------------
    def _drop_foreign_handle(self) -> None:
        # A handle opened before fork() belongs to the parent's native client.
        if self._handle is not None and self._handle_pid != os.getpid():
            self._handle = None
            self._handle_client = None

    def _open_handle(self, flags: int) -> Any:
        self._drop_foreign_handle()
        if self._handle is None:
            client = self.fs._get_client()
            self._handle = client.get_file_handle_sync(self._blob.client_url, flags)
            self._handle_client = client
            self._handle_pid = os.getpid()
        return self._handle

    def _release_handle(self) -> None:
        self._drop_foreign_handle()
        handle, client = self._handle, self._handle_client
        self._handle = None
        self._handle_client = None
        if handle is not None and client is not None:
            client.release_file_handle_sync(handle)

    # -- reading ---------------------------------------------------------
    def _fetch_range(self, start: int, end: int) -> bytes:
        end = min(end, self.size)
        if end <= start:
            return b""
        fs: DacsFileSystem = self.fs

        def via_dacs() -> bytes:
            try:
                return fs._pread(self._open_handle(os.O_RDONLY), start, end)
            except Exception:
                # Do not reuse a handle that just failed.
                try:
                    self._release_handle()
                except Exception:
                    logger.debug("Releasing a failed DACS handle raised", exc_info=True)
                raise

        return fs._guarded(
            self._blob.path, via_dacs, lambda: bytes(fs._get_direct().read_range(self._blob, start, end))
        )

    # -- writing ---------------------------------------------------------
    def _initiate_upload(self) -> None:
        fs: DacsFileSystem = self.fs
        overwrite = self.mode == "wb" and fs.consistent_mode
        flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
        try:
            self._open_handle(flags)
        except Exception as exc:
            error = translate_error(exc, self._blob.path)
            if isinstance(error, FileExistsError) and self.mode == "wb":
                raise FileExistsError(
                    errno.EEXIST,
                    "The blob already exists and the DACS client only overwrites in consistent mode "
                    "(consistent_mode=True with write_mode 'sync' or 'blob_only')",
                    self._blob.path,
                ) from exc
            if error is exc:
                raise
            raise error from exc

    def _upload_chunk(self, final: bool = False) -> bool:
        data = self.buffer.getvalue()
        if data:
            # The native layer may reference write buffers until the handle is
            # closed, so every chunk stays alive until then.
            chunk = (ctypes.c_char * len(data)).from_buffer_copy(data)
            self._keepalive.append(chunk)
            handle = self._handle
            written = 0
            try:
                while written < len(data):
                    window = chunk if written == 0 else ctypes.byref(chunk, written)
                    n = int(handle.write_sync(window, len(data) - written, self.offset + written))
                    if n <= 0:
                        raise OSError(errno.EIO, f"DACS write made no progress (returned {n})")
                    written += n
            except Exception as exc:
                error = translate_error(exc, self._blob.path)
                if error is exc:
                    raise
                raise error from exc
        return True

    def commit(self) -> None:
        raise NotImplementedError("Transactions are not supported by the DACS client")

    def discard(self) -> None:
        raise NotImplementedError("Transactions are not supported by the DACS client")

    def close(self) -> None:
        if self.closed:
            return
        try:
            super().close()
        finally:
            self.closed = True
            try:
                # Closing the native handle is what commits a written blob.
                self._release_handle()
            except Exception as exc:
                error = translate_error(exc, self._blob.path)
                if error is exc:
                    raise
                raise error from exc
            finally:
                self._keepalive.clear()
