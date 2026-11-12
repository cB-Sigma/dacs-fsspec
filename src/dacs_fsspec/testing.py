"""In-memory stand-ins for DACS, for tests that must run without a cluster.

:class:`FakeDacs` is a ``client_factory`` backed by a dictionary.  The clients
it creates implement the same synchronous, ctypes-buffer based surface as
``py_dacs_client.dacs_client.DACSCacheClient`` (the subset declared in
:mod:`dacs_fsspec.client`) and fail the way the real client fails: with
exceptions that are *not* ``OSError`` subclasses and carry an ``errno_code``.

    >>> from dacs_fsspec import DacsFileSystem
    >>> from dacs_fsspec.testing import FakeDacs
    >>> fake = FakeDacs({"dacs://acct/data/a.txt": b"hello"})
    >>> DacsFileSystem(client_factory=fake).cat_file("dacs://acct/data/a.txt")
    b'hello'

It imitates behaviour that was read from the client's source; it is not a
simulator of the cache and says nothing about performance.
"""

from __future__ import annotations

import collections
import ctypes
import errno
import os
import stat
from typing import Any

from .paths import BlobPath

__all__ = [
    "DACSError",
    "DACSExistsError",
    "DACSIsDirectoryError",
    "DACSNetworkError",
    "DACSNotDirectoryError",
    "DACSNotFoundError",
    "DACSReadOnlyError",
    "DACSStorageError",
    "FakeBlobBackend",
    "FakeDacs",
    "FakeDacsClient",
    "FakeHandle",
    "OperationNotSupportedError",
    "not_found",
    "unreachable",
]


# -- exceptions shaped like py_dacs_client.dacs_exceptions --------------------
class DACSError(Exception):
    pass


class OperationNotSupportedError(DACSError):
    pass


class DACSStorageError(DACSError):
    def __init__(self, message: str, errno_code: int | None = None, path: str | None = None) -> None:
        self.message = message
        self.errno_code = errno_code
        self.path = path
        super().__init__(f"{message} | errno: {errno_code}")


class DACSNotFoundError(DACSStorageError):
    pass


class DACSExistsError(DACSStorageError):
    pass


class DACSNetworkError(DACSStorageError):
    pass


class DACSReadOnlyError(DACSStorageError):
    pass


class DACSNotDirectoryError(DACSStorageError):
    pass


class DACSIsDirectoryError(DACSStorageError):
    pass


def not_found(path: str) -> DACSNotFoundError:
    return DACSNotFoundError("stat failed: No such file or directory", errno.ENOENT, path)


def unreachable() -> DACSNetworkError:
    return DACSNetworkError("open failed: Connection refused", errno.ECONNREFUSED)


class FakeStat:
    def __init__(self, mode: int, size: int, mtime: int) -> None:
        self.st_mode = mode
        self.st_size = size
        self.st_mtime = mtime
        self.st_atime = mtime
        self.st_ctime = mtime
        self.st_blksize = 4096


class FakeHandle:
    def __init__(self, client: FakeDacsClient, url: str, flags: int) -> None:
        self.client = client
        self.pathname = url
        self.flags = flags
        self.fd: int | None = 3
        self.writes = bytearray()
        self.writing = bool(flags & os.O_WRONLY)

    def read_sync(self, buf: Any, size: int, offset: int) -> int:
        factory = self.client.factory
        factory.calls["read"] += 1
        factory.reads.append((self.pathname, offset, size))
        factory.maybe_fail()
        if self.fd is None:
            raise ValueError("File descriptor is None")
        data = factory.store[self.pathname][offset : offset + size]
        if factory.max_read is not None:
            data = data[: factory.max_read]
        ctypes.memmove(buf, data, len(data))
        return len(data)

    def write_sync(self, buf: Any, size: int, offset: int) -> int:
        factory = self.client.factory
        factory.calls["write"] += 1
        factory.maybe_fail()
        if factory.max_write is not None:
            size = min(size, factory.max_write)
        data = ctypes.string_at(buf, size)
        if offset > len(self.writes):
            self.writes.extend(b"\0" * (offset - len(self.writes)))
        self.writes[offset : offset + size] = data
        return size


class FakeDacsClient:
    def __init__(self, factory: FakeDacs, write_mode: str, consistent_mode: bool) -> None:
        self.factory = factory
        self.write_mode = write_mode
        self.consistent_mode = consistent_mode
        self.pid = os.getpid()
        self.open_handles: list[FakeHandle] = []
        self.shutdowns = 0

    # -- helpers -------------------------------------------------------
    def _is_dir(self, url: str) -> bool:
        prefix = url.rstrip("/") + "/"
        return any(key.startswith(prefix) for key in self.factory.store)

    # -- DACSCacheClient surface -----------------------------------------
    def get_file_stat_sync(self, blob_path: str) -> FakeStat:
        factory = self.factory
        factory.calls["stat"] += 1
        factory.maybe_fail()
        assert blob_path.startswith("az://"), blob_path
        if blob_path in factory.store:
            return FakeStat(stat.S_IFREG | 0o644, len(factory.store[blob_path]), factory.mtime)
        if self._is_dir(blob_path):
            return FakeStat(stat.S_IFDIR | 0o755, 0, 0)
        raise not_found(blob_path)

    def get_file_size_sync(self, blob_path: str) -> int:
        return self.get_file_stat_sync(blob_path).st_size

    def list_dir_sync(self, blob_path: str) -> list[dict[str, Any]]:
        factory = self.factory
        factory.calls["list"] += 1
        factory.maybe_fail()
        url = blob_path.rstrip("/")
        if url in factory.store:
            raise DACSNotDirectoryError("opendir failed: Not a directory", errno.ENOTDIR, url)
        prefix = url + "/"
        children: dict[str, dict[str, Any]] = {}
        for key, data in factory.store.items():
            if not key.startswith(prefix):
                continue
            name, _, rest = key[len(prefix) :].partition("/")
            if rest:
                children[name] = {"d_name": name, "d_type": 4, "d_size": 0, "d_mtime": 0}
            else:
                children[name] = {"d_name": name, "d_type": 8, "d_size": len(data), "d_mtime": factory.mtime}
        if not children and not factory.empty_listing_for_missing:
            raise DACSNotFoundError("opendir failed: No such file or directory", errno.ENOENT, url)
        dots = [{"d_name": ".", "d_type": 4, "d_size": 0, "d_mtime": 0}, {"d_name": "..", "d_type": 4}]
        return dots + list(children.values())

    def get_file_handle_sync(self, blob_path: str, flags: int) -> FakeHandle:
        factory = self.factory
        factory.calls["open"] += 1
        factory.maybe_fail()
        exists = blob_path in factory.store
        if flags & os.O_WRONLY:
            if self.write_mode == "write_disabled":
                raise DACSReadOnlyError("open failed: Read-only file system", errno.EROFS, blob_path)
            if exists and flags & os.O_EXCL:
                raise DACSExistsError("open failed: File exists", errno.EEXIST, blob_path)
            if exists and not self.consistent_mode:
                raise OperationNotSupportedError("overwrite requires consistent mode")
        elif not exists:
            if self._is_dir(blob_path):
                raise DACSIsDirectoryError("open failed: Is a directory", errno.EISDIR, blob_path)
            raise not_found(blob_path)
        handle = FakeHandle(self, blob_path, flags)
        self.open_handles.append(handle)
        return handle

    def release_file_handle_sync(self, file_handle: FakeHandle) -> None:
        factory = self.factory
        factory.calls["close"] += 1
        self.open_handles.remove(file_handle)
        file_handle.fd = None
        if factory.fail_on_close is not None:
            raise factory.fail_on_close
        if file_handle.writing:
            factory.store[file_handle.pathname] = bytes(file_handle.writes)

    def shutdown_sync(self) -> None:
        self.shutdowns += 1


class FakeDacs:
    """An in-memory "cluster": pass it as ``client_factory``.

    ``blobs`` maps ``dacs://`` URLs (or ``account/container/key`` paths) to
    bytes.  Attributes steer and observe the fake:

    * ``calls`` / ``reads`` -- counters and a log of native reads, to assert
      how much I/O a code path performs;
    * ``fail`` -- an exception raised by every client call (for example
      :func:`unreachable`), to simulate an outage;
    * ``fail_construct`` / ``fail_on_close`` -- fail client creation or commit;
    * ``max_read`` / ``max_write`` -- force short native reads and writes.

    Instances are not picklable by value; for multi-process tests expose one
    through a module-level function and pass that function instead.
    """

    def __init__(self, blobs: dict[str, bytes] | None = None) -> None:
        self.store: dict[str, bytes] = {}
        self.clients: list[FakeDacsClient] = []
        self.calls: collections.Counter[str] = collections.Counter()
        self.reads: list[Any] = []
        self.fail: BaseException | None = None
        self.fail_on_close: BaseException | None = None
        self.fail_construct: BaseException | None = None
        self.max_read: int | None = None
        self.max_write: int | None = None
        self.empty_listing_for_missing = False
        self.mtime = 1_700_000_000
        for path, data in (blobs or {}).items():
            self.put(path, data)

    def put(self, path: str, data: bytes) -> None:
        self.store[BlobPath.parse(path).client_url] = data

    def get(self, path: str) -> bytes:
        return self.store[BlobPath.parse(path).client_url]

    def maybe_fail(self) -> None:
        if self.fail is not None:
            raise self.fail

    @property
    def open_handles(self) -> int:
        return sum(len(client.open_handles) for client in self.clients)

    def __call__(self, write_mode: str, consistent_mode: bool) -> FakeDacsClient:
        self.calls["construct"] += 1
        if self.fail_construct is not None:
            raise self.fail_construct
        client = FakeDacsClient(self, write_mode, consistent_mode)
        self.clients.append(client)
        return client


class FakeBlobBackend:
    """Direct "Azure Blob" access over the same store (MetadataBackend + DirectReader)."""

    def __init__(self, factory: FakeDacs) -> None:
        self.factory = factory
        self.calls: collections.Counter[str] = collections.Counter()

    def _is_dir(self, url: str) -> bool:
        prefix = url.rstrip("/") + "/"
        return any(key.startswith(prefix) for key in self.factory.store)

    def info(self, path: BlobPath) -> dict[str, Any]:
        self.calls["info"] += 1
        if path.is_root or path.is_account:
            return {"name": path.path, "size": 0, "type": "directory"}
        url = path.client_url
        if url in self.factory.store:
            return {"name": path.path, "size": len(self.factory.store[url]), "type": "file", "mtime": 1.0, "etag": "x"}
        if self._is_dir(url):
            return {"name": path.path, "size": 0, "type": "directory"}
        raise FileNotFoundError(path.path)

    def ls(self, path: BlobPath) -> list[dict[str, Any]]:
        self.calls["ls"] += 1
        prefix = path.client_url.rstrip("/") + "/"
        names = sorted({key[len(prefix) :].split("/", 1)[0] for key in self.factory.store if key.startswith(prefix)})
        if not names:
            raise FileNotFoundError(path.path)
        return [self.info(path.child(name)) for name in names]

    def read_range(self, path: BlobPath, start: int, end: int) -> bytes:
        self.calls["read"] += 1
        return self.factory.store[path.client_url][start:end]
