"""Pluggable backends for metadata and for direct (cache-bypassing) reads.

Two implementations ship with the package:

* :class:`ClientMetadataBackend` answers ``info``/``ls`` with the DACS client's
  own ``stat`` and ``list_dir`` calls.  No extra dependency, one credential
  path.  This is the default.
* :class:`AdlfsBackend` talks to Azure Blob Storage directly through
  `adlfs <https://github.com/fsspec/adlfs>`_.  It can serve as the metadata
  backend (``metadata="adlfs"``) and is what fallback reads use.

Anything implementing :class:`MetadataBackend` (and :class:`DirectReader` for
fallback) can be passed to :class:`~dacs_fsspec.DacsFileSystem` instead.
"""

from __future__ import annotations

import stat as stat_module
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from .auth import resolve_auth
from .client import DT_DIR, DacsClientProtocol
from .errors import translate_error
from .paths import BlobPath

__all__ = ["AdlfsBackend", "ClientMetadataBackend", "DirectReader", "MetadataBackend"]

Info = dict[str, Any]

_NO_ACCOUNT_LISTING = "Storage accounts cannot be enumerated; list dacs://<account>/<container>/ instead"


@runtime_checkable
class MetadataBackend(Protocol):
    """Answers namespace questions about blobs.

    Returned dictionaries follow fsspec conventions: ``name`` is the canonical
    ``account/container/key`` path, ``type`` is ``"file"`` or ``"directory"``
    and ``size`` is in bytes (0 for directories).  A missing path must raise
    ``FileNotFoundError``.
    """

    def info(self, path: BlobPath) -> Info: ...

    def ls(self, path: BlobPath) -> list[Info]: ...


@runtime_checkable
class DirectReader(Protocol):
    """Reads blob bytes without going through DACS."""

    def read_range(self, path: BlobPath, start: int, end: int) -> bytes: ...


def _directory(path: BlobPath) -> Info:
    return {"name": path.path, "size": 0, "type": "directory"}


class ClientMetadataBackend:
    """Metadata straight from the DACS client (``get_file_stat_sync`` / ``list_dir_sync``)."""

    def __init__(self, get_client: Callable[[], DacsClientProtocol]) -> None:
        self._get_client = get_client

    def info(self, path: BlobPath) -> Info:
        if path.is_root or path.is_account:
            return _directory(path)
        client = self._get_client()
        try:
            st = client.get_file_stat_sync(path.client_url)
        except Exception as exc:
            raise translate_error(exc, path.path) from exc
        if stat_module.S_ISDIR(int(st.st_mode)):
            return _directory(path)
        return {
            "name": path.path,
            "size": int(st.st_size),
            "type": "file",
            "mtime": float(st.st_mtime),
        }

    def ls(self, path: BlobPath) -> list[Info]:
        if path.is_root:
            raise NotImplementedError(_NO_ACCOUNT_LISTING)
        client = self._get_client()
        try:
            entries = client.list_dir_sync(path.client_url.rstrip("/") + "/")
        except Exception as exc:
            raise translate_error(exc, path.path) from exc
        out: list[Info] = []
        for entry in entries:
            name = str(entry.get("d_name", ""))
            if name in ("", ".", ".."):
                # "." and ".." are artefacts of the client's POSIX emulation.
                continue
            child = path.child(name)
            if entry.get("d_type") == DT_DIR or name.endswith("/"):
                out.append(_directory(child))
            else:
                out.append(
                    {
                        "name": child.path,
                        "size": int(entry.get("d_size", 0) or 0),
                        "type": "file",
                        "mtime": float(entry.get("d_mtime", 0) or 0),
                    }
                )
        out.sort(key=lambda item: str(item["name"]))
        return out


class AdlfsBackend:
    """Direct Azure Blob access through adlfs (``pip install 'dacs-fsspec[adlfs]'``).

    One ``AzureBlobFileSystem`` is created lazily per storage account.  Unless
    ``blob_options`` says otherwise, credentials follow the DACS client's own
    resolution order (see :mod:`dacs_fsspec.auth`).
    """

    def __init__(
        self,
        blob_options: dict[str, Any] | None = None,
        filesystem_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._options = dict(blob_options or {})
        self._factory = filesystem_factory
        self._filesystems: dict[str, Any] = {}

    def _fs(self, path: BlobPath) -> Any:
        fs = self._filesystems.get(path.account)
        if fs is None:
            factory = self._factory
            if factory is None:
                try:
                    from adlfs import AzureBlobFileSystem
                except ImportError as exc:
                    raise ImportError("Direct Azure Blob access needs adlfs: pip install 'dacs-fsspec[adlfs]'") from exc
                factory = AzureBlobFileSystem
            options: dict[str, Any] = {"account_name": path.account_name}
            if "." in path.account:
                options["account_host"] = path.host
            options.update(self._options or resolve_auth().adlfs_options())
            fs = factory(**options)
            self._filesystems[path.account] = fs
        return fs

    def _convert(self, path: BlobPath, raw: dict[str, Any]) -> Info:
        name = str(raw.get("name", "")).strip("/")
        full = f"{path.account}/{name}" if name else path.account
        kind = "directory" if raw.get("type") == "directory" else "file"
        info: Info = {"name": full, "size": int(raw.get("size") or 0), "type": kind}
        modified = raw.get("last_modified")
        if kind == "file" and modified is not None:
            timestamp = getattr(modified, "timestamp", None)
            info["mtime"] = float(timestamp()) if callable(timestamp) else float(modified)
        for extra in ("etag", "content_settings", "creation_time", "metadata"):
            if raw.get(extra) is not None:
                info[extra] = raw[extra]
        return info

    def info(self, path: BlobPath) -> Info:
        if path.is_root or path.is_account:
            return _directory(path)
        return self._convert(path, self._fs(path).info(path.adlfs_path))

    def ls(self, path: BlobPath) -> list[Info]:
        if path.is_root:
            raise NotImplementedError(_NO_ACCOUNT_LISTING)
        listing = self._fs(path).ls(path.adlfs_path, detail=True)
        out = [self._convert(path, raw) for raw in listing]
        out.sort(key=lambda item: str(item["name"]))
        return out

    def read_range(self, path: BlobPath, start: int, end: int) -> bytes:
        if end <= start:
            return b""
        return bytes(self._fs(path).cat_file(path.adlfs_path, start=start, end=end))
