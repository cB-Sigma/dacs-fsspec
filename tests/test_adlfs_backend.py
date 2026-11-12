from __future__ import annotations

import datetime
import inspect
from typing import Any, ClassVar

import pytest

from dacs_fsspec import AdlfsBackend, BlobPath
from dacs_fsspec.auth import StaticTokenCredential
from dacs_fsspec.backends import DirectReader, MetadataBackend


class StubAzureFS:
    """Records how adlfs would be constructed and answers like adlfs does."""

    instances: ClassVar[list[StubAzureFS]] = []

    def __init__(self, **options: Any) -> None:
        self.options = options
        StubAzureFS.instances.append(self)
        self.when = datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc)

    def info(self, path: str) -> dict[str, Any]:
        if path == "cont/dir":
            return {"name": "cont/dir/", "size": 0, "type": "directory"}
        if path == "cont/dir/a.bin":
            return {"name": path, "size": 5, "type": "file", "last_modified": self.when, "etag": "0x1", "tags": None}
        raise FileNotFoundError(path)

    def ls(self, path: str, detail: bool = True) -> list[dict[str, Any]]:
        assert detail is True
        if path == "":
            return [{"name": "cont", "size": 0, "type": "directory"}]
        return [
            {"name": "cont/dir/sub/", "size": None, "type": "directory"},
            {"name": "cont/dir/a.bin", "size": 5, "type": "file", "last_modified": 12.5},
        ]

    def cat_file(self, path: str, start: int, end: int) -> bytearray:
        return bytearray(b"hello world"[start:end])


@pytest.fixture(autouse=True)
def _reset() -> None:
    StubAzureFS.instances.clear()


def test_backend_implements_both_protocols() -> None:
    backend = AdlfsBackend()
    assert isinstance(backend, MetadataBackend)
    assert isinstance(backend, DirectReader)


def test_info_ls_and_read() -> None:
    backend = AdlfsBackend({"anon": True}, filesystem_factory=StubAzureFS)
    info = backend.info(BlobPath.parse("acct/cont/dir/a.bin"))
    assert info == {
        "name": "acct/cont/dir/a.bin",
        "size": 5,
        "type": "file",
        "mtime": datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc).timestamp(),
        "etag": "0x1",
    }
    assert backend.info(BlobPath.parse("acct/cont/dir")) == {"name": "acct/cont/dir", "size": 0, "type": "directory"}
    assert backend.info(BlobPath.parse("acct"))["type"] == "directory"
    assert backend.info(BlobPath.parse(""))["type"] == "directory"
    assert backend.ls(BlobPath.parse("acct/cont/dir")) == [
        {"name": "acct/cont/dir/a.bin", "size": 5, "type": "file", "mtime": 12.5},
        {"name": "acct/cont/dir/sub", "size": 0, "type": "directory"},
    ]
    assert backend.ls(BlobPath.parse("acct")) == [{"name": "acct/cont", "size": 0, "type": "directory"}]
    with pytest.raises(NotImplementedError, match="cannot be enumerated"):
        backend.ls(BlobPath.parse(""))
    with pytest.raises(FileNotFoundError):
        backend.info(BlobPath.parse("acct/cont/missing"))
    data = backend.read_range(BlobPath.parse("acct/cont/dir/a.bin"), 6, 11)
    assert data == b"world"
    assert type(data) is bytes
    assert backend.read_range(BlobPath.parse("acct/cont/dir/a.bin"), 5, 5) == b""
    assert len(StubAzureFS.instances) == 1, "one adlfs filesystem per storage account"
    assert StubAzureFS.instances[0].options == {"account_name": "acct", "anon": True}


def test_one_filesystem_per_account_and_sovereign_hosts() -> None:
    backend = AdlfsBackend({"anon": True}, filesystem_factory=StubAzureFS)
    backend.ls(BlobPath.parse("one/cont/dir"))
    backend.ls(BlobPath.parse("two.blob.core.usgovcloudapi.net/cont/dir"))
    backend.ls(BlobPath.parse("one/cont/dir"))
    assert [fs.options for fs in StubAzureFS.instances] == [
        {"account_name": "one", "anon": True},
        {"account_name": "two", "account_host": "two.blob.core.usgovcloudapi.net", "anon": True},
    ]


def test_credentials_mirror_the_dacs_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def options() -> dict[str, Any]:
        StubAzureFS.instances.clear()
        AdlfsBackend(filesystem_factory=StubAzureFS).ls(BlobPath.parse("acct/cont/dir"))
        return StubAzureFS.instances[0].options

    assert options() == {"account_name": "acct", "anon": False}
    monkeypatch.setenv("SI_azBlobAccessToken", "tok")
    assert isinstance(options()["credential"], StaticTokenCredential)
    monkeypatch.setenv("USE_SAS_TOKEN", "true")
    monkeypatch.setenv("STORAGE_INTCPT_SAS_TOKEN", "sv=1&sig=2")
    assert options() == {"account_name": "acct", "sas_token": "sv=1&sig=2"}


def test_missing_adlfs_gives_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "adlfs", None)
    with pytest.raises(ImportError, match=r"dacs-fsspec\[adlfs\]"):
        AdlfsBackend().info(BlobPath.parse("acct/cont/blob"))


def test_real_adlfs_accepts_the_options_we_pass() -> None:
    """Drift check against the installed adlfs: every keyword we send must exist."""
    adlfs = pytest.importorskip("adlfs")
    parameters = inspect.signature(adlfs.AzureBlobFileSystem.__init__).parameters
    for keyword in ("account_name", "account_host", "sas_token", "credential", "anon"):
        assert keyword in parameters, keyword
    methods = adlfs.AzureBlobFileSystem
    assert {"start", "end"} <= set(inspect.signature(methods.cat_file).parameters)
    assert "detail" in inspect.signature(methods.ls).parameters


def test_real_adlfs_can_be_constructed_offline() -> None:
    pytest.importorskip("adlfs")
    backend = AdlfsBackend({"anon": True})
    fs = backend._fs(BlobPath.parse("acct.blob.core.usgovcloudapi.net/cont/blob"))
    assert type(fs).__name__ == "AzureBlobFileSystem"
    assert fs.account_name == "acct"
    assert fs.account_host == "acct.blob.core.usgovcloudapi.net"
    assert backend._fs(BlobPath.parse("acct.blob.core.usgovcloudapi.net/other/x")) is fs
