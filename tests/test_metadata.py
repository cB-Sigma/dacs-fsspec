from __future__ import annotations

import datetime
import errno
from collections.abc import Callable

import pytest

import fakes
from conftest import PAYLOAD
from dacs_fsspec import DacsFileSystem
from dacs_fsspec.backends import ClientMetadataBackend, MetadataBackend
from fakes import FakeBlobBackend, FakeFactory


def test_info_file_and_directory(fs: DacsFileSystem) -> None:
    assert fs.info("dacs://acct/data/big.bin") == {
        "name": "acct/data/big.bin",
        "size": len(PAYLOAD),
        "type": "file",
        "mtime": 1_700_000_000.0,
    }
    assert fs.info("dacs://acct/data/train") == {"name": "acct/data/train", "size": 0, "type": "directory"}
    assert fs.info("dacs://acct/data/train/")["type"] == "directory"
    assert fs.info("dacs://acct/data")["type"] == "directory"
    assert fs.size("dacs://acct/data/hello.txt") == 12
    assert fs.isfile("dacs://acct/data/hello.txt")
    assert not fs.isfile("dacs://acct/data/train")
    assert fs.isdir("dacs://acct/data/train")
    assert not fs.isdir("dacs://acct/data/hello.txt")
    assert not fs.isdir("dacs://acct/data/missing")


def test_root_and_account_are_directories_without_client_calls(fs: DacsFileSystem, factory: FakeFactory) -> None:
    assert fs.info("")["type"] == "directory"
    assert fs.info("dacs://acct") == {"name": "acct", "size": 0, "type": "directory"}
    assert factory.calls["stat"] == 0
    with pytest.raises(NotImplementedError, match="cannot be enumerated"):
        fs.ls("")


def test_ls(fs: DacsFileSystem) -> None:
    listing = fs.ls("dacs://acct/data", detail=True)
    assert [(e["name"], e["type"], e["size"]) for e in listing] == [
        ("acct/data/big.bin", "file", len(PAYLOAD)),
        ("acct/data/empty.bin", "file", 0),
        ("acct/data/hello.txt", "file", 12),
        ("acct/data/train", "directory", 0),
    ]
    assert fs.ls("dacs://acct/data/train/", detail=False) == [
        "acct/data/train/nested",
        "acct/data/train/part-0.csv",
        "acct/data/train/part-1.csv",
    ]
    assert fs.ls("dacs://acct", detail=False) == ["acct/data", "acct/other"]
    assert fs.listdir("acct/other", detail=False) == ["acct/other/readme.md"]


def test_ls_on_a_file_returns_the_file(fs: DacsFileSystem) -> None:
    assert fs.ls("dacs://acct/data/hello.txt", detail=False) == ["acct/data/hello.txt"]


def test_ls_missing_raises(fs: DacsFileSystem, factory: FakeFactory) -> None:
    with pytest.raises(FileNotFoundError):
        fs.ls("dacs://acct/data/missing")
    # Some backends answer an unknown prefix with an empty listing instead.
    factory.empty_listing_for_missing = True
    with pytest.raises(FileNotFoundError):
        fs.ls("dacs://acct/data/missing")
    assert fs.ls("dacs://nobody", detail=False) == []


def test_exists_only_swallows_not_found(fs: DacsFileSystem, factory: FakeFactory) -> None:
    assert fs.exists("dacs://acct/data/big.bin")
    assert fs.exists("dacs://acct/data/train")
    assert not fs.exists("dacs://acct/data/nope")
    factory.fail = fakes.DACSStorageError("stat failed", errno.ENOTDIR)
    assert not fs.exists("dacs://acct/data/hello.txt/child")
    factory.fail = fakes.DACSStorageError("stat failed: Permission denied", errno.EACCES)
    with pytest.raises(PermissionError):
        fs.exists("dacs://acct/data/big.bin")


def test_glob_find_walk_du(fs: DacsFileSystem) -> None:
    assert fs.glob("dacs://acct/data/*.bin") == ["acct/data/big.bin", "acct/data/empty.bin"]
    assert fs.glob("dacs://acct/data/**/*.txt") == ["acct/data/hello.txt", "acct/data/train/nested/deep.txt"]
    assert fs.glob("dacs://acct/data/train/part-[0-9].csv") == [
        "acct/data/train/part-0.csv",
        "acct/data/train/part-1.csv",
    ]
    assert fs.glob("dacs://acct/data/none-*") == []
    assert fs.find("dacs://acct/data/train") == [
        "acct/data/train/nested/deep.txt",
        "acct/data/train/part-0.csv",
        "acct/data/train/part-1.csv",
    ]
    assert fs.find("dacs://acct/data/train", maxdepth=1, withdirs=True) == [
        "acct/data/train",
        "acct/data/train/nested",
        "acct/data/train/part-0.csv",
        "acct/data/train/part-1.csv",
    ]
    walked = list(fs.walk("dacs://acct/data"))
    assert walked[0] == ("acct/data", ["train"], ["big.bin", "empty.bin", "hello.txt"])
    assert ("acct/data/train/nested", [], ["deep.txt"]) in walked
    assert fs.du("dacs://acct/data/train") == 12 + 8 + 4
    expanded = fs.expand_path("dacs://acct/data/train", recursive=True)
    assert "acct/data/train/nested/deep.txt" in expanded


def test_modified_and_created(fs: DacsFileSystem) -> None:
    assert fs.modified("dacs://acct/data/big.bin") == datetime.datetime.fromtimestamp(
        1_700_000_000, tz=datetime.timezone.utc
    )
    with pytest.raises(NotImplementedError, match="No modification time"):
        fs.modified("dacs://acct/data/train")
    with pytest.raises(NotImplementedError, match="created"):
        fs.created("dacs://acct/data/big.bin")


def test_ukey_and_checksum_are_stable(fs: DacsFileSystem) -> None:
    assert fs.ukey("dacs://acct/data/big.bin") == fs.ukey("acct/data/big.bin")
    assert isinstance(fs.checksum("dacs://acct/data/big.bin"), int)


def test_custom_metadata_backend(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    backend = FakeBlobBackend(factory)
    assert isinstance(backend, MetadataBackend)
    fs = make_fs(metadata=backend)
    assert fs.info("dacs://acct/data/big.bin")["etag"] == "x"
    assert fs.ls("dacs://acct/data/train", detail=False)[0] == "acct/data/train/nested"
    with fs.open("dacs://acct/data/big.bin") as f:
        assert f.read(4) == PAYLOAD[:4]
    # Metadata came from the backend, data from the DACS client.
    assert factory.calls["stat"] == 0
    assert factory.calls["list"] == 0
    assert factory.calls["read"] == 1
    assert backend.calls["read"] == 0
    assert fs.status()["metadata"] == "FakeBlobBackend"


def test_invalid_metadata_option(make_fs: Callable[..., DacsFileSystem]) -> None:
    with pytest.raises(ValueError, match="metadata must be"):
        make_fs(metadata="s3")
    with pytest.raises(TypeError, match="metadata must be"):
        make_fs(metadata=object())


def test_client_backend_skips_dot_entries_and_handles_trailing_slash_names() -> None:
    class Client:
        def list_dir_sync(self, blob_path: str) -> list[dict[str, object]]:
            assert blob_path == "az://acct.blob.core.windows.net/c/dir/"
            return [
                {"d_name": "."},
                {"d_name": ".."},
                {"d_name": ""},
                {"d_name": "sub/", "d_type": 0},
                {"d_name": "file", "d_type": 8, "d_size": 3, "d_mtime": 9},
                {"d_name": "nosize", "d_type": 8},
            ]

    from dacs_fsspec import BlobPath

    backend = ClientMetadataBackend(lambda: Client())  # type: ignore[arg-type, return-value]
    assert backend.ls(BlobPath.parse("acct/c/dir")) == [
        {"name": "acct/c/dir/file", "size": 3, "type": "file", "mtime": 9.0},
        {"name": "acct/c/dir/nosize", "size": 0, "type": "file", "mtime": 0.0},
        {"name": "acct/c/dir/sub", "size": 0, "type": "directory"},
    ]
