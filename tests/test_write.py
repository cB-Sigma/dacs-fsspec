from __future__ import annotations

import errno
import os
from collections.abc import Callable

import pytest

import fakes
from conftest import PAYLOAD
from dacs_fsspec import DacsFileSystem
from fakes import FakeFactory

NEW = "dacs://acct/data/out/new.bin"


def test_writes_are_disabled_by_default(fs: DacsFileSystem, factory: FakeFactory) -> None:
    assert fs.write_mode == "write_disabled"
    with pytest.raises(OSError, match="Writes are disabled") as info:
        fs.open(NEW, "wb")
    assert info.value.errno == errno.EROFS
    with pytest.raises(OSError, match="write_mode="):
        fs.pipe_file(NEW, b"x")
    assert factory.calls["open"] == 0


@pytest.mark.parametrize("mode", ["sync", "blob_only", "cache_only", "cache_blob_async", "write_sync"])
def test_write_new_blob(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory, mode: str) -> None:
    fs = make_fs(write_mode=mode)
    with fs.open(NEW, "wb") as f:
        assert f.writable()
        assert not f.readable()
        assert f.write(b"hello ") == 6
        assert f.write(b"world") == 5
        assert f.tell() == 11
    assert factory.get(NEW) == b"hello world"
    assert factory.clients[0].write_mode.startswith("write_")
    assert factory.clients[0].write_mode != "write_disabled"
    assert factory.open_handles == 0
    assert fs.cat_file(NEW) == b"hello world"


def test_blob_is_committed_only_on_close(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    fs = make_fs(write_mode="sync", block_size=1000)
    f = fs.open(NEW, "wb")
    f.write(PAYLOAD[:5000])
    assert factory.calls["write"] >= 1, "chunks are streamed to the native handle as blocks fill"
    assert not fs.exists(NEW)
    assert len(f._keepalive) >= 1, "written buffers stay alive until the handle is closed"
    f.close()
    assert f._keepalive == []
    assert factory.get(NEW) == PAYLOAD[:5000]


def test_multi_block_write_uses_sequential_offsets(
    make_fs: Callable[..., DacsFileSystem], factory: FakeFactory
) -> None:
    fs = make_fs(write_mode="sync", block_size=4096)
    with fs.open(NEW, "wb") as f:
        for start in range(0, len(PAYLOAD), 1500):
            f.write(PAYLOAD[start : start + 1500])
    assert factory.get(NEW) == PAYLOAD
    assert factory.calls["open"] == 1


def test_partial_native_writes_are_completed(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    factory.max_write = 333
    fs = make_fs(write_mode="sync")
    fs.pipe_file(NEW, PAYLOAD[:2000])
    assert factory.get(NEW) == PAYLOAD[:2000]


def test_write_without_progress_is_an_error(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    factory.max_write = 0
    fs = make_fs(write_mode="sync")
    with pytest.raises(OSError, match="no progress"):
        fs.pipe_file(NEW, b"data")
    assert factory.open_handles == 0


def test_empty_write_creates_an_empty_blob(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    fs = make_fs(write_mode="sync")
    with fs.open(NEW, "wb"):
        pass
    assert factory.get(NEW) == b""


def test_overwrite_needs_consistent_mode(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    fs = make_fs(write_mode="sync")
    with pytest.raises(FileExistsError, match="consistent_mode=True") as info:
        fs.pipe_file("dacs://acct/data/hello.txt", b"new")
    assert info.value.errno == errno.EEXIST
    assert factory.get("acct/data/hello.txt") == b"hello\nworld\n"

    consistent = make_fs(write_mode="sync", consistent_mode=True)
    consistent.pipe_file("dacs://acct/data/hello.txt", b"new")
    assert factory.get("acct/data/hello.txt") == b"new"
    assert factory.clients[-1].consistent_mode is True


def test_exclusive_create(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    fs = make_fs(write_mode="blob_only", consistent_mode=True)
    with pytest.raises(FileExistsError):
        fs.pipe_file("dacs://acct/data/hello.txt", b"new", mode="create")
    with fs.open(NEW, "xb") as f:
        f.write(b"created")
    assert factory.get(NEW) == b"created"


def test_consistent_mode_requires_a_capable_write_mode(make_fs: Callable[..., DacsFileSystem]) -> None:
    for mode in ("disabled", "cache_only", "cache_blob_async"):
        with pytest.raises(ValueError, match="consistent_mode=True requires"):
            make_fs(write_mode=mode, consistent_mode=True)


def test_append_and_transactions_are_unsupported(make_fs: Callable[..., DacsFileSystem]) -> None:
    fs = make_fs(write_mode="sync")
    with pytest.raises(NotImplementedError, match="Append mode"):
        fs.open(NEW, "ab")
    with pytest.raises(NotImplementedError, match="autocommit"):
        fs.open(NEW, "wb", autocommit=False)
    with pytest.raises(NotImplementedError, match="File mode"):
        fs.open(NEW, "r+b")
    with fs.open(NEW, "wb") as f:
        with pytest.raises(NotImplementedError, match="Transactions"):
            f.commit()
        with pytest.raises(NotImplementedError, match="Transactions"):
            f.discard()


def test_put_and_upload(
    make_fs: Callable[..., DacsFileSystem], factory: FakeFactory, tmp_path: os.PathLike[str]
) -> None:
    fs = make_fs(write_mode="sync")
    local = os.path.join(tmp_path, "local.bin")
    with open(local, "wb") as handle:
        handle.write(PAYLOAD)
    fs.put_file(local, NEW)
    assert factory.get(NEW) == PAYLOAD
    fs.put(local, "dacs://acct/data/out/second.bin")
    assert factory.get("acct/data/out/second.bin") == PAYLOAD
    fs.write_text("dacs://acct/data/out/note.txt", "héllo", encoding="utf-8")
    assert factory.get("acct/data/out/note.txt") == "héllo".encode()


def test_errors_while_writing_are_translated(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    fs = make_fs(write_mode="sync", block_size=10)
    f = fs.open(NEW, "wb")
    f.write(b"0123456789abcdef")
    factory.fail = fakes.DACSStorageError("write failed: Disk quota exceeded", errno.EDQUOT)
    with pytest.raises(OSError, match="quota") as info:
        f.write(b"0123456789abcdef")
    assert info.value.errno == errno.EDQUOT
    factory.fail = None
    f.close()


def test_open_failure_for_write_is_translated(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    fs = make_fs(write_mode="sync")
    factory.fail = fakes.DACSStorageError("open failed: Permission denied", errno.EACCES)
    with pytest.raises(PermissionError):
        fs.pipe_file(NEW, b"x")


def test_close_failure_surfaces(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    fs = make_fs(write_mode="sync")
    f = fs.open(NEW, "wb")
    f.write(b"data")
    factory.fail_on_close = fakes.DACSStorageError("close failed: Input/output error", errno.EIO)
    with pytest.raises(OSError, match="close failed"):
        f.close()
    assert f.closed
    assert f._keepalive == []


def test_writes_never_fall_back(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    direct = fakes.FakeBlobBackend(factory)
    fs = make_fs(write_mode="sync", fallback=direct)
    factory.fail = fakes.unreachable()
    with pytest.raises(ConnectionRefusedError):
        fs.pipe_file(NEW, b"x")
    assert "az://acct.blob.core.windows.net/data/out/new.bin" not in factory.store


def test_namespace_mutations_are_unsupported(make_fs: Callable[..., DacsFileSystem]) -> None:
    fs = make_fs(write_mode="sync")
    blob = "dacs://acct/data/hello.txt"
    for action in (
        lambda: fs.rm(blob),
        lambda: fs.rm_file(blob),
        lambda: fs._rm(blob),
        lambda: fs.rm("dacs://acct/data/train", recursive=True),
        lambda: fs.rmdir("dacs://acct/data/train"),
        lambda: fs.mv(blob, "dacs://acct/data/moved.txt"),
        lambda: fs.cp_file(blob, "dacs://acct/data/copy.txt"),
        lambda: fs.copy(blob, "dacs://acct/data/copy.txt"),
        lambda: fs.touch("dacs://acct/data/touched"),
        lambda: fs.sign(blob),
    ):
        with pytest.raises(NotImplementedError, match="not supported"):
            action()
    assert fs.exists(blob)


def test_mkdir_is_a_noop(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    fs = make_fs(write_mode="sync")
    before = dict(factory.store)
    fs.mkdir("dacs://acct/data/newdir")
    fs.makedirs("dacs://acct/data/a/b/c", exist_ok=True)
    assert factory.store == before
    assert sum(factory.calls.values()) == 0
