from __future__ import annotations

import errno
import io
import os
from collections.abc import Callable
from pathlib import Path

import pytest

import fakes
from conftest import PAYLOAD
from dacs_fsspec import DacsFile, DacsFileSystem
from fakes import FakeFactory

BIG = "dacs://acct/data/big.bin"


def test_read_whole_file(fs: DacsFileSystem, factory: FakeFactory) -> None:
    with fs.open(BIG, "rb") as f:
        assert isinstance(f, DacsFile)
        assert f.size == len(PAYLOAD)
        assert f.read() == PAYLOAD
    assert factory.open_handles == 0


def test_seek_tell_and_partial_reads(fs: DacsFileSystem) -> None:
    with fs.open(BIG, "rb") as f:
        assert f.tell() == 0
        assert f.read(10) == PAYLOAD[:10]
        assert f.tell() == 10
        assert f.seek(1000) == 1000
        assert f.read(5) == PAYLOAD[1000:1005]
        assert f.seek(-20, os.SEEK_END) == len(PAYLOAD) - 20
        assert f.read() == PAYLOAD[-20:]
        assert f.seek(5, os.SEEK_SET) == 5
        assert f.seek(5, os.SEEK_CUR) == 10
        assert f.read(1) == PAYLOAD[10:11]
        assert f.seekable()
        assert f.readable()
        assert not f.writable()
        with pytest.raises(ValueError, match="Seek before start"):
            f.seek(-1)


def test_eof_behaviour(fs: DacsFileSystem) -> None:
    with fs.open(BIG, "rb") as f:
        f.seek(len(PAYLOAD) - 3)
        assert f.read(100) == PAYLOAD[-3:]
        assert f.read(100) == b""
        assert f.read() == b""
        f.seek(len(PAYLOAD) + 500)
        assert f.read(10) == b""


def test_empty_blob(fs: DacsFileSystem, factory: FakeFactory) -> None:
    with fs.open("dacs://acct/data/empty.bin", "rb") as f:
        assert f.size == 0
        assert f.read() == b""
    assert fs.cat_file("dacs://acct/data/empty.bin") == b""
    assert factory.calls["read"] == 0


def test_readinto_and_iteration(fs: DacsFileSystem) -> None:
    with fs.open(BIG, "rb") as f:
        target = bytearray(64)
        assert f.readinto(target) == 64
        assert bytes(target) == PAYLOAD[:64]
    with fs.open("dacs://acct/data/hello.txt", "rb") as f:
        assert list(f) == [b"hello\n", b"world\n"]
    with fs.open("dacs://acct/data/hello.txt", "r", encoding="utf-8") as f:
        assert f.readline() == "hello\n"
        assert f.read() == "world\n"


def test_readahead_batches_native_reads(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    fs = make_fs(block_size=64 * 1024)
    with fs.open(BIG, "rb") as f:
        out = io.BytesIO()
        while chunk := f.read(1000):
            out.write(chunk)
    assert out.getvalue() == PAYLOAD
    # 300 small reads are served by a handful of positional reads on ONE handle.
    assert factory.calls["read"] <= len(PAYLOAD) // (64 * 1024) + 2
    assert factory.calls["open"] == 1
    assert factory.calls["close"] == 1
    for _path, offset, size in factory.reads:
        assert offset + size <= len(PAYLOAD), "never asks the client for bytes past EOF"


def test_default_block_size_and_cache_type(fs: DacsFileSystem, make_fs: Callable[..., DacsFileSystem]) -> None:
    with fs.open(BIG, "rb") as f:
        assert f.blocksize == 8 * 2**20
        assert f.cache.name == "readahead"
    with make_fs(cache_type="none", block_size=1024).open(BIG, "rb") as f:
        assert f.blocksize == 1024
        assert f.cache.name == "none"
    with fs.open(BIG, "rb", block_size=4096, cache_type="bytes") as f:
        assert f.blocksize == 4096
        assert f.cache.name == "bytes"
        assert f.read(10) == PAYLOAD[:10]


def test_explicit_size_skips_the_stat(fs: DacsFileSystem, factory: FakeFactory) -> None:
    with fs.open(BIG, "rb", size=100) as f:
        assert f.read() == PAYLOAD[:100]
    assert factory.calls["stat"] == 0


def test_short_native_reads_are_completed(make_fs: Callable[..., DacsFileSystem], factory: FakeFactory) -> None:
    factory.max_read = 777
    fs = make_fs(block_size=10_000)
    with fs.open(BIG, "rb") as f:
        f.seek(123)
        assert f.read(5000) == PAYLOAD[123:5123]
    assert fs.cat_file(BIG, 0, 3000) == PAYLOAD[:3000]


def test_blob_shorter_than_reported_returns_what_exists(fs: DacsFileSystem, factory: FakeFactory) -> None:
    with fs.open(BIG, "rb", size=len(PAYLOAD) + 1000, cache_type="none") as f:
        f.seek(len(PAYLOAD) - 10)
        assert f.read(500) == PAYLOAD[-10:]


def test_negative_native_return_is_an_error(fs: DacsFileSystem) -> None:
    class Broken:
        def read_sync(self, buf: object, size: int, offset: int) -> int:
            return -1

    with pytest.raises(OSError, match="read failed") as info:
        fs._pread(Broken(), 0, 10)
    assert info.value.errno == errno.EIO
    assert fs._pread(Broken(), 5, 5) == b""


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (None, None, PAYLOAD),
        (0, 10, PAYLOAD[:10]),
        (100, None, PAYLOAD[100:]),
        (None, 100, PAYLOAD[:100]),
        (-50, None, PAYLOAD[-50:]),
        (-50, -10, PAYLOAD[-50:-10]),
        (10, -10, PAYLOAD[10:-10]),
        (len(PAYLOAD) - 5, len(PAYLOAD) + 500, PAYLOAD[-5:]),
        (len(PAYLOAD) + 5, len(PAYLOAD) + 500, b""),
        (50, 50, b""),
        (60, 50, b""),
        (-(10**9), 4, PAYLOAD[:4]),
    ],
)
def test_cat_file_ranges(fs: DacsFileSystem, start: int | None, end: int | None, expected: bytes) -> None:
    assert fs.cat_file(BIG, start=start, end=end) == expected


def test_cat_file_closes_its_handle(fs: DacsFileSystem, factory: FakeFactory) -> None:
    fs.cat_file(BIG, 0, 10)
    assert factory.calls["open"] == factory.calls["close"] == 1
    assert factory.open_handles == 0


def test_cat_ranges_and_cat(fs: DacsFileSystem) -> None:
    out = fs.cat_ranges([BIG, BIG, "acct/data/hello.txt"], [0, 1000, 0], [10, 1010, 5])
    assert out == [PAYLOAD[:10], PAYLOAD[1000:1010], b"hello"]
    many = fs.cat("dacs://acct/data/train/*.csv")
    assert many == {
        "acct/data/train/part-0.csv": b"a,b\n1,2\n3,4\n",
        "acct/data/train/part-1.csv": b"a,b\n5,6\n",
    }
    assert fs.read_bytes(BIG, start=5, end=9) == PAYLOAD[5:9]
    assert fs.head(BIG, 16) == PAYLOAD[:16]
    assert fs.tail(BIG, 16) == PAYLOAD[-16:]
    assert fs.read_text("dacs://acct/data/hello.txt") == "hello\nworld\n"


def test_missing_blob_raises_file_not_found(fs: DacsFileSystem) -> None:
    for action in (
        lambda: fs.open("dacs://acct/data/nope.bin", "rb"),
        lambda: fs.cat_file("dacs://acct/data/nope.bin"),
        lambda: fs.info("dacs://acct/data/nope.bin"),
        lambda: fs.get_file("dacs://acct/data/nope.bin", os.devnull),
    ):
        with pytest.raises(FileNotFoundError) as info:
            action()
        assert info.value.errno == errno.ENOENT
        assert info.value.filename == "acct/data/nope.bin"
        assert type(info.value.__cause__).__name__ == "DACSNotFoundError"


def test_blob_deleted_between_stat_and_open(fs: DacsFileSystem, factory: FakeFactory) -> None:
    f = fs.open(BIG, "rb")
    del factory.store["az://acct.blob.core.windows.net/data/big.bin"]
    with pytest.raises(FileNotFoundError):
        f.read(10)
    f.close()


def test_opening_a_directory_is_an_error(fs: DacsFileSystem) -> None:
    with pytest.raises(IsADirectoryError):
        fs.open("dacs://acct/data/train", "rb")
    with pytest.raises(IsADirectoryError):
        fs.cat_file("dacs://acct/data/train")
    with pytest.raises(IsADirectoryError, match="needs a blob path"):
        fs.open("dacs://acct/data", "rb")
    with pytest.raises(IsADirectoryError):
        fs.cat_file("dacs://acct")


def test_permission_errors_surface_as_permission_error(fs: DacsFileSystem, factory: FakeFactory) -> None:
    factory.fail = fakes.DACSStorageError("open failed: Permission denied", errno.EACCES)
    with pytest.raises(PermissionError):
        fs.cat_file(BIG)


def test_failed_read_releases_the_handle_and_recovers(fs: DacsFileSystem, factory: FakeFactory) -> None:
    with fs.open(BIG, "rb", cache_type="none") as f:
        assert f.read(10) == PAYLOAD[:10]
        factory.fail = fakes.DACSStorageError("read failed: Permission denied", errno.EACCES)
        with pytest.raises(PermissionError):
            f.read(10)
        assert factory.open_handles == 0
        factory.fail = None
        assert f.read(10) == PAYLOAD[10:20]


def test_get_file_and_get(fs: DacsFileSystem, tmp_path: os.PathLike[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DacsFileSystem, "TRANSFER_CHUNK_SIZE", 50_000)
    target = os.path.join(tmp_path, "big.bin")
    seen: list[int] = []

    class Progress:
        def set_size(self, size: int) -> None:
            seen.append(size)

        def relative_update(self, inc: int) -> None:
            seen.append(inc)

    fs.get_file(BIG, target, callback=Progress())
    assert Path(target).read_bytes() == PAYLOAD
    assert seen[0] == len(PAYLOAD)
    assert sum(seen[1:]) == len(PAYLOAD)
    assert len(seen) == 7

    buffer = io.BytesIO()
    fs.get_file(BIG, None, outfile=buffer)
    assert buffer.getvalue() == PAYLOAD
    assert not buffer.closed
    filelike = io.BytesIO()
    fs.get_file(BIG, filelike)
    assert filelike.getvalue() == PAYLOAD

    fs.get_file("dacs://acct/data/train", os.path.join(tmp_path, "made"))
    assert os.path.isdir(os.path.join(tmp_path, "made"))

    fs.get("dacs://acct/data/train", os.path.join(tmp_path, "tree"), recursive=True)
    assert Path(tmp_path, "tree", "nested", "deep.txt").read_bytes() == b"deep"
    assert sorted(os.listdir(os.path.join(tmp_path, "tree"))) == ["nested", "part-0.csv", "part-1.csv"]


def test_file_repr_and_double_close(fs: DacsFileSystem) -> None:
    f = fs.open(BIG, "rb")
    assert "acct/data/big.bin" in repr(f)
    f.close()
    f.close()
    assert f.closed
    with pytest.raises(ValueError, match="closed"):
        f.read(1)
