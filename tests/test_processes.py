"""Pickling, forking and worker-process behaviour."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import sys
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

import fakes
from conftest import PAYLOAD
from dacs_fsspec import DacsFileSystem, client
from fakes import FakeFactory

FORK_IS_SAFE = sys.platform.startswith("linux")
SHARD = "dacs://acct/data/shard-03.bin"
EXPECTED = bytes([3]) * 1003


def _read_in_worker(fs: DacsFileSystem) -> tuple[int, bytes, int]:
    data = fs.cat_file(SHARD)
    return os.getpid(), data, fs._get_client().pid  # type: ignore[attr-defined]


def _read_file_in_worker(f: object) -> tuple[int, bytes]:
    return os.getpid(), f.read(100)  # type: ignore[attr-defined]


def test_constructing_the_filesystem_does_not_create_a_client(
    factory: FakeFactory, make_fs: Callable[..., DacsFileSystem]
) -> None:
    fs = make_fs(write_mode="sync")
    pickle.dumps(fs)
    assert factory.calls["construct"] == 0
    fs.info("dacs://acct/data/big.bin")
    fs.cat_file("dacs://acct/data/big.bin", 0, 1)
    assert factory.calls["construct"] == 1


def test_pickle_round_trip_preserves_options() -> None:
    fs = DacsFileSystem(
        client_factory=fakes.shared_factory,
        write_mode="sync",
        consistent_mode=True,
        block_size=1234,
        fallback=False,
        skip_instance_cache=True,
    )
    fs.cat_file(SHARD)  # make sure a live client exists and is not dragged along
    payload = pickle.dumps(fs)
    assert b"FakeDacsClient" not in payload
    DacsFileSystem.clear_instance_cache()
    clone = pickle.loads(payload)
    assert isinstance(clone, DacsFileSystem)
    assert clone is not fs
    assert (clone.write_mode, clone.consistent_mode, clone.block_size) == ("write_sync", True, 1234)
    assert clone._client is None
    assert clone.cat_file(SHARD) == EXPECTED


def test_pickled_open_file_reopens_at_the_same_position() -> None:
    fs = DacsFileSystem(client_factory=fakes.shared_factory)
    f = fs.open(SHARD, "rb")
    f.seek(500)
    clone = pickle.loads(pickle.dumps(f))
    assert clone.tell() == 500
    assert clone.read() == EXPECTED[500:]
    assert clone._handle is not f._handle
    clone.close()
    f.close()


def test_write_files_cannot_be_pickled() -> None:
    fs = DacsFileSystem(client_factory=fakes.shared_factory, write_mode="sync")
    with fs.open("dacs://acct/data/pickle-out.bin", "wb") as f, pytest.raises(RuntimeError, match="not supported"):
        pickle.dumps(f)


@pytest.mark.parametrize("method", ["fork", "spawn", "forkserver"])
def test_worker_processes_build_their_own_client(method: str) -> None:
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{method} is not available on this platform")
    if method == "fork" and not FORK_IS_SAFE:
        pytest.skip("fork() from a threaded test process is only exercised on Linux")
    fs = DacsFileSystem(client_factory=fakes.shared_factory)
    assert fs.cat_file(SHARD) == EXPECTED  # the parent already holds a live client
    parent_client = fs._get_client()
    context = multiprocessing.get_context(method)
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        results = list(pool.map(_read_in_worker, [fs] * 4))
    for pid, data, client_pid in results:
        assert data == EXPECTED
        assert pid != os.getpid()
        assert client_pid == pid, "the worker used a client it created itself"
    assert fs._get_client() is parent_client


def test_fork_with_an_open_file() -> None:
    if not FORK_IS_SAFE:
        pytest.skip("fork() from a threaded test process is only exercised on Linux")
    fs = DacsFileSystem(client_factory=fakes.shared_factory)
    f = fs.open(SHARD, "rb", cache_type="none")
    assert f.read(10) == EXPECTED[:10]
    parent_handle = f._handle
    read_end, write_end = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: must not touch the parent's native handle
        status = 1
        try:
            data = f.read(10)
            foreign = f._handle is parent_handle
            ok = data == EXPECTED[10:20] and not foreign and f._handle_pid == os.getpid()
            os.write(write_end, b"ok" if ok else b"bad")
            status = 0
        finally:
            os._exit(status)
    os.close(write_end)
    _, exit_status = os.waitpid(pid, 0)
    with os.fdopen(read_end, "rb") as pipe:
        assert pipe.read() == b"ok"
    assert exit_status == 0
    assert f._handle is parent_handle
    assert f.read(10) == EXPECTED[10:20]
    f.close()


def test_handles_from_another_process_are_dropped_not_closed(
    fs: DacsFileSystem, factory: FakeFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = fs.open("dacs://acct/data/big.bin", "rb", cache_type="none")
    f.read(10)
    assert factory.calls["close"] == 0
    monkeypatch.setattr(os, "getpid", lambda: 99999)
    f.close()
    assert factory.calls["close"] == 0, "a forked child must not release its parent's handle"


def test_filesystem_rebinds_its_client_after_a_pid_change(
    fs: DacsFileSystem, factory: FakeFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = fs._get_client()
    assert fs._get_client() is first
    monkeypatch.setattr(os, "getpid", lambda: 99999)
    second = fs._get_client()
    assert second is not first
    assert factory.calls["construct"] == 2


def test_threads_share_one_client(fs: DacsFileSystem, factory: FakeFactory) -> None:
    def read(offset: int) -> bytes:
        return fs.cat_file("dacs://acct/data/big.bin", offset, offset + 1000)

    offsets = list(range(0, 100_000, 1000))
    with ThreadPoolExecutor(8) as pool:
        chunks = list(pool.map(read, offsets))
    assert b"".join(chunks) == PAYLOAD[:100_000]
    assert factory.calls["construct"] == 1
    assert factory.open_handles == 0


def test_instance_cache_and_registry_interplay(factory: FakeFactory) -> None:
    one = DacsFileSystem(client_factory=factory)
    two = DacsFileSystem(client_factory=factory)
    assert one is two, "fsspec caches instances with identical options"
    other = DacsFileSystem(client_factory=factory, block_size=1)
    assert other is not one
    assert other._get_client() is one._get_client(), "same process + configuration share a client"
    assert client.get_client(factory) is one._get_client()
