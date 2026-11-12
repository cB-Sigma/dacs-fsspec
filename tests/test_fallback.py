from __future__ import annotations

import errno
import warnings
from collections.abc import Callable

import pytest

import dacs_fsspec.core as core
import fakes
from conftest import PAYLOAD
from dacs_fsspec import DacsFallbackWarning, DacsFileSystem, DacsUnavailableError
from fakes import FakeBlobBackend, FakeFactory

BIG = "dacs://acct/data/big.bin"


def test_fallback_is_off_by_default(fs: DacsFileSystem, factory: FakeFactory) -> None:
    assert fs.fallback_enabled is False
    factory.fail = fakes.unreachable()
    with pytest.raises(ConnectionRefusedError):
        fs.cat_file(BIG)
    with pytest.raises(ConnectionRefusedError):
        fs.info(BIG)


def test_missing_client_error_points_at_the_fallback_option(
    make_fs: Callable[..., DacsFileSystem], factory: FakeFactory
) -> None:
    factory.fail_construct = ImportError("No module named 'py_dacs_client'")
    fs = make_fs()
    with pytest.raises(DacsUnavailableError, match="fallback=True") as info:
        fs.cat_file(BIG)
    assert "DACS_FSSPEC_FALLBACK" in str(info.value)
    assert isinstance(info.value, OSError)


def test_fallback_when_client_cannot_be_constructed(
    make_fs: Callable[..., DacsFileSystem], factory: FakeFactory, direct: FakeBlobBackend
) -> None:
    factory.fail_construct = ImportError("No module named 'py_dacs_client'")
    fs = make_fs(fallback=direct)
    with pytest.warns(DacsFallbackWarning, match="reading directly from Azure Blob Storage") as caught:
        assert fs.cat_file(BIG, 0, 100) == PAYLOAD[:100]
    assert len(caught) == 1
    assert "py_dacs_client" in str(caught[0].message)
    assert fs.status()["reads"] == "azure-blob-direct"
    assert "ImportError" in fs.status()["fallback_reason"]


def test_single_warning_then_silent(
    make_fs: Callable[..., DacsFileSystem], factory: FakeFactory, direct: FakeBlobBackend
) -> None:
    fs = make_fs(fallback=direct)
    factory.fail = fakes.unreachable()
    with pytest.warns(DacsFallbackWarning) as caught:
        assert fs.cat_file(BIG, 0, 10) == PAYLOAD[:10]
    assert len(caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with fs.open(BIG, "rb") as f:
            assert f.read() == PAYLOAD
        assert fs.ls("dacs://acct/data/train", detail=False)
        assert fs.exists(BIG)
        assert fs.glob("dacs://acct/data/*.txt") == ["acct/data/hello.txt"]
    assert direct.calls["read"] >= 2
    assert direct.calls["info"] >= 1


def test_while_in_fallback_the_client_is_left_alone(
    make_fs: Callable[..., DacsFileSystem], factory: FakeFactory, direct: FakeBlobBackend
) -> None:
    fs = make_fs(fallback=direct)
    factory.fail = fakes.unreachable()
    with pytest.warns(DacsFallbackWarning):
        fs.info(BIG)
    attempts = sum(factory.calls.values())
    for _ in range(5):
        fs.cat_file(BIG, 0, 10)
    assert sum(factory.calls.values()) == attempts


def test_fallback_mid_file(
    make_fs: Callable[..., DacsFileSystem], factory: FakeFactory, direct: FakeBlobBackend
) -> None:
    fs = make_fs(fallback=direct, block_size=1000, cache_type="none")
    with fs.open(BIG, "rb") as f:
        assert f.read(1000) == PAYLOAD[:1000]
        assert direct.calls["read"] == 0
        factory.fail = fakes.DACSStorageError("read failed: Connection reset by peer", errno.ECONNRESET)
        with pytest.warns(DacsFallbackWarning):
            assert f.read(1000) == PAYLOAD[1000:2000]
        assert f.read() == PAYLOAD[2000:]
    assert direct.calls["read"] >= 2
    assert factory.open_handles == 0


def test_blob_level_errors_do_not_trigger_fallback(
    make_fs: Callable[..., DacsFileSystem], factory: FakeFactory, direct: FakeBlobBackend
) -> None:
    fs = make_fs(fallback=direct)
    with pytest.raises(FileNotFoundError):
        fs.cat_file("dacs://acct/data/nope")
    factory.fail = fakes.DACSStorageError("open failed: Permission denied", errno.EACCES)
    with pytest.raises(PermissionError):
        fs.cat_file(BIG)
    assert fs.status()["reads"] == "dacs"
    assert sum(direct.calls.values()) == 0


def test_retry_window_returns_to_dacs(
    make_fs: Callable[..., DacsFileSystem],
    factory: FakeFactory,
    direct: FakeBlobBackend,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = [1000.0]
    monkeypatch.setattr(core.time, "monotonic", lambda: now[0])
    fs = make_fs(fallback=direct, fallback_retry_seconds=30)
    factory.fail = fakes.unreachable()
    with pytest.warns(DacsFallbackWarning):
        fs.cat_file(BIG, 0, 10)
    factory.fail = None

    now[0] += 29
    before = factory.calls["read"]
    fs.cat_file(BIG, 0, 10)
    assert factory.calls["read"] == before, "still inside the fallback window"

    now[0] += 2
    with caplog.at_level("INFO", logger="dacs_fsspec"):
        assert fs.cat_file(BIG, 0, 10) == PAYLOAD[:10]
    assert factory.calls["read"] == before + 1
    assert "reachable again" in caplog.text
    assert fs.status() == {
        "reads": "dacs",
        "fallback_enabled": True,
        "fallback_reason": None,
        "metadata": "dacs",
        "write_mode": "write_disabled",
        "consistent_mode": False,
    }


def test_failed_retry_extends_the_window_without_a_second_warning(
    make_fs: Callable[..., DacsFileSystem],
    factory: FakeFactory,
    direct: FakeBlobBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(core.time, "monotonic", lambda: now[0])
    fs = make_fs(fallback=direct, fallback_retry_seconds=10)
    factory.fail = fakes.unreachable()
    with pytest.warns(DacsFallbackWarning):
        fs.cat_file(BIG, 0, 10)
    now[0] = 11
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fs.cat_file(BIG, 0, 10) == PAYLOAD[:10]
    assert fs._fallback_until == 21


def test_sticky_fallback_when_retry_is_disabled(
    make_fs: Callable[..., DacsFileSystem],
    factory: FakeFactory,
    direct: FakeBlobBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(core.time, "monotonic", lambda: now[0])
    fs = make_fs(fallback=direct, fallback_retry_seconds=None)
    factory.fail = fakes.unreachable()
    with pytest.warns(DacsFallbackWarning):
        fs.cat_file(BIG, 0, 10)
    factory.fail = None
    now[0] = 10**9
    before = sum(factory.calls.values())
    fs.cat_file(BIG, 0, 10)
    assert sum(factory.calls.values()) == before


def test_env_var_enables_fallback(make_fs: Callable[..., DacsFileSystem], monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("adlfs")
    assert make_fs().fallback_enabled is False
    DacsFileSystem.clear_instance_cache()
    monkeypatch.setenv("DACS_FSSPEC_FALLBACK", "1")
    assert make_fs().fallback_enabled is True
    assert make_fs(fallback=False).fallback_enabled is False
    monkeypatch.setenv("DACS_FSSPEC_FALLBACK", "no")
    DacsFileSystem.clear_instance_cache()
    assert make_fs().fallback_enabled is False


def test_fallback_true_uses_adlfs_backend(make_fs: Callable[..., DacsFileSystem]) -> None:
    pytest.importorskip("adlfs")
    from dacs_fsspec import AdlfsBackend

    fs = make_fs(fallback=True, blob_options={"anon": True})
    backend = fs._get_direct()
    assert isinstance(backend, AdlfsBackend)
    assert backend._options == {"anon": True}
    assert fs._get_direct() is backend


def test_fallback_object_must_implement_both_protocols(make_fs: Callable[..., DacsFileSystem]) -> None:
    class ReaderOnly:
        def read_range(self, path: object, start: int, end: int) -> bytes:
            return b""

    with pytest.raises(TypeError, match="fallback must be"):
        make_fs(fallback=ReaderOnly())


def test_custom_factory_errors_count_as_unavailable(
    make_fs: Callable[..., DacsFileSystem], factory: FakeFactory, direct: FakeBlobBackend
) -> None:
    factory.fail_construct = RuntimeError("native library exploded")
    fs = make_fs(fallback=direct)
    with pytest.warns(DacsFallbackWarning, match="native library exploded"):
        assert fs.cat_file(BIG, 0, 4) == PAYLOAD[:4]


def test_adlfs_metadata_is_not_guarded_by_fallback(
    make_fs: Callable[..., DacsFileSystem], factory: FakeFactory
) -> None:
    backend = FakeBlobBackend(factory)
    fs = make_fs(metadata=backend, fallback=backend)
    factory.fail = fakes.unreachable()
    assert fs.info(BIG)["size"] == len(PAYLOAD)
    assert fs.status()["reads"] == "dacs", "metadata never touched DACS, so nothing fell back yet"
    with pytest.warns(DacsFallbackWarning):
        assert fs.cat_file(BIG, 0, 3) == PAYLOAD[:3]


def test_adlfs_requirement_is_checked_up_front(
    make_fs: Callable[..., DacsFileSystem], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(core.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match=r"fallback=True needs adlfs"):
        make_fs(fallback=True)
    with pytest.raises(ImportError, match=r"metadata='adlfs' needs adlfs"):
        make_fs(metadata="adlfs")
    monkeypatch.setenv("DACS_FSSPEC_FALLBACK", "1")
    with pytest.raises(ImportError, match=r"dacs-fsspec\[adlfs\]"):
        make_fs()
