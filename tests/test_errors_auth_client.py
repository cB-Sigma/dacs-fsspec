from __future__ import annotations

import asyncio
import errno
import os

import pytest

import fakes
from dacs_fsspec import DacsUnavailableError, client
from dacs_fsspec.auth import StaticTokenCredential, resolve_auth
from dacs_fsspec.errors import errno_of, is_unavailable, translate_error


# -- errors -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (errno.ENOENT, FileNotFoundError),
        (errno.EACCES, PermissionError),
        (errno.EPERM, PermissionError),
        (errno.EEXIST, FileExistsError),
        (errno.EISDIR, IsADirectoryError),
        (errno.ENOTDIR, NotADirectoryError),
        (errno.ETIMEDOUT, TimeoutError),
        (errno.ECONNREFUSED, ConnectionRefusedError),
        (errno.EROFS, OSError),
        (errno.EIO, OSError),
    ],
)
def test_translate_maps_errno_to_builtin(code: int, expected: type) -> None:
    original = fakes.DACSStorageError("stat failed", code)
    translated = translate_error(original, "acct/c/k")
    assert isinstance(translated, expected)
    assert isinstance(translated, OSError)
    assert translated.errno == code
    assert translated.filename == "acct/c/k"
    assert "stat failed" in str(translated)


def test_translate_passthrough_and_unsupported() -> None:
    builtin = FileNotFoundError("x")
    assert translate_error(builtin) is builtin
    not_impl = NotImplementedError("x")
    assert translate_error(not_impl) is not_impl
    plain = RuntimeError("boom")
    assert translate_error(plain) is plain
    unavailable = DacsUnavailableError("no client")
    assert translate_error(unavailable) is unavailable
    assert isinstance(translate_error(fakes.OperationNotSupportedError("glob")), NotImplementedError)
    assert isinstance(translate_error(fakes.DACSStorageError("x", errno.ENOTSUP)), NotImplementedError)


def test_errno_of() -> None:
    assert errno_of(fakes.DACSStorageError("x", errno.ENOENT)) == errno.ENOENT
    assert errno_of(OSError(errno.EIO, "io")) == errno.EIO
    assert errno_of(fakes.DACSStorageError("x", None)) is None
    assert errno_of(ValueError("x")) is None


def test_is_unavailable() -> None:
    assert is_unavailable(DacsUnavailableError("x"))
    assert is_unavailable(fakes.unreachable())
    assert is_unavailable(fakes.DACSStorageError("x", errno.ETIMEDOUT))
    assert is_unavailable(fakes.DACSStorageError("x", errno.EIO))
    assert is_unavailable(ConnectionResetError("x"))
    assert is_unavailable(TimeoutError("x"))
    assert not is_unavailable(fakes.not_found("p"))
    assert not is_unavailable(FileNotFoundError("x"))
    assert not is_unavailable(PermissionError("x"))
    assert not is_unavailable(fakes.DACSStorageError("x", errno.EACCES))
    assert not is_unavailable(ValueError("x"))


# -- auth -----------------------------------------------------------------------
def test_auth_order_matches_the_client() -> None:
    everything = {
        "USE_SAS_TOKEN": "true",
        "STORAGE_INTCPT_SAS_TOKEN": " sv=1&sig=abc \n",
        "SI_azBlobAccessToken": "tok",
        "AZURE_CLIENT_ID": "cid",
        "MSI_ENDPOINT": "http://msi",
        "MSI_SECRET": "s",
    }
    sas = resolve_auth(everything)
    assert sas.mode == "sas_token"
    assert sas.adlfs_options() == {"sas_token": "sv=1&sig=abc"}
    assert "sig=abc" not in repr(sas)

    # The SAS token is ignored unless USE_SAS_TOKEN is exactly "true".
    without_flag = dict(everything, USE_SAS_TOKEN="1")
    token = resolve_auth(without_flag)
    assert token.mode == "access_token"
    assert isinstance(token.adlfs_options()["credential"], StaticTokenCredential)

    msi = resolve_auth({k: v for k, v in everything.items() if k.startswith(("AZURE", "MSI"))})
    assert msi.mode == "managed_identity"
    assert msi.adlfs_options() == {"anon": False}

    # All three MSI variables are required, otherwise the Azure Identity SDK is used.
    partial = resolve_auth({"AZURE_CLIENT_ID": "cid"})
    assert partial.mode == "azure_identity"
    assert partial.adlfs_options() == {"anon": False}


def test_auth_reads_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_auth().mode == "azure_identity"
    monkeypatch.setenv("SI_azBlobAccessToken", "tok")
    assert resolve_auth().mode == "access_token"


def test_static_token_credential() -> None:
    pytest.importorskip("azure.core.credentials")

    async def run() -> None:
        async with StaticTokenCredential("tok") as credential:
            token = await credential.get_token("https://storage.azure.com/.default")
            assert token.token == "tok"
            assert token.expires_on > 0
            await credential.close()

    asyncio.run(run())


# -- client registry ------------------------------------------------------------
@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (None, "write_disabled"),
        (False, "write_disabled"),
        ("disabled", "write_disabled"),
        ("sync", "write_sync"),
        ("SYNC", "write_sync"),
        ("write_sync", "write_sync"),
        ("blob_only", "write_blob_only"),
        ("cache_only", "write_cache_only"),
        ("cache_blob_async", "write_cache_blob_async"),
    ],
)
def test_normalize_write_mode(given: object, expected: str) -> None:
    assert client.normalize_write_mode(given) == expected


def test_normalize_write_mode_accepts_enum_members_and_rejects_junk() -> None:
    class Member:
        value = "write_blob_only"

    assert client.normalize_write_mode(Member()) == "write_blob_only"
    with pytest.raises(ValueError, match="expected one of"):
        client.normalize_write_mode("append")
    with pytest.raises(ValueError, match="Invalid write_mode"):
        client.normalize_write_mode(3)


def test_get_client_is_cached_per_configuration() -> None:
    factory = fakes.FakeFactory()
    first = client.get_client(factory, "write_disabled", False)
    assert client.get_client(factory, "write_disabled", False) is first
    other = client.get_client(factory, "write_sync", True)
    assert other is not first
    assert factory.calls["construct"] == 2
    client.reset_clients(shutdown=True)
    assert first.shutdowns == 1  # type: ignore[attr-defined]
    assert client.get_client(factory, "write_disabled", False) is not first


def test_registry_is_dropped_when_the_pid_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = fakes.FakeFactory()
    parent = client.get_client(factory)
    monkeypatch.setattr(os, "getpid", lambda: 4242)
    assert client.get_client(factory) is not parent


def test_after_fork_hook_clears_the_registry() -> None:
    factory = fakes.FakeFactory()
    before = client.get_client(factory)
    client._reset_after_fork()
    assert client.get_client(factory) is not before


def test_default_factory_reports_a_missing_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "py_dacs_client", None)
    with pytest.raises(DacsUnavailableError, match="pip install dacs-client"):
        client.default_client_factory("write_disabled", False)


def test_default_factory_wraps_constructor_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import enum
    import sys
    import types

    class WriteMode(enum.Enum):
        WRITE_DISABLED = "write_disabled"
        WRITE_SYNC = "write_sync"

    seen = {}

    class DACSCacheClient:
        def __init__(self, write_mode: object = None, consistent_mode: bool = False) -> None:
            seen["args"] = (write_mode, consistent_mode)
            if consistent_mode:
                raise FileNotFoundError("STORAGE_INTERCEPT_CONFIG_FILEPATH must point at an existing regular file")

    package = types.ModuleType("py_dacs_client")
    package.WriteMode = WriteMode  # type: ignore[attr-defined]
    module = types.ModuleType("py_dacs_client.dacs_client")
    module.DACSCacheClient = DACSCacheClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "py_dacs_client", package)
    monkeypatch.setitem(sys.modules, "py_dacs_client.dacs_client", module)

    built = client.default_client_factory("write_sync", False)
    assert isinstance(built, DACSCacheClient)
    assert seen["args"] == (WriteMode.WRITE_SYNC, False)
    with pytest.raises(DacsUnavailableError, match="could not be initialised"):
        client.default_client_factory("write_sync", True)


def test_second_default_configuration_logs_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    factory = fakes.FakeFactory()
    monkeypatch.setattr(client, "default_client_factory", factory)
    client.get_client(None, "write_disabled", False)
    with caplog.at_level("WARNING", logger="dacs_fsspec"):
        client.get_client(None, "write_sync", False)
    assert "second DACS client configuration" in caplog.text
