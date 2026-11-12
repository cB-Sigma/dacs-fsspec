"""Compatibility checks against the real ``dacs-client`` wheel.

These run only where ``py_dacs_client`` imports (Linux, glibc >= 2.35).  They
need no cluster: they introspect the real classes so that API drift between
the client and this adapter is caught, and they exercise the code paths that
do not perform I/O.
"""

from __future__ import annotations

import ctypes
import enum
import errno
import inspect
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from dacs_fsspec import BlobPath, DacsClientProtocol, DacsFileSystem, client
from dacs_fsspec.auth import resolve_auth
from dacs_fsspec.errors import is_unavailable, translate_error

pytestmark = pytest.mark.real_client

try:
    import py_dacs_client
    from py_dacs_client import dacs_client as real
    from py_dacs_client import dacs_data_loader, dacs_exceptions, dacs_utils
except Exception as exc:  # ImportError, or OSError when the native library cannot load
    pytest.skip(f"dacs-client is not importable here: {type(exc).__name__}: {exc}", allow_module_level=True)


def _parameters(function: object) -> list[str]:
    return [name for name in inspect.signature(function).parameters if name != "self"]  # type: ignore[arg-type]


@pytest.fixture
def caller_owned_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the client at a config file we own so construction writes nothing elsewhere."""
    config = tmp_path / "storageIntercept.config"
    config.write_text("storagePath /mnt/dacs_cache\ntype blob\n")
    monkeypatch.setenv("STORAGE_INTERCEPT_CONFIG_FILEPATH", str(config))
    return config


def test_client_exposes_the_methods_we_call() -> None:
    cls = real.DACSCacheClient
    assert _parameters(cls.__init__) == ["executor", "write_mode", "consistent_mode"]
    assert _parameters(cls.get_file_stat_sync) == ["blob_path"]
    assert _parameters(cls.list_dir_sync) == ["blob_path"]
    assert _parameters(cls.get_file_handle_sync) == ["blob_path", "flags"]
    assert _parameters(cls.release_file_handle_sync) == ["file_handle"]
    assert callable(cls.shutdown_sync)
    for method in ("get_file_stat_sync", "list_dir_sync", "get_file_handle_sync", "release_file_handle_sync"):
        assert not inspect.iscoroutinefunction(getattr(cls, method)), method


def test_handle_exposes_the_methods_we_call() -> None:
    handle = real.DACSCacheFileHandle
    assert _parameters(handle.read_sync) == ["buf", "size", "offset"]
    assert _parameters(handle.write_sync) == ["buf", "size", "offset"]
    assert _parameters(handle.__init__)[:2] == ["pathname", "flags"]


def test_write_mode_values_match() -> None:
    assert issubclass(py_dacs_client.WriteMode, enum.Enum)
    assert {member.value for member in py_dacs_client.WriteMode} == set(client.WRITE_MODES.values())
    for member in py_dacs_client.WriteMode:
        assert client.normalize_write_mode(member) == member.value
        assert py_dacs_client.WriteMode(client.normalize_write_mode(member.value)) is member


def test_stat_structure_has_the_fields_we_read() -> None:
    fields = {name for name, _ in dacs_data_loader.Stat64._fields_}
    assert {"st_mode", "st_size", "st_mtime"} <= fields


def test_our_client_urls_are_already_in_canonical_form() -> None:
    for path in ("acct/cont/dir/blob.bin", "acct.blob.core.usgovcloudapi.net/cont/blob", "acct/cont"):
        url = BlobPath.parse(path).client_url
        assert real.normalize_azure_path(url) == url
    assert real.normalize_azure_path("az://acct/cont/blob") == BlobPath.parse("acct/cont/blob").client_url
    assert (
        real.normalize_azure_path(BlobPath.parse("acct/cont/blob").https_url)
        == BlobPath.parse("acct/cont/blob").client_url
    )
    assert real.DEFAULT_INTERCEPT_PATH == "/mnt/dacs_cache"


def test_native_argtypes_accept_our_buffers() -> None:
    """The exact ctypes objects we hand to read_sync/write_sync convert for the native call."""
    library = dacs_data_loader.storage_intercept_lib
    read_buffer_type = library.pread.argtypes[1]
    write_buffer_type = library.pwrite.argtypes[1]
    out = bytearray(64)
    window = (ctypes.c_char * 32).from_buffer(out, 16)
    read_buffer_type.from_param(window)
    chunk = (ctypes.c_char * 8).from_buffer_copy(b"12345678")
    write_buffer_type.from_param(chunk)
    write_buffer_type.from_param(ctypes.byref(chunk, 4))


def test_real_exceptions_translate_to_builtins() -> None:
    cases = {
        dacs_exceptions.DACSNotFoundError: (errno.ENOENT, FileNotFoundError),
        dacs_exceptions.DACSPermissionError: (errno.EACCES, PermissionError),
        dacs_exceptions.DACSAuthenticationError: (errno.EPERM, PermissionError),
        dacs_exceptions.DACSExistsError: (errno.EEXIST, FileExistsError),
        dacs_exceptions.DACSIsDirectoryError: (errno.EISDIR, IsADirectoryError),
        dacs_exceptions.DACSNotDirectoryError: (errno.ENOTDIR, NotADirectoryError),
        dacs_exceptions.DACSTimeoutError: (errno.ETIMEDOUT, TimeoutError),
        dacs_exceptions.DACSNetworkError: (errno.ECONNREFUSED, ConnectionRefusedError),
        dacs_exceptions.DACSReadOnlyError: (errno.EROFS, OSError),
    }
    for real_class, (code, builtin) in cases.items():
        assert dacs_exceptions.errno_to_exception_class(code) is real_class
        raised = real_class(message="open failed", operation="open", path="/mnt/x", error_code=-1, errno_code=code)
        assert not isinstance(raised, OSError)
        translated = translate_error(raised, "acct/cont/blob")
        assert isinstance(translated, builtin), real_class
        assert translated.errno == code  # type: ignore[attr-defined]
        assert translated.filename == "acct/cont/blob"  # type: ignore[attr-defined]
    unsupported = translate_error(dacs_exceptions.OperationNotSupportedError("glob() is not supported"))
    assert isinstance(unsupported, NotImplementedError)


def test_real_exceptions_classify_for_fallback() -> None:
    def make(cls: type, code: int) -> Exception:
        return cls(message="x", errno_code=code)  # type: ignore[no-any-return]

    assert is_unavailable(make(dacs_exceptions.DACSNetworkError, errno.ECONNREFUSED))
    assert is_unavailable(make(dacs_exceptions.DACSTimeoutError, errno.ETIMEDOUT))
    assert is_unavailable(make(dacs_exceptions.DACSServiceUnavailableError, errno.EAGAIN))
    assert is_unavailable(make(dacs_exceptions.DACSStorageError, errno.EIO))
    assert not is_unavailable(make(dacs_exceptions.DACSNotFoundError, errno.ENOENT))
    assert not is_unavailable(make(dacs_exceptions.DACSPermissionError, errno.EACCES))
    assert not is_unavailable(make(dacs_exceptions.DACSAuthenticationError, errno.EPERM))


def test_check_native_result_raises_what_we_expect() -> None:
    ctypes.set_errno(errno.ENOENT)
    with pytest.raises(dacs_exceptions.DACSNotFoundError) as info:
        dacs_exceptions.check_native_result(-1, operation="open", path="/mnt/dacs_cache/a/b/c")
    assert isinstance(translate_error(info.value, "a/b/c"), FileNotFoundError)


def test_auth_resolution_matches_the_client_source() -> None:
    """Our mirror of the auth order must agree with the conditions in the client's config writer."""
    source = inspect.getsource(dacs_utils._initialize_dacs_client_config)
    for variable in (
        "USE_SAS_TOKEN",
        "STORAGE_INTCPT_SAS_TOKEN",
        "SI_azBlobAccessToken",
        "AZURE_CLIENT_ID",
        "MSI_ENDPOINT",
        "MSI_SECRET",
        "CACHE_DISCOVERY_URL",
        "CACHE_SERVER_LIST",
        "CACHE_SERVER_PORT",
        "STORAGE_INTERCEPT_CONFIG_FILEPATH",
    ):
        assert variable in source, variable
    order = [source.index(token) for token in ("azBlobSASToken {", "azBlobAccessToken {", "azBlobMSIEndpoint {")]
    assert order == sorted(order)
    assert resolve_auth({"USE_SAS_TOKEN": "true", "STORAGE_INTCPT_SAS_TOKEN": "s"}).mode == "sas_token"


def test_default_factory_builds_a_real_client(caller_owned_config: Path) -> None:
    built = client.default_client_factory("write_sync", True)
    assert isinstance(built, real.DACSCacheClient)
    assert isinstance(built, DacsClientProtocol)
    assert built.write_mode is py_dacs_client.WriteMode.WRITE_SYNC
    assert built.consistent_mode is True
    assert os.environ["STORAGE_INTERCEPT_CONFIG_FILEPATH"] == str(caller_owned_config)
    assert caller_owned_config.read_text().startswith("storagePath")


def test_filesystem_resolves_the_real_client_lazily(caller_owned_config: Path) -> None:
    fs = DacsFileSystem(skip_instance_cache=True)
    assert fs._client is None
    assert isinstance(fs._get_client(), real.DACSCacheClient)
    assert fs._get_client() is fs._get_client()


def test_client_init_failure_becomes_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dacs_fsspec import DacsUnavailableError

    monkeypatch.setenv("STORAGE_INTERCEPT_CONFIG_FILEPATH", str(tmp_path / "does-not-exist.config"))
    with pytest.raises(DacsUnavailableError, match="could not be initialised"):
        client.default_client_factory("write_disabled", False)
