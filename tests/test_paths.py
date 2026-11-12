from __future__ import annotations

import pytest

from dacs_fsspec import BlobPath, DacsFileSystem, from_adlfs_url, from_blobfile_url, to_dacs_url
from dacs_fsspec.paths import strip_protocol


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("dacs://acct/cont/a/b.bin", "acct/cont/a/b.bin"),
        ("dacs://acct/cont/a/b.bin/", "acct/cont/a/b.bin"),
        ("dacs://ACCT/cont/Key.BIN", "acct/cont/Key.BIN"),
        ("dacs://acct.blob.core.windows.net/cont/a", "acct/cont/a"),
        ("https://acct.blob.core.windows.net/cont/a?sv=1&sig=2", "acct/cont/a"),
        ("https://acct.blob.core.usgovcloudapi.net/cont/a", "acct.blob.core.usgovcloudapi.net/cont/a"),
        ("acct/cont/a", "acct/cont/a"),
        ("/acct/cont//a", "acct/cont/a"),
        ("dacs://acct/cont", "acct/cont"),
        ("dacs://acct", "acct"),
        ("dacs://", ""),
        ("", ""),
    ],
)
def test_strip_protocol(url: str, expected: str) -> None:
    assert strip_protocol(url) == expected
    assert DacsFileSystem._strip_protocol(url) == expected


def test_strip_protocol_list() -> None:
    assert DacsFileSystem._strip_protocol(["dacs://a/b/c", "a/b/d"]) == ["a/b/c", "a/b/d"]


@pytest.mark.parametrize("url", ["az://acct/cont/a", "abfs://cont/a", "abfss://cont@acct.dfs.core.windows.net/a"])
def test_ambiguous_schemes_are_rejected_with_guidance(url: str) -> None:
    with pytest.raises(ValueError, match="from_blobfile_url"):
        strip_protocol(url)


@pytest.mark.parametrize("url", ["s3://bucket/key", "http://acct.blob.core.windows.net/c/k"])
def test_foreign_schemes_are_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        strip_protocol(url)


@pytest.mark.parametrize(
    "url", ["dacs://../cont/a", "dacs://./cont/a", "dacs://user@acct/cont/a", "dacs://acct:443/c/a"]
)
def test_invalid_hosts_are_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="Invalid blob URL"):
        strip_protocol(url)


def test_blob_path_public_cloud() -> None:
    blob = BlobPath.parse("dacs://acct/cont/dir/file.bin")
    assert (blob.account, blob.container, blob.key) == ("acct", "cont", "dir/file.bin")
    assert blob.host == "acct.blob.core.windows.net"
    assert blob.account_name == "acct"
    assert blob.path == "acct/cont/dir/file.bin"
    assert blob.dacs_url == "dacs://acct/cont/dir/file.bin"
    assert blob.client_url == "az://acct.blob.core.windows.net/cont/dir/file.bin"
    assert blob.https_url == "https://acct.blob.core.windows.net/cont/dir/file.bin"
    assert blob.adlfs_path == "cont/dir/file.bin"
    assert blob.abfs_url == "abfs://cont@acct.dfs.core.windows.net/dir/file.bin"
    assert not (blob.is_root)
    assert not (blob.is_account)
    assert not (blob.is_container)


def test_blob_path_sovereign_cloud_keeps_full_host() -> None:
    blob = BlobPath.parse("dacs://acct.blob.core.chinacloudapi.cn/cont/k")
    assert blob.host == "acct.blob.core.chinacloudapi.cn"
    assert blob.account_name == "acct"
    assert blob.client_url == "az://acct.blob.core.chinacloudapi.cn/cont/k"
    assert blob.abfs_url == "abfs://cont@acct.dfs.core.chinacloudapi.cn/k"


def test_blob_path_levels_and_children() -> None:
    root = BlobPath.parse("")
    assert root.is_root
    assert root.path == ""
    account = root.child("acct")
    assert account.is_account
    container = account.child("cont")
    assert container.is_container
    assert container.client_url == "az://acct.blob.core.windows.net/cont"
    assert container.child("a").child("b/").path == "acct/cont/a/b"
    with pytest.raises(ValueError, match="storage account is required"):
        root.client_url
    with pytest.raises(ValueError, match="storage account is required"):
        root.https_url
    with pytest.raises(ValueError, match="need a container"):
        account.abfs_url


def test_from_blobfile_url() -> None:
    assert from_blobfile_url("az://acct/cont/k") == "dacs://acct/cont/k"
    assert from_blobfile_url("az://acct.blob.core.windows.net/cont/k") == "dacs://acct/cont/k"
    assert from_blobfile_url("https://acct.blob.core.windows.net/cont/k") == "dacs://acct/cont/k"
    assert from_blobfile_url("dacs://acct/cont/k") == "dacs://acct/cont/k"
    with pytest.raises(ValueError, match="Expected an az://"):
        from_blobfile_url("abfs://cont/k")
    with pytest.raises(ValueError, match="Expected an az://"):
        from_blobfile_url("acct/cont/k")


def test_from_adlfs_url() -> None:
    assert from_adlfs_url("az://cont/dir/k", account_name="acct") == "dacs://acct/cont/dir/k"
    assert from_adlfs_url("abfs://cont/k", "acct") == "dacs://acct/cont/k"
    assert from_adlfs_url("abfss://cont@acct.dfs.core.windows.net/dir/k") == "dacs://acct/cont/dir/k"
    assert (
        from_adlfs_url("abfs://cont@acct.dfs.core.usgovcloudapi.net/k")
        == "dacs://acct.blob.core.usgovcloudapi.net/cont/k"
    )
    with pytest.raises(ValueError, match="does not name a storage account"):
        from_adlfs_url("az://cont/k")
    with pytest.raises(ValueError, match="does not name a container"):
        from_adlfs_url("abfs:///k", "acct")
    with pytest.raises(ValueError, match="Expected an az://"):
        from_adlfs_url("https://acct.blob.core.windows.net/c/k")


def test_to_dacs_url_picks_the_dialect() -> None:
    assert to_dacs_url("az://acct/cont/k") == "dacs://acct/cont/k"
    assert to_dacs_url("az://cont/k", account_name="acct") == "dacs://acct/cont/k"
    assert to_dacs_url("abfs://cont@acct.dfs.core.windows.net/k") == "dacs://acct/cont/k"
    assert to_dacs_url("https://acct.blob.core.windows.net/cont/k") == "dacs://acct/cont/k"


def test_round_trip_through_unstrip(fs: DacsFileSystem) -> None:
    assert fs.unstrip_protocol("acct/cont/k") == "dacs://acct/cont/k"
    assert fs._strip_protocol(fs.unstrip_protocol("acct/cont/k")) == "acct/cont/k"
