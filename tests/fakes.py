"""Test helpers: the packaged in-memory fake plus a cross-process factory."""

from __future__ import annotations

from dacs_fsspec.testing import (
    DACSError,
    DACSNetworkError,
    DACSNotFoundError,
    DACSStorageError,
    FakeBlobBackend,
    FakeDacs,
    FakeDacsClient,
    FakeHandle,
    OperationNotSupportedError,
    not_found,
    unreachable,
)

FakeFactory = FakeDacs

__all__ = [
    "DACSError",
    "DACSNetworkError",
    "DACSNotFoundError",
    "DACSStorageError",
    "FakeBlobBackend",
    "FakeDacsClient",
    "FakeFactory",
    "FakeHandle",
    "OperationNotSupportedError",
    "not_found",
    "seed_blobs",
    "shared_cluster",
    "shared_factory",
    "unreachable",
]


# -- a picklable, importable factory for cross-process tests -------------------
def seed_blobs() -> dict[str, bytes]:
    blobs = {f"acct/data/shard-{i:02d}.bin": bytes([i]) * (1000 + i) for i in range(8)}
    blobs["acct/data/hello.txt"] = b"hello from the fake cache\n"
    return blobs


_SHARED: FakeDacs | None = None


def shared_cluster() -> FakeDacs:
    """The process-wide fake behind :func:`shared_factory`."""
    global _SHARED
    if _SHARED is None:
        _SHARED = FakeDacs(seed_blobs())
    return _SHARED


def shared_factory(write_mode: str, consistent_mode: bool) -> FakeDacsClient:
    """Module-level factory: picklable by reference, deterministic in any process."""
    return shared_cluster()(write_mode, consistent_mode)
