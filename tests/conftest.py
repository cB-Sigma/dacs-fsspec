from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from dacs_fsspec import DacsFileSystem, reset_clients
from fakes import FakeBlobBackend, FakeFactory

# Pseudo-random but deterministic payload, large enough to span many blocks.
PAYLOAD = bytes((i * 7 + (i >> 8)) % 251 for i in range(300_000))

BLOBS: dict[str, bytes] = {
    "acct/data/big.bin": PAYLOAD,
    "acct/data/hello.txt": b"hello\nworld\n",
    "acct/data/empty.bin": b"",
    "acct/data/train/part-0.csv": b"a,b\n1,2\n3,4\n",
    "acct/data/train/part-1.csv": b"a,b\n5,6\n",
    "acct/data/train/nested/deep.txt": b"deep",
    "acct/other/readme.md": b"# other",
}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in (
        "DACS_FSSPEC_FALLBACK",
        "USE_SAS_TOKEN",
        "STORAGE_INTCPT_SAS_TOKEN",
        "SI_azBlobAccessToken",
        "AZURE_CLIENT_ID",
        "MSI_ENDPOINT",
        "MSI_SECRET",
        "CACHE_DISCOVERY_URL",
        "CACHE_SERVER_LIST",
        "CACHE_SERVER_PORT",
    ):
        monkeypatch.delenv(name, raising=False)
    DacsFileSystem.clear_instance_cache()
    reset_clients()
    yield
    DacsFileSystem.clear_instance_cache()
    reset_clients()


@pytest.fixture
def factory() -> FakeFactory:
    return FakeFactory(BLOBS)


@pytest.fixture
def direct(factory: FakeFactory) -> FakeBlobBackend:
    return FakeBlobBackend(factory)


@pytest.fixture
def make_fs(factory: FakeFactory) -> Callable[..., DacsFileSystem]:
    def make(**options: Any) -> DacsFileSystem:
        options.setdefault("client_factory", factory)
        return DacsFileSystem(**options)

    return make


@pytest.fixture
def fs(make_fs: Callable[..., DacsFileSystem]) -> DacsFileSystem:
    return make_fs()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("DACS_FSSPEC_TEST_URL"):
        return
    skip = pytest.mark.skip(reason="live DACS cluster required: set DACS_FSSPEC_TEST_URL")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


def pytest_sessionfinish(session: pytest.Session) -> None:
    # Importing the real dacs-client makes its native layer write a per-process
    # configuration file; do not leave ours behind.
    leftover = os.path.join("/tmp", f"storageIntercept.{os.getpid()}.config")
    if os.path.isfile(leftover):
        os.remove(leftover)
