"""End-to-end tests against a live DACS deployment.

Skipped unless ``DACS_FSSPEC_TEST_URL`` names an existing, non-empty blob that
the current environment can read through DACS, for example::

    DACS_FSSPEC_TEST_URL=dacs://account/container/path/to/blob pytest -m integration

Set ``DACS_FSSPEC_TEST_WRITE_PREFIX`` (a ``dacs://`` prefix that may be written
to) to also exercise writes.
"""

from __future__ import annotations

import os
import uuid

import pytest

from dacs_fsspec import DacsFileSystem

pytestmark = pytest.mark.integration

URL = os.environ.get("DACS_FSSPEC_TEST_URL", "")
WRITE_PREFIX = os.environ.get("DACS_FSSPEC_TEST_WRITE_PREFIX", "")


@pytest.fixture
def live() -> DacsFileSystem:
    return DacsFileSystem(skip_instance_cache=True)


def test_stat_and_ranged_read(live: DacsFileSystem) -> None:
    info = live.info(URL)
    assert info["type"] == "file"
    size = info["size"]
    assert size > 0
    head = live.cat_file(URL, 0, min(size, 1024))
    assert len(head) == min(size, 1024)
    assert live.cat_file(URL, -1) == live.cat_file(URL, size - 1, size)
    assert live.cat_file(URL, size, size + 10) == b""


def test_buffered_read_matches_cat(live: DacsFileSystem) -> None:
    size = live.size(URL)
    window = min(size, 4 * 2**20)
    with live.open(URL, "rb", block_size=2**20) as f:
        streamed = f.read(window)
        f.seek(0)
        assert f.read(16) == streamed[:16]
    assert streamed == live.cat_file(URL, 0, window)


def test_listing_the_parent_contains_the_blob(live: DacsFileSystem) -> None:
    path = live._strip_protocol(URL)
    assert path in live.ls(live._parent(path), detail=False)
    assert live.exists(URL)
    assert not live.exists(URL + ".does-not-exist-" + uuid.uuid4().hex)


def test_metadata_backends_agree(live: DacsFileSystem) -> None:
    pytest.importorskip("adlfs")
    direct = DacsFileSystem(metadata="adlfs", skip_instance_cache=True)
    assert direct.info(URL)["size"] == live.info(URL)["size"]


@pytest.mark.skipif(not WRITE_PREFIX, reason="set DACS_FSSPEC_TEST_WRITE_PREFIX to test writes")
def test_write_then_read_back() -> None:
    fs = DacsFileSystem(write_mode="sync", skip_instance_cache=True)
    target = f"{WRITE_PREFIX.rstrip('/')}/dacs-fsspec-{uuid.uuid4().hex}.bin"
    payload = os.urandom(3 * 2**20 + 17)
    with fs.open(target, "wb", block_size=2**20) as f:
        f.write(payload)
    assert fs.cat_file(target) == payload
    with pytest.raises(FileExistsError):
        fs.pipe_file(target, b"again")
