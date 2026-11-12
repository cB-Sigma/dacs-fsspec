"""Runs anywhere, no cluster needed: the filesystem against the in-memory fake.

    python examples/offline_demo.py

Useful to see the API and to model your own tests on.
"""

from __future__ import annotations

import io
import warnings

import fsspec

from dacs_fsspec import DacsFallbackWarning, DacsFileSystem
from dacs_fsspec.testing import FakeBlobBackend, FakeDacs, unreachable

fake = FakeDacs(
    {
        "dacs://acct/datasets/train/part-0.csv": b"id,label\n1,cat\n2,dog\n",
        "dacs://acct/datasets/train/part-1.csv": b"id,label\n3,bird\n",
        "dacs://acct/models/weights.bin": bytes(range(256)) * 4096,
    }
)
options = {"client_factory": fake}

# 1. Plain fsspec entry points work with a dacs:// URL.
with fsspec.open("dacs://acct/datasets/train/part-0.csv", "rt", **options) as f:
    print("first line:", f.readline().strip())

fs = fsspec.filesystem("dacs", **options)
print("glob:", fs.glob("dacs://acct/datasets/train/*.csv"))
print("size:", fs.size("dacs://acct/models/weights.bin"))

# 2. Buffered reads: many small reads become a few positional reads on one handle.
fake.calls.clear()
with fs.open("dacs://acct/models/weights.bin", "rb", block_size=256 * 1024) as f:
    sink = io.BytesIO()
    while chunk := f.read(4096):
        sink.write(chunk)
print(f"read {sink.tell()} bytes with {fake.calls['read']} native reads on {fake.calls['open']} handle")

# 3. Ranged reads.
print("bytes 10..15:", fs.cat_file("dacs://acct/models/weights.bin", start=10, end=16))

# 4. Fallback: when DACS is unreachable, read from Blob directly, warning once.
resilient = DacsFileSystem(client_factory=fake, fallback=FakeBlobBackend(fake))
fake.fail = unreachable()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    data = resilient.cat_file("dacs://acct/datasets/train/part-1.csv")
    resilient.cat_file("dacs://acct/datasets/train/part-0.csv")
fallback_warnings = [w for w in caught if issubclass(w.category, DacsFallbackWarning)]
print(f"fallback served {len(data)} bytes; warnings emitted: {len(fallback_warnings)}")
print("status:", resilient.status()["reads"])
