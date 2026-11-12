"""pandas and pyarrow through the DACS cache.

    python examples/pandas_parquet.py dacs://account/container/path/table.parquet

Needs a pod with the DACS client configured (see `python -m dacs_fsspec doctor`).
"""

from __future__ import annotations

import sys

import pandas as pd

url = sys.argv[1]

# Parquet does random access (footer, then column chunks), so a cache that
# keeps the bytes it has seen works better than plain read-ahead.
storage_options = {"cache_type": "bytes", "block_size": 4 * 2**20}

frame = pd.read_parquet(url, storage_options=storage_options)
print(frame.head())
print(f"{len(frame)} rows, {len(frame.columns)} columns")
