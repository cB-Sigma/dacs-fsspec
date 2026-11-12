"""Dask workers reading Parquet through the DACS cache.

    python examples/dask_dataframe.py "dacs://account/container/lake/events/*.parquet"

The filesystem is pickled to every worker; each worker process builds its own
DACS client lazily, so the workers need the client and its environment
variables (CACHE_DISCOVERY_URL, CACHE_SERVER_PORT, credentials).
"""

from __future__ import annotations

import sys

import dask.dataframe as dd

frame = dd.read_parquet(sys.argv[1], storage_options={"cache_type": "bytes"})
print(frame.npartitions, "partitions")
print(frame.head())
