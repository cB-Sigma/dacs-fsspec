"""Hugging Face `datasets` reading data files through the DACS cache.

    python examples/huggingface_datasets.py "dacs://account/container/corpus/*.parquet"

`datasets` resolves data files with fsspec, so a dacs:// URL works wherever a
remote URL is accepted.  Streaming avoids copying the files into the local
Hugging Face cache first.
"""

from __future__ import annotations

import sys

from datasets import load_dataset

dataset = load_dataset("parquet", data_files=sys.argv[1], split="train", streaming=True)
for index, row in enumerate(dataset):
    print(row)
    if index == 4:
        break
