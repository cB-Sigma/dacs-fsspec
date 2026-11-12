"""A PyTorch DataLoader whose workers read through the DACS cache.

    python examples/torch_dataloader.py dacs://account/container/dataset/shards/

Each worker process creates its own DACS client the first time it reads; the
dataset object itself only carries URLs.
"""

from __future__ import annotations

import sys

import fsspec
from torch.utils.data import DataLoader

from dacs_fsspec.torch import BlobDataset


def decode(data: bytes, url: str) -> tuple[str, int]:
    # Replace with real decoding (image, tokenised shard, ...).
    return url, len(data)


def main(prefix: str) -> None:
    fs = fsspec.filesystem("dacs")
    urls = [fs.unstrip_protocol(path) for path in fs.find(prefix)]
    dataset = BlobDataset(urls, transform=decode, with_url=True)
    # "spawn" (or "forkserver") starts workers without inheriting the parent's
    # native client state, which is the safer choice with any native library.
    loader = DataLoader(dataset, batch_size=8, num_workers=4, multiprocessing_context="spawn")
    total = 0
    for _urls, sizes in loader:
        total += int(sizes.sum())
    print(f"read {len(urls)} blobs, {total} bytes")


if __name__ == "__main__":
    main(sys.argv[1])
