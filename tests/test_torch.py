from __future__ import annotations

import io
import multiprocessing
import pickle
import sys

import pytest

import fakes
from fakes import FakeFactory

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

from dacs_fsspec import DacsFileSystem  # noqa: E402
from dacs_fsspec.torch import BlobDataset, load_safetensors, torch_load  # noqa: E402

OPTIONS = {"client_factory": fakes.shared_factory}
URLS = [f"dacs://acct/data/shard-{i:02d}.bin" for i in range(8)]


def _as_tensor(data: bytes) -> torch.Tensor:
    return torch.tensor([data[0], len(data)])


def _with_url(data: bytes, url: str) -> tuple[str, int]:
    return url, len(data)


def test_blob_dataset_basics() -> None:
    dataset = BlobDataset(URLS, storage_options=OPTIONS)
    assert len(dataset) == 8
    assert dataset[3] == bytes([3]) * 1003
    assert BlobDataset(URLS, transform=len, storage_options=OPTIONS)[5] == 1005
    assert BlobDataset(URLS, transform=_with_url, with_url=True, storage_options=OPTIONS)[1] == (URLS[1], 1001)


def test_blob_dataset_pickles_without_a_client() -> None:
    dataset = BlobDataset(URLS, transform=_as_tensor, storage_options=OPTIONS)
    dataset[0]
    clone = pickle.loads(pickle.dumps(dataset))
    assert clone.urls == URLS
    assert clone[2].tolist() == [2, 1002]


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_dataloader_workers(method: str) -> None:
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{method} is not available on this platform")
    if method == "fork" and not sys.platform.startswith("linux"):
        pytest.skip("fork() from a threaded test process is only exercised on Linux")
    dataset = BlobDataset(URLS, transform=_as_tensor, storage_options=OPTIONS)
    dataset[0]  # the parent process has already used its own client
    loader = DataLoader(dataset, batch_size=4, num_workers=2, multiprocessing_context=method)
    rows = torch.cat(list(loader)).tolist()
    assert rows == [[i, 1000 + i] for i in range(8)]


def test_torch_load_through_the_filesystem(factory: FakeFactory) -> None:
    buffer = io.BytesIO()
    state = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4), "step": torch.tensor(7)}
    torch.save(state, buffer)
    factory.put("acct/models/ckpt.pt", buffer.getvalue())
    fs = DacsFileSystem(client_factory=factory, block_size=256)
    loaded = torch_load("dacs://acct/models/ckpt.pt", fs=fs, weights_only=True)
    assert torch.equal(loaded["weight"], state["weight"])
    assert int(loaded["step"]) == 7
    again = torch_load("dacs://acct/models/ckpt.pt", storage_options={"client_factory": factory})
    assert torch.equal(again["weight"], state["weight"])
    assert factory.open_handles == 0


def test_load_safetensors(factory: FakeFactory) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    tensors = {"a": torch.ones(4, 4), "b": torch.arange(10)}
    factory.put("acct/models/model.safetensors", safetensors_torch.save(tensors))
    fs = DacsFileSystem(client_factory=factory)
    loaded = load_safetensors("dacs://acct/models/model.safetensors", fs=fs)
    assert set(loaded) == {"a", "b"}
    assert torch.equal(loaded["b"], tensors["b"])
    moved = load_safetensors("dacs://acct/models/model.safetensors", device=torch.device("cpu"), fs=fs)
    assert torch.equal(moved["a"], tensors["a"])


def test_safetensors_lazy_open_via_file_object(factory: FakeFactory) -> None:
    """Reading only the header through the buffered file touches a fraction of the blob."""
    safetensors_torch = pytest.importorskip("safetensors.torch")
    import json
    import struct

    blob = safetensors_torch.save({"big": torch.zeros(200_000), "small": torch.ones(3)})
    factory.put("acct/models/lazy.safetensors", blob)
    fs = DacsFileSystem(client_factory=factory, block_size=4096)
    with fs.open("dacs://acct/models/lazy.safetensors", "rb") as f:
        (header_size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_size))
    assert set(header) - {"__metadata__"} == {"big", "small"}
    assert sum(size for _, _, size in factory.reads) < len(blob) // 10


def test_torch_save_through_a_write_handle(factory: FakeFactory) -> None:
    fs = DacsFileSystem(client_factory=factory, write_mode="sync", block_size=512)
    state = {"weight": torch.arange(2000, dtype=torch.float32)}
    with fs.open("dacs://acct/checkpoints/step-1.pt", "wb") as f:
        torch.save(state, f)
    loaded = torch_load("dacs://acct/checkpoints/step-1.pt", fs=fs, weights_only=True)
    assert torch.equal(loaded["weight"], state["weight"])
