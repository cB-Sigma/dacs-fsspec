"""Third-party libraries driving the filesystem (each test is skipped if its library is missing)."""

from __future__ import annotations

import io
import tarfile

import pytest

import fakes
from dacs_fsspec import DacsFileSystem
from fakes import FakeFactory


def _parquet(rows: dict[str, list[object]]) -> bytes:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    sink = io.BytesIO()
    pq.write_table(pa.table(rows), sink)
    return sink.getvalue()


def test_dask_dataframe_reads_a_glob() -> None:
    dd = pytest.importorskip("dask.dataframe")
    dask = pytest.importorskip("dask")
    factory = fakes.shared_cluster()
    for part in range(3):
        factory.put(f"acct/lake/dask/part-{part}.parquet", _parquet({"part": [part] * 5, "v": list(range(5))}))
    frame = dd.read_parquet("dacs://acct/lake/dask/*.parquet", storage_options={"client_factory": fakes.shared_factory})
    with dask.config.set(scheduler="synchronous"):
        result = frame.compute()
    assert len(result) == 15
    assert sorted(result["part"].unique()) == [0, 1, 2]


def test_huggingface_datasets_streaming_and_eager(factory: FakeFactory, tmp_path: object) -> None:
    datasets = pytest.importorskip("datasets")
    for part in range(2):
        factory.put(
            f"acct/corpus/part-{part}.parquet",
            _parquet({"text": [f"doc-{part}-{i}" for i in range(4)], "label": [part] * 4}),
        )
    options = {"client_factory": factory}
    streamed = datasets.load_dataset(
        "parquet",
        data_files="dacs://acct/corpus/*.parquet",
        split="train",
        streaming=True,
        storage_options=options,
    )
    rows = list(streamed)
    assert len(rows) == 8
    assert rows[0] == {"text": "doc-0-0", "label": 0}

    eager = datasets.load_dataset(
        "parquet",
        data_files="dacs://acct/corpus/*.parquet",
        split="train",
        storage_options=options,
        cache_dir=str(tmp_path),
    )
    assert eager.num_rows == 8


def test_zarr_store_over_the_filesystem(factory: FakeFactory) -> None:
    zarr = pytest.importorskip("zarr")
    np = pytest.importorskip("numpy")
    if int(zarr.__version__.split(".")[0]) < 3:
        pytest.skip("written against the zarr v3 storage API")
    from zarr.storage import MemoryStore

    # Build a small array in memory, then publish its chunks as blobs.
    memory = MemoryStore()
    source = zarr.create_array(memory, shape=(64, 64), chunks=(16, 16), dtype="i4")
    source[:] = np.arange(64 * 64, dtype="i4").reshape(64, 64)
    for key, value in memory._store_dict.items():
        factory.put(f"acct/arrays/grid.zarr/{key}", value.to_bytes())

    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
    from zarr.storage import FsspecStore

    fs = AsyncFileSystemWrapper(DacsFileSystem(client_factory=factory), asynchronous=True)
    store = FsspecStore(fs, path="acct/arrays/grid.zarr", read_only=True)
    array = zarr.open_array(store, mode="r")
    assert array.shape == (64, 64)
    assert int(array[17, 33]) == 17 * 64 + 33
    assert np.array_equal(array[:], np.arange(64 * 64, dtype="i4").reshape(64, 64))


def test_webdataset_with_a_registered_scheme(factory: FakeFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    wds = pytest.importorskip("webdataset")
    sink = io.BytesIO()
    with tarfile.open(fileobj=sink, mode="w") as archive:
        for index in range(5):
            for extension, payload in (("txt", f"sample {index}".encode()), ("cls", str(index % 2).encode())):
                member = tarfile.TarInfo(f"{index:04d}.{extension}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    factory.put("acct/shards/train-0000.tar", sink.getvalue())
    fs = DacsFileSystem(client_factory=factory)

    # webdataset opens URLs through its own scheme table rather than fsspec,
    # so the dacs:// scheme is registered once (this is the documented recipe).
    monkeypatch.setitem(wds.gopen_schemes, "dacs", lambda url, mode="rb", bufsize=8192, **kw: fs.open(url, mode))

    dataset = wds.WebDataset("dacs://acct/shards/train-0000.tar", shardshuffle=False, empty_check=False)
    samples = list(dataset)
    assert [sample["__key__"] for sample in samples] == [f"{i:04d}" for i in range(5)]
    assert samples[3]["txt"] == b"sample 3"
