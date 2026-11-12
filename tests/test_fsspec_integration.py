"""The package seen through fsspec's public entry points and its consumers."""

from __future__ import annotations

import importlib.metadata
import io

import fsspec
import pytest
from fsspec.core import url_to_fs

import fakes
from conftest import PAYLOAD
from dacs_fsspec import DacsFile, DacsFileSystem
from fakes import FakeFactory


def test_entry_point_is_declared() -> None:
    specs = importlib.metadata.entry_points(group="fsspec.specs")
    (entry,) = [e for e in specs if e.name == "dacs"]
    assert entry.value == "dacs_fsspec.core:DacsFileSystem"
    assert entry.load() is DacsFileSystem


def test_protocol_lookup() -> None:
    assert fsspec.get_filesystem_class("dacs") is DacsFileSystem
    assert "dacs" in fsspec.available_protocols()
    assert DacsFileSystem.protocol == "dacs"


def test_fsspec_filesystem(factory: FakeFactory) -> None:
    fs = fsspec.filesystem("dacs", client_factory=factory)
    assert isinstance(fs, DacsFileSystem)
    assert fs.cat_file("dacs://acct/data/hello.txt") == b"hello\nworld\n"


def test_fsspec_open(factory: FakeFactory) -> None:
    with fsspec.open("dacs://acct/data/big.bin", "rb", client_factory=factory) as f:
        assert isinstance(f, DacsFile)
        assert f.read(32) == PAYLOAD[:32]
    with fsspec.open("dacs://acct/data/hello.txt", "rt", client_factory=factory) as f:
        assert f.read() == "hello\nworld\n"
    with fsspec.open("https://acct.blob.core.windows.net/data/hello.txt", protocol="dacs", client_factory=factory) as f:
        assert f.read() == b"hello\nworld\n"


def test_fsspec_open_for_write(factory: FakeFactory) -> None:
    with fsspec.open("dacs://acct/data/w.txt", "wt", client_factory=factory, write_mode="sync") as f:
        f.write("written")
    assert factory.get("acct/data/w.txt") == b"written"


def test_url_to_fs(factory: FakeFactory) -> None:
    fs, path = url_to_fs("dacs://acct/data/train/part-0.csv", client_factory=factory)
    assert isinstance(fs, DacsFileSystem)
    assert path == "acct/data/train/part-0.csv"
    assert fs.unstrip_protocol(path) == "dacs://acct/data/train/part-0.csv"
    assert fs.cat_file(path).startswith(b"a,b")


def test_open_files_with_glob(factory: FakeFactory) -> None:
    files = fsspec.open_files("dacs://acct/data/train/*.csv", "rb", client_factory=factory)
    assert [f.path for f in files] == ["acct/data/train/part-0.csv", "acct/data/train/part-1.csv"]
    with files[1] as f:
        assert f.read() == b"a,b\n5,6\n"


def test_get_mapper(factory: FakeFactory) -> None:
    mapper = fsspec.get_mapper("dacs://acct/data/train", client_factory=factory)
    assert sorted(mapper) == ["nested/deep.txt", "part-0.csv", "part-1.csv"]
    assert mapper["nested/deep.txt"] == b"deep"
    assert "part-0.csv" in mapper
    assert "missing" not in mapper
    with pytest.raises(KeyError):
        mapper["missing"]


def test_chained_local_cache(factory: FakeFactory, tmp_path: object) -> None:
    url = "simplecache::dacs://acct/data/big.bin"
    options = {"dacs": {"client_factory": factory}, "simplecache": {"cache_storage": str(tmp_path)}}
    with fsspec.open(url, "rb", **options) as f:
        assert f.read() == PAYLOAD


def test_pandas_csv_and_parquet(factory: FakeFactory) -> None:
    pd = pytest.importorskip("pandas")
    options = {"client_factory": factory}
    frame = pd.read_csv("dacs://acct/data/train/part-0.csv", storage_options=options)
    assert frame.to_dict("list") == {"a": [1, 3], "b": [2, 4]}

    pytest.importorskip("pyarrow")
    source = pd.DataFrame({"x": range(1000), "y": [f"row-{i}" for i in range(1000)]})
    buffer = io.BytesIO()
    source.to_parquet(buffer)
    factory.put("acct/data/table.parquet", buffer.getvalue())
    loaded = pd.read_parquet("dacs://acct/data/table.parquet", storage_options=options)
    assert loaded.equals(source)
    # Column projection only needs part of the file: footer plus one column chunk.
    only_x = pd.read_parquet("dacs://acct/data/table.parquet", columns=["x"], storage_options=options)
    assert list(only_x.columns) == ["x"]


def test_pandas_write_parquet(factory: FakeFactory) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"x": [1, 2, 3]})
    options = {"client_factory": factory, "write_mode": "sync"}
    frame.to_parquet("dacs://acct/data/written.parquet", storage_options=options)
    assert pd.read_parquet("dacs://acct/data/written.parquet", storage_options=options).equals(frame)


def test_pyarrow_dataset_over_fsspec(factory: FakeFactory) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from pyarrow.fs import FSSpecHandler, PyFileSystem

    for part in range(3):
        sink = io.BytesIO()
        pq.write_table(pa.table({"part": [part] * 10, "value": list(range(10))}), sink)
        factory.put(f"acct/lake/events/part-{part}.parquet", sink.getvalue())
    fs = DacsFileSystem(client_factory=factory)
    dataset = ds.dataset("acct/lake/events", filesystem=PyFileSystem(FSSpecHandler(fs)), format="parquet")
    table = dataset.to_table()
    assert table.num_rows == 30
    assert sorted(set(table.column("part").to_pylist())) == [0, 1, 2]


def test_shared_module_level_factory_is_usable_via_storage_options() -> None:
    with fsspec.open("dacs://acct/data/hello.txt", "rb", client_factory=fakes.shared_factory) as f:
        assert f.read() == b"hello from the fake cache\n"
