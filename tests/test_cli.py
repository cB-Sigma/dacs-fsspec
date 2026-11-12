from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import fakes
from dacs_fsspec import __version__, cli
from dacs_fsspec import client as client_module
from fakes import FakeFactory


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch) -> FakeFactory:
    """Make the CLI's default client factory resolve to the in-memory fake."""
    factory = FakeFactory(
        {
            "acct/data/hello.txt": b"hello\nworld\n",
            "acct/data/train/part-0.csv": b"a,b\n1,2\n",
        }
    )
    monkeypatch.setattr(client_module, "default_client_factory", factory)
    return factory


def run(*argv: str) -> tuple[int, bytes]:
    out = io.BytesIO()
    code = cli.main(list(argv), out=out)
    return code, out.getvalue()


def test_ls(cluster: FakeFactory) -> None:
    code, out = run("ls", "dacs://acct/data")
    assert code == 0
    assert out.decode().splitlines() == ["dacs://acct/data/hello.txt", "dacs://acct/data/train/"]
    code, out = run("ls", "-l", "dacs://acct/data")
    lines = out.decode().splitlines()
    assert lines[0].split() == ["12", "dacs://acct/data/hello.txt"]
    assert lines[1].split() == ["-", "dacs://acct/data/train/"]


def test_stat(cluster: FakeFactory) -> None:
    code, out = run("stat", "dacs://acct/data/hello.txt")
    assert code == 0
    assert json.loads(out) == {"mtime": 1700000000.0, "name": "acct/data/hello.txt", "size": 12, "type": "file"}


def test_cat_whole_and_range(cluster: FakeFactory) -> None:
    assert run("cat", "dacs://acct/data/hello.txt") == (0, b"hello\nworld\n")
    assert run("cat", "dacs://acct/data/hello.txt", "--start", "6") == (0, b"world\n")
    assert run("cat", "dacs://acct/data/hello.txt", "--start", "1", "--end", "4") == (0, b"ell")


def test_cp_download_and_upload(cluster: FakeFactory, tmp_path: Path) -> None:
    target = tmp_path / "hello.txt"
    assert run("cp", "dacs://acct/data/hello.txt", str(target))[0] == 0
    assert target.read_bytes() == b"hello\nworld\n"

    folder = tmp_path / "folder"
    folder.mkdir()
    assert run("cp", "dacs://acct/data/hello.txt", str(folder))[0] == 0
    assert (folder / "hello.txt").read_bytes() == b"hello\nworld\n"

    source = tmp_path / "upload.bin"
    source.write_bytes(b"payload")
    assert run("cp", str(source), "dacs://acct/data/uploaded.bin")[0] == 0
    assert cluster.get("acct/data/uploaded.bin") == b"payload"
    assert cluster.clients[-1].write_mode == "write_sync"

    assert run("cp", str(source), "dacs://acct/data/uploaded.bin", "--overwrite", "--write-mode", "blob_only")[0] == 0
    assert cluster.clients[-1].write_mode == "write_blob_only"
    assert cluster.clients[-1].consistent_mode is True


def test_cp_rejects_remote_to_remote_and_local_to_local(cluster: FakeFactory, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="exactly one"):
        run("cp", "dacs://acct/data/hello.txt", "dacs://acct/data/copy.txt")
    with pytest.raises(SystemExit, match="exactly one"):
        run("cp", str(tmp_path / "a"), str(tmp_path / "b"))


def test_errors_become_exit_code_2(cluster: FakeFactory, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run("cat", "dacs://acct/data/missing.bin")
    assert code == 2
    assert out == b""
    assert "FileNotFoundError" in capsys.readouterr().err
    source_exists = run("cp", os.devnull, "dacs://acct/data/hello.txt")
    assert source_exists[0] == 2
    assert "consistent_mode" in capsys.readouterr().err
    assert run("ls", "az://acct/data")[0] == 2
    assert "Ambiguous URL" in capsys.readouterr().err


def test_fallback_and_metadata_flags_reach_the_filesystem(cluster: FakeFactory) -> None:
    pytest.importorskip("adlfs")
    args = cli.build_parser().parse_args(["ls", "--fallback", "--metadata", "adlfs", "dacs://acct/data"])
    fs = cli._filesystem(args)
    assert fs.fallback_enabled is True
    assert fs.status()["metadata"] == "AdlfsBackend"


def test_version_and_usage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"], out=io.BytesIO())
    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out
    with pytest.raises(SystemExit) as info:
        cli.main([], out=io.BytesIO())
    assert info.value.code == 2


def test_doctor_reports_problems_without_leaking_secrets(cluster: FakeFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "client_report", lambda: {"importable": False, "error": "not installed", "version": None})
    monkeypatch.setenv("USE_SAS_TOKEN", "true")
    monkeypatch.setenv("STORAGE_INTCPT_SAS_TOKEN", "sv=2024&sig=TOPSECRET")
    monkeypatch.setenv("MSI_SECRET", "ALSOSECRET")
    code, raw = run("doctor", "-v")
    text = raw.decode()
    assert code == 1
    assert f"dacs-fsspec {__version__}" in text
    assert "dacs:// registered" in text
    assert "importable    NO: not installed" in text
    assert "NOT configured" in text
    assert "SAS token" in text
    assert "TOPSECRET" not in text
    assert "ALSOSECRET" not in text
    assert "(set, 21 characters, hidden)" in text
    assert "USE_SAS_TOKEN                      true" in text
    assert "# cache server port" in text
    assert "- the DACS client is not importable" in text

    monkeypatch.setenv("DACS_FSSPEC_FALLBACK", "true")
    monkeypatch.setattr(cli, "_version_of", lambda distribution: "not installed")
    code, raw = run("doctor")
    assert code == 1
    assert "DACS_FSSPEC_FALLBACK is on but adlfs is not installed" in raw.decode()
    assert "- no cache servers are configured" in text


def test_doctor_healthy_with_probe(cluster: FakeFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "client_report", lambda: {"importable": True, "error": None, "version": "0.1.0"})
    monkeypatch.setenv("CACHE_DISCOVERY_URL", "cache-discovery.dacs.svc.cluster.local")
    monkeypatch.setenv("CACHE_SERVER_PORT", "9065")
    pytest.importorskip("adlfs")
    monkeypatch.setenv("DACS_FSSPEC_FALLBACK", "1")
    code, raw = run("doctor", "--probe", "dacs://acct/data/hello.txt")
    text = raw.decode()
    assert code == 0, text
    assert "importable    yes (dacs-client 0.1.0)" in text
    assert "discovery via cache-discovery.dacs.svc.cluster.local port 9065" in text
    assert "fallback      on" in text
    assert "stat          ok (file, 12 bytes)" in text
    assert "read          ok (12 bytes, served by dacs)" in text
    assert "No problems found" in text


def test_doctor_probe_failure(cluster: FakeFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "client_report", lambda: {"importable": True, "error": None, "version": None})
    monkeypatch.setenv("CACHE_SERVER_LIST", "10.0.0.1,10.0.0.2")
    monkeypatch.setenv("CACHE_SERVER_PORT", "9065")
    code, raw = run("doctor", "--probe", "dacs://acct/data/missing")
    text = raw.decode()
    assert code == 1
    assert "static server list 10.0.0.1,10.0.0.2 port 9065" in text
    assert "unknown version" in text
    assert "FAILED        FileNotFoundError" in text


def test_cache_report_variants() -> None:
    assert cli.cache_report({})[0] is False
    assert cli.cache_report({"CACHE_DISCOVERY_URL": "x"})[0] is False, "the port is required"
    assert cli.cache_report({"CACHE_SERVER_LIST": "a,b"})[0] is False
    ok, text = cli.cache_report({"CACHE_DISCOVERY_URL": "d", "CACHE_SERVER_LIST": "a", "CACHE_SERVER_PORT": "1"})
    assert ok
    assert "discovery" in text, "discovery takes precedence over the static list"
    ok, text = cli.cache_report({"STORAGE_INTERCEPT_CONFIG_FILEPATH": "/etc/si.config"})
    assert ok
    assert "caller-owned" in text


def test_client_report_states(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    monkeypatch.setitem(sys.modules, "py_dacs_client", None)
    report = cli.client_report()
    assert report["importable"] is False
    assert "not installed" in report["error"]

    package = types.ModuleType("py_dacs_client")
    package.__path__ = []  # type: ignore[attr-defined]
    package.__spec__ = __import__("importlib.machinery").machinery.ModuleSpec("py_dacs_client", None)
    monkeypatch.setitem(sys.modules, "py_dacs_client", package)
    monkeypatch.setitem(sys.modules, "py_dacs_client.dacs_client", types.ModuleType("py_dacs_client.dacs_client"))
    broken = cli.client_report()
    assert broken["installed"] is True
    assert broken["importable"] is False
    assert "AttributeError" in broken["error"]

    sys.modules["py_dacs_client.dacs_client"].DACSCacheClient = object  # type: ignore[attr-defined]
    assert cli.client_report()["importable"] is True


def test_module_entry_point_keeps_stdout_clean(tmp_path: Path) -> None:
    """Run ``python -m dacs_fsspec`` for real: noise printed to stdout by the
    client must not end up in the command's output stream."""
    site = tmp_path / "sitecustomize.py"
    site.write_text(
        "import sys\n"
        f"sys.path.insert(0, {os.path.dirname(fakes.__file__)!r})\n"
        "import fakes, dacs_fsspec.client as c\n"
        "def noisy(write_mode, consistent_mode):\n"
        "    print('DACS_CTR:INFO native log line on stdout')\n"
        "    import os; os.write(1, b'raw fd 1 noise\\n')\n"
        "    return fakes.shared_factory(write_mode, consistent_mode)\n"
        "c.default_client_factory = noisy\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    result = subprocess.run(
        [sys.executable, "-m", "dacs_fsspec", "cat", "dacs://acct/data/hello.txt"],
        capture_output=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == b"hello from the fake cache\n"
    assert b"native log line" in result.stderr
    assert b"raw fd 1 noise" in result.stderr

    version = subprocess.run(
        [sys.executable, "-m", "dacs_fsspec", "--version"], capture_output=True, env=env, timeout=120, check=False
    )
    assert version.stdout.decode().strip() == f"dacs-fsspec {__version__}"
