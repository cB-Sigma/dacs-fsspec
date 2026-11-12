"""Command line interface: ``python -m dacs_fsspec`` / ``dacs-fsspec``.

Small on purpose: enough to check that a pod can see its data through DACS
(``doctor``, ``stat``, ``ls``) and to move single objects (``cat``, ``cp``).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from typing import IO, Any

from ._version import __version__
from .auth import resolve_auth
from .core import FALLBACK_ENV, DacsFileSystem
from .paths import PROTOCOL

# Environment variables read by the DACS client (verified against its source)
# and by this package.  The flag marks values that must never be printed.
ENVIRONMENT: tuple[tuple[str, bool, str], ...] = (
    ("CACHE_DISCOVERY_URL", False, "cache server discovery endpoint (needs CACHE_SERVER_PORT)"),
    ("CACHE_SERVER_LIST", False, "static comma-separated cache servers (needs CACHE_SERVER_PORT)"),
    ("CACHE_SERVER_PORT", False, "cache server port"),
    ("USE_SAS_TOKEN", False, "'true' selects SAS token authentication"),
    ("STORAGE_INTCPT_SAS_TOKEN", True, "SAS token (used when USE_SAS_TOKEN=true)"),
    ("SI_azBlobAccessToken", True, "pre-fetched Azure access token"),
    ("AZURE_CLIENT_ID", False, "managed identity client id"),
    ("MSI_ENDPOINT", False, "managed identity endpoint"),
    ("MSI_SECRET", True, "managed identity secret"),
    ("STORAGE_INTERCEPT_CONFIG_FILEPATH", False, "caller-owned native config file (overrides everything)"),
    ("AZURE_BLOB_ENDPOINT_SUFFIX", False, "extra trusted blob endpoint suffix"),
    ("DACS_DATA_LOG_LEVEL", False, "client log level (default INFO, logged to stdout)"),
    (FALLBACK_ENV, False, "dacs-fsspec: fall back to direct Blob reads when DACS is unavailable"),
)


def _is_remote(path: str) -> bool:
    return path.startswith(f"{PROTOCOL}://") or path.startswith("https://")


def _filesystem(args: argparse.Namespace) -> DacsFileSystem:
    options: dict[str, Any] = {"metadata": args.metadata}
    if args.fallback:
        options["fallback"] = True
    if getattr(args, "write_mode", None):
        options["write_mode"] = args.write_mode
        options["consistent_mode"] = bool(getattr(args, "overwrite", False))
    return DacsFileSystem(**options)


def _text(out: IO[bytes], line: str = "") -> None:
    out.write((line + "\n").encode("utf-8"))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def _cmd_ls(args: argparse.Namespace, out: IO[bytes]) -> int:
    fs = _filesystem(args)
    for entry in fs.ls(args.url, detail=True):
        url = fs.unstrip_protocol(entry["name"])
        if entry["type"] == "directory":
            url += "/"
        if args.long:
            size = "-" if entry["type"] == "directory" else str(entry["size"])
            _text(out, f"{size:>14}  {url}")
        else:
            _text(out, url)
    return 0


def _cmd_stat(args: argparse.Namespace, out: IO[bytes]) -> int:
    info = _filesystem(args).info(args.url)
    _text(out, json.dumps(info, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_cat(args: argparse.Namespace, out: IO[bytes]) -> int:
    fs = _filesystem(args)
    if args.start is not None or args.end is not None:
        out.write(fs.cat_file(args.url, start=args.start, end=args.end))
        return 0
    with fs.open(args.url, "rb", cache_type="none") as source:
        while True:
            chunk = source.read(fs.TRANSFER_CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
    return 0


def _cmd_cp(args: argparse.Namespace, out: IO[bytes]) -> int:
    src_remote, dst_remote = _is_remote(args.src), _is_remote(args.dst)
    if src_remote == dst_remote:
        raise SystemExit("cp: exactly one of SRC and DST must be a dacs:// URL (remote-to-remote is unsupported)")
    if src_remote:
        fs = _filesystem(args)
        destination = args.dst
        if os.path.isdir(destination):
            destination = os.path.join(destination, args.src.rstrip("/").rsplit("/", 1)[-1])
        fs.get_file(args.src, destination)
    else:
        args.write_mode = args.write_mode or "sync"
        _filesystem(args).put_file(args.src, args.dst)
    return 0


def client_report() -> dict[str, Any]:
    """Whether the DACS client is installed and importable in this interpreter."""
    report: dict[str, Any] = {"installed": False, "importable": False, "version": None, "error": None}
    try:
        report["installed"] = importlib.util.find_spec("py_dacs_client") is not None
    except (ImportError, ValueError):
        report["installed"] = False
    if not report["installed"]:
        report["error"] = "py_dacs_client is not installed (pip install dacs-client)"
        return report
    try:
        module = importlib.import_module("py_dacs_client.dacs_client")
        report["importable"] = module.DACSCacheClient is not None
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}".strip().splitlines()[0]
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        report["version"] = importlib.metadata.version("dacs-client")
    return report


def cache_report(env: Mapping[str, str]) -> tuple[bool, str]:
    """How the client will locate cache servers, mirroring its config logic."""
    port = env.get("CACHE_SERVER_PORT")
    if env.get("STORAGE_INTERCEPT_CONFIG_FILEPATH"):
        return True, "defined by the caller-owned config file (STORAGE_INTERCEPT_CONFIG_FILEPATH)"
    if env.get("CACHE_DISCOVERY_URL") and port:
        return True, f"discovery via {env['CACHE_DISCOVERY_URL']} port {port}"
    if env.get("CACHE_SERVER_LIST") and port:
        return True, f"static server list {env['CACHE_SERVER_LIST']} port {port}"
    return False, (
        "NOT configured: the client will read straight from Azure Blob Storage. "
        "Set CACHE_DISCOVERY_URL (or CACHE_SERVER_LIST) and CACHE_SERVER_PORT"
    )


def _version_of(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _entry_point_registered() -> bool:
    try:
        import fsspec

        return fsspec.get_filesystem_class(PROTOCOL) is DacsFileSystem
    except Exception:
        return False


def _cmd_doctor(args: argparse.Namespace, out: IO[bytes]) -> int:
    env = os.environ
    problems: list[str] = []
    libc = " ".join(part for part in platform.libc_ver() if part) or "unknown libc"

    _text(out, f"dacs-fsspec {__version__}")
    _text(out, f"  python        {platform.python_version()} ({sys.platform}, {platform.machine()}, {libc})")
    _text(out, f"  fsspec        {_version_of('fsspec')}")
    _text(out, f"  adlfs         {_version_of('adlfs')}  (optional: metadata='adlfs' and fallback reads)")
    registered = _entry_point_registered()
    _text(out, f"  entry point   {PROTOCOL}:// {'registered' if registered else 'NOT registered'}")
    if not registered:
        problems.append("the dacs:// protocol is not registered with fsspec (is the package installed?)")

    client = client_report()
    _text(out)
    _text(out, "DACS client (py_dacs_client)")
    if client["importable"]:
        _text(out, f"  importable    yes (dacs-client {client['version'] or 'unknown version'})")
    else:
        _text(out, f"  importable    NO: {client['error']}")
        problems.append("the DACS client is not importable")

    cache_ok, cache_text = cache_report(env)
    _text(out)
    _text(out, "Effective client configuration")
    _text(out, f"  cache         {cache_text}")
    _text(out, f"  auth          {resolve_auth(env).description}")
    fallback = env.get(FALLBACK_ENV, "").strip().lower() in ("1", "true", "yes", "on")
    _text(out, f"  fallback      {'on' if fallback else 'off'} ({FALLBACK_ENV})")
    if not cache_ok:
        problems.append("no cache servers are configured")
    if fallback and _version_of("adlfs") == "not installed":
        problems.append(f"{FALLBACK_ENV} is on but adlfs is not installed (pip install 'dacs-fsspec[adlfs]')")

    _text(out)
    _text(out, "Environment")
    for name, secret, description in ENVIRONMENT:
        value = env.get(name)
        if value is None:
            shown = "(unset)"
        elif secret:
            shown = f"(set, {len(value)} characters, hidden)"
        else:
            shown = value
        _text(out, f"  {name:<34} {shown}")
        if args.verbose:
            _text(out, f"  {'':<34} # {description}")

    if args.probe:
        _text(out)
        _text(out, f"Probe {args.probe}")
        try:
            fs = _filesystem(args)
            info = fs.info(args.probe)
            _text(out, f"  stat          ok ({info['type']}, {info['size']} bytes)")
            if info["type"] == "file" and info["size"]:
                first = fs.cat_file(args.probe, start=0, end=min(int(info["size"]), 4096))
                _text(out, f"  read          ok ({len(first)} bytes, served by {fs.status()['reads']})")
        except Exception as exc:
            _text(out, f"  FAILED        {type(exc).__name__}: {exc}")
            problems.append(f"probe failed: {type(exc).__name__}")

    _text(out)
    if problems:
        _text(out, "Problems found:")
        for problem in problems:
            _text(out, f"  - {problem}")
        return 1
    _text(out, "No problems found. (This does not prove a cache hit; use --probe URL to exercise a read.)")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--metadata",
        choices=("dacs", "adlfs"),
        default="dacs",
        help="where ls/stat come from: the DACS client (default) or Azure Blob via adlfs",
    )
    common.add_argument(
        "--fallback",
        action="store_true",
        help="read directly from Azure Blob Storage if DACS is unavailable (needs adlfs)",
    )

    parser = argparse.ArgumentParser(
        prog="dacs-fsspec",
        description="Inspect and move Azure blobs through the DACS cache.",
    )
    parser.add_argument("--version", action="version", version=f"dacs-fsspec {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    ls = commands.add_parser("ls", parents=[common], help="list a container or prefix")
    ls.add_argument("url")
    ls.add_argument("-l", "--long", action="store_true", help="show sizes")
    ls.set_defaults(run=_cmd_ls)

    stat = commands.add_parser("stat", parents=[common], help="print blob metadata as JSON")
    stat.add_argument("url")
    stat.set_defaults(run=_cmd_stat)

    cat = commands.add_parser("cat", parents=[common], help="write a blob (or a byte range) to stdout")
    cat.add_argument("url")
    cat.add_argument("--start", type=int, default=None, help="first byte (inclusive)")
    cat.add_argument("--end", type=int, default=None, help="last byte (exclusive)")
    cat.set_defaults(run=_cmd_cat)

    cp = commands.add_parser(
        "cp", parents=[common], help="copy one blob to a local file, or a local file to a new blob"
    )
    cp.add_argument("src")
    cp.add_argument("dst")
    cp.add_argument(
        "--write-mode",
        choices=("sync", "blob_only", "cache_only", "cache_blob_async"),
        default=None,
        help="DACS write mode for uploads (default: sync)",
    )
    cp.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing blob (client consistent mode; sync and blob_only only)",
    )
    cp.set_defaults(run=_cmd_cp)

    doctor = commands.add_parser(
        "doctor", parents=[common], help="show the effective configuration and check the client"
    )
    doctor.add_argument("--probe", metavar="URL", default=None, help="also stat and read the first bytes of this blob")
    doctor.add_argument("-v", "--verbose", action="store_true", help="describe each environment variable")
    doctor.set_defaults(run=_cmd_doctor)
    return parser


def _claim_stdout() -> IO[bytes]:
    """Reserve the real stdout for command output.

    The DACS client (Python and native layers) logs to stdout.  To keep
    ``cat`` output and pipelines clean, command output goes to a private
    duplicate of fd 1 and fd 1 itself is pointed at stderr.
    """
    sys.stdout.flush()
    private = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    return private


def main(argv: Sequence[str] | None = None, out: IO[bytes] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stream = out if out is not None else _claim_stdout()
    try:
        return int(args.run(args, stream))
    except BrokenPipeError:
        return 1
    except (OSError, NotImplementedError, ValueError, ImportError) as exc:
        sys.stderr.write(f"dacs-fsspec: {type(exc).__name__}: {exc}\n")
        return 2
    finally:
        with contextlib.suppress(BrokenPipeError, ValueError):
            stream.flush()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
