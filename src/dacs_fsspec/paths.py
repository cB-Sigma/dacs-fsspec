"""Blob addressing: parsing and converting between URL dialects.

The DACS Python client addresses blobs as ``az://<account>/<container>/<blob>``
(short form), ``az://<account>.blob.core.windows.net/<container>/<blob>``
(canonical long form) or ``https://<host>/<container>/<blob>``.  That is the
*blobfile* dialect of ``az://``, where the first component is the storage
account.  It is **not** the *adlfs* dialect, where ``az://<container>/<blob>``
puts the container first and takes the account from ``storage_options``.

To stay unambiguous this package uses its own scheme::

    dacs://<account>/<container>/<blob>
    dacs://<account>.<blob endpoint suffix>/<container>/<blob>

and offers explicit converters from and to the other dialects.
"""

from __future__ import annotations

from dataclasses import dataclass

PROTOCOL = "dacs"
PUBLIC_BLOB_SUFFIX = ".blob.core.windows.net"
_DFS_MARKER = ".dfs."
_BLOB_MARKER = ".blob."

__all__ = [
    "PROTOCOL",
    "PUBLIC_BLOB_SUFFIX",
    "BlobPath",
    "from_adlfs_url",
    "from_blobfile_url",
    "strip_protocol",
    "to_dacs_url",
]


def _clean_host(host: str, original: str) -> str:
    host = host.strip().lower()
    if host in ("", ".", ".."):
        raise ValueError(f"Invalid blob URL {original!r}: expected a storage account or blob hostname")
    if "@" in host or ":" in host:
        raise ValueError(f"Invalid blob URL {original!r}: credentials and ports are not allowed in the hostname")
    if host.endswith(PUBLIC_BLOB_SUFFIX) and host != PUBLIC_BLOB_SUFFIX:
        # Collapse the public-cloud long form so one blob has one canonical path.
        host = host[: -len(PUBLIC_BLOB_SUFFIX)]
    return host


def strip_protocol(path: str) -> str:
    """Return the canonical protocol-less path ``account/container/blob``.

    Accepts ``dacs://`` URLs, ``https://<blob host>/...`` URLs and already
    stripped paths.  ``az://`` and ``abfs://`` are rejected on purpose because
    their meaning depends on the library that produced them; use
    :func:`from_blobfile_url` or :func:`from_adlfs_url` to convert explicitly.
    """
    original = path
    scheme, sep, rest = path.partition("://")
    if sep:
        scheme = scheme.lower()
        if scheme in ("az", "abfs", "abfss", "adl", "wasb", "wasbs"):
            raise ValueError(
                f"Ambiguous URL {original!r}: '{scheme}://' means different things to different "
                "libraries. Convert it with dacs_fsspec.from_blobfile_url() (account first) or "
                "dacs_fsspec.from_adlfs_url() (container first)."
            )
        if scheme not in (PROTOCOL, "https"):
            raise ValueError(f"Unsupported URL scheme in {original!r}; expected 'dacs://' or 'https://'")
        if scheme == "https":
            rest = rest.split("?", 1)[0].split("#", 1)[0]
        path = rest
    path = path.lstrip("/")
    if not path:
        return ""
    host, slash, tail = path.partition("/")
    host = _clean_host(host, original)
    tail = tail.rstrip("/") if slash else ""
    while "//" in tail:
        tail = tail.replace("//", "/")
    tail = tail.lstrip("/")
    return f"{host}/{tail}" if tail else host


@dataclass(frozen=True)
class BlobPath:
    """A parsed blob location.

    ``account`` is either a bare storage account name (public Azure) or a
    complete blob hostname (sovereign or custom clouds).
    """

    account: str
    container: str = ""
    key: str = ""

    @classmethod
    def parse(cls, path: str) -> BlobPath:
        stripped = strip_protocol(path)
        if not stripped:
            return cls("")
        account, _, rest = stripped.partition("/")
        container, _, key = rest.partition("/")
        return cls(account, container, key)

    # -- structure -----------------------------------------------------
    @property
    def is_root(self) -> bool:
        return not self.account

    @property
    def is_account(self) -> bool:
        return bool(self.account) and not self.container

    @property
    def is_container(self) -> bool:
        return bool(self.container) and not self.key

    @property
    def host(self) -> str:
        """Full blob endpoint hostname."""
        if "." in self.account:
            return self.account
        return self.account + PUBLIC_BLOB_SUFFIX

    @property
    def account_name(self) -> str:
        """Bare storage account name (first DNS label of the host)."""
        return self.account.split(".", 1)[0]

    @property
    def path(self) -> str:
        """Canonical fsspec path: ``account/container/key``."""
        return "/".join(part for part in (self.account, self.container, self.key) if part)

    def child(self, name: str) -> BlobPath:
        name = name.strip("/")
        if self.is_root:
            return BlobPath(name)
        if self.is_account:
            return BlobPath(self.account, name)
        key = f"{self.key}/{name}" if self.key else name
        return BlobPath(self.account, self.container, key)

    # -- renderings ----------------------------------------------------
    def _tail(self) -> str:
        return "/".join(part for part in (self.container, self.key) if part)

    def _require_account(self) -> None:
        if self.is_root:
            raise ValueError("A storage account is required: dacs://<account>/<container>/<blob>")

    @property
    def dacs_url(self) -> str:
        return f"{PROTOCOL}://{self.path}"

    @property
    def client_url(self) -> str:
        """The canonical long ``az://`` form the DACS client documents."""
        self._require_account()
        return f"az://{self.host}/{self._tail()}"

    @property
    def https_url(self) -> str:
        self._require_account()
        return f"https://{self.host}/{self._tail()}"

    @property
    def adlfs_path(self) -> str:
        """``container/key`` as adlfs expects it (the account goes in storage options)."""
        return self._tail()

    @property
    def abfs_url(self) -> str:
        """``abfs://container@account.dfs.../key``, self-describing for adlfs."""
        self._require_account()
        if not self.container:
            raise ValueError("abfs:// URLs need a container")
        dfs_host = self.host.replace(_BLOB_MARKER, _DFS_MARKER, 1)
        return f"abfs://{self.container}@{dfs_host}/{self.key}"


def from_blobfile_url(url: str) -> str:
    """Convert a blobfile / boostedblob / DACS-client URL to ``dacs://``.

    Accepts ``az://account/container/blob``, the long
    ``az://account.blob.core.windows.net/container/blob`` and
    ``https://account.blob.core.windows.net/container/blob``.
    """
    scheme, sep, rest = url.partition("://")
    if not sep or scheme.lower() not in ("az", "https", PROTOCOL):
        raise ValueError(f"Expected an az://, https:// or dacs:// URL, got {url!r}")
    if scheme.lower() == "az":
        url = f"{PROTOCOL}://{rest}"
    return BlobPath.parse(url).dacs_url


def from_adlfs_url(url: str, account_name: str | None = None) -> str:
    """Convert an adlfs-style URL to ``dacs://``.

    Accepts ``az://container/blob`` and ``abfs[s]://container/blob`` (both need
    ``account_name``, exactly as adlfs needs it in ``storage_options``) and the
    self-describing ``abfs[s]://container@account.dfs.core.windows.net/blob``.
    """
    scheme, sep, rest = url.partition("://")
    if not sep or scheme.lower() not in ("az", "abfs", "abfss"):
        raise ValueError(f"Expected an az://, abfs:// or abfss:// URL, got {url!r}")
    authority, _, key = rest.partition("/")
    if "@" in authority:
        container, _, host = authority.partition("@")
        host = host.lower().replace(_DFS_MARKER, _BLOB_MARKER, 1)
        account = host
    else:
        if not account_name:
            raise ValueError(
                f"{url!r} does not name a storage account; pass account_name=... "
                "(the same value adlfs takes in storage_options)"
            )
        container, account = authority, account_name
    if not container:
        raise ValueError(f"{url!r} does not name a container")
    return BlobPath.parse(f"{PROTOCOL}://{account}/{container}/{key}").dacs_url


def to_dacs_url(url: str, account_name: str | None = None) -> str:
    """Best-effort conversion of any supported dialect to ``dacs://``.

    ``az://`` is treated as the adlfs dialect when ``account_name`` is given and
    as the blobfile dialect otherwise.
    """
    scheme = url.partition("://")[0].lower()
    if scheme in ("abfs", "abfss") or (scheme == "az" and account_name):
        return from_adlfs_url(url, account_name)
    return from_blobfile_url(url)
