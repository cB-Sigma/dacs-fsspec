"""fsspec filesystem for DACS, the Distributed Accelerator for Cloud Storage.

    >>> import fsspec
    >>> with fsspec.open("dacs://account/container/model.safetensors", "rb") as f:
    ...     header = f.read(8)

This is an unofficial community project; it is not affiliated with or
endorsed by Microsoft.
"""

from ._version import __version__
from .backends import AdlfsBackend, ClientMetadataBackend, DirectReader, MetadataBackend
from .client import DacsClientProtocol, DacsHandleProtocol, default_client_factory, reset_clients
from .core import DacsFile, DacsFileSystem
from .errors import DacsFallbackWarning, DacsUnavailableError
from .paths import BlobPath, from_adlfs_url, from_blobfile_url, to_dacs_url

__all__ = [
    "AdlfsBackend",
    "BlobPath",
    "ClientMetadataBackend",
    "DacsClientProtocol",
    "DacsFallbackWarning",
    "DacsFile",
    "DacsFileSystem",
    "DacsHandleProtocol",
    "DacsUnavailableError",
    "DirectReader",
    "MetadataBackend",
    "__version__",
    "default_client_factory",
    "from_adlfs_url",
    "from_blobfile_url",
    "reset_clients",
    "to_dacs_url",
]
