"""Exception translation between the DACS client and fsspec conventions.

The DACS client raises its own hierarchy (``DACSError`` and subclasses) whose
storage errors carry an ``errno_code`` attribute.  fsspec consumers expect the
builtin ``OSError`` family (``FileNotFoundError`` and friends).  Translation is
duck-typed on ``errno_code`` and the class name so that this module never has
to import the native client.
"""

from __future__ import annotations

import errno
import os

__all__ = [
    "DacsFallbackWarning",
    "DacsUnavailableError",
    "errno_of",
    "is_unavailable",
    "translate_error",
]

#: errno values that mean "the accelerator could not be reached or did not
#: answer", as opposed to "the blob is missing / forbidden / invalid".  EIO is
#: what the client reports for native failures it cannot classify.
UNAVAILABLE_ERRNOS = frozenset(
    {
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.ENETUNREACH,
        errno.ENETDOWN,
        errno.EHOSTUNREACH,
        errno.EHOSTDOWN,
        errno.ETIMEDOUT,
        errno.EAGAIN,
        errno.EPIPE,
        errno.EIO,
    }
)

_UNSUPPORTED_ERRNOS = frozenset({errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS})


class DacsFallbackWarning(UserWarning):
    """Emitted once when reads start bypassing DACS and go straight to Azure Blob."""


class DacsUnavailableError(OSError):
    """The DACS client could not be imported, constructed or reached."""


def errno_of(exc: BaseException) -> int | None:
    """Best-effort errno for a client exception."""
    code = getattr(exc, "errno_code", None)
    if isinstance(code, int) and code > 0:
        return code
    if isinstance(exc, OSError) and isinstance(exc.errno, int) and exc.errno > 0:
        return exc.errno
    return None


def is_unavailable(exc: BaseException) -> bool:
    """True if ``exc`` says DACS itself is unusable rather than the blob."""
    if isinstance(exc, DacsUnavailableError):
        return True
    if isinstance(exc, (FileNotFoundError, PermissionError, FileExistsError, IsADirectoryError, NotADirectoryError)):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    code = errno_of(exc)
    return code in UNAVAILABLE_ERRNOS


def translate_error(exc: BaseException, path: str | None = None) -> BaseException:
    """Map a client exception to the exception fsspec users expect.

    Builtin ``OSError`` subclasses, ``NotImplementedError`` and non-client
    exceptions pass through unchanged.  The original exception should be
    chained by the caller (``raise translate_error(e, path) from e``).
    """
    if isinstance(exc, (NotImplementedError, DacsUnavailableError)):
        return exc
    name = type(exc).__name__
    code = errno_of(exc)
    if name == "OperationNotSupportedError" or (code in _UNSUPPORTED_ERRNOS and not isinstance(exc, OSError)):
        return NotImplementedError(f"Not supported by the DACS client: {exc}")
    if isinstance(exc, OSError):
        return exc
    if code is None:
        return exc
    message = getattr(exc, "message", None) or str(exc) or os.strerror(code)
    # OSError picks the right subclass (FileNotFoundError, PermissionError,
    # FileExistsError, IsADirectoryError, TimeoutError, ...) from the errno.
    return OSError(code, str(message), path)
