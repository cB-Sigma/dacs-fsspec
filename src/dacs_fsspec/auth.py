"""Mirror the DACS client's credential resolution for direct Blob access.

The client decides how to authenticate from environment variables, in this
order (see ``_initialize_dacs_client_config`` in ``py_dacs_client.dacs_utils``):

1. ``USE_SAS_TOKEN=true`` and ``STORAGE_INTCPT_SAS_TOKEN``  -> SAS token
2. ``SI_azBlobAccessToken``                                 -> static bearer token
3. ``AZURE_CLIENT_ID`` + ``MSI_ENDPOINT`` + ``MSI_SECRET``  -> managed identity
4. otherwise                                                -> Azure Identity SDK

When this package talks to Azure Blob directly (metadata delegate or
fallback reads) it follows the same order so that both paths act as the same
principal.  Modes 3 and 4 are both served by ``DefaultAzureCredential``, which
honours ``AZURE_CLIENT_ID`` and the ``MSI_*`` variables itself.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["AuthConfig", "StaticTokenCredential", "resolve_auth"]

SAS = "sas_token"
ACCESS_TOKEN = "access_token"
MANAGED_IDENTITY = "managed_identity"
AZURE_IDENTITY = "azure_identity"


@dataclass(frozen=True)
class AuthConfig:
    """The authentication mode the DACS client will pick, plus its secret if any."""

    mode: str
    description: str
    secret: str | None = field(default=None, repr=False)

    def adlfs_options(self) -> dict[str, Any]:
        """``storage_options`` for adlfs that authenticate the same way."""
        if self.mode == SAS:
            return {"sas_token": self.secret}
        if self.mode == ACCESS_TOKEN:
            return {"credential": StaticTokenCredential(self.secret or "")}
        return {"anon": False}


class StaticTokenCredential:
    """Async token credential wrapping a pre-fetched bearer token.

    Matches what the DACS client does with ``SI_azBlobAccessToken``: the token
    is used as given and never refreshed.
    """

    def __init__(self, token: str, lifetime_seconds: int = 3600) -> None:
        self._token = token
        self._expires_on = int(time.time()) + lifetime_seconds

    async def get_token(self, *scopes: str, **kwargs: Any) -> Any:
        from azure.core.credentials import AccessToken

        return AccessToken(self._token, self._expires_on)

    async def close(self) -> None:
        return None

    async def __aenter__(self) -> StaticTokenCredential:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


def resolve_auth(env: Mapping[str, str] | None = None) -> AuthConfig:
    """Return the auth mode the DACS client would choose for ``env``."""
    env = os.environ if env is None else env
    if env.get("USE_SAS_TOKEN", "false") == "true" and env.get("STORAGE_INTCPT_SAS_TOKEN"):
        return AuthConfig(
            SAS,
            "SAS token (USE_SAS_TOKEN=true, STORAGE_INTCPT_SAS_TOKEN)",
            env["STORAGE_INTCPT_SAS_TOKEN"].strip(),
        )
    if env.get("SI_azBlobAccessToken"):
        return AuthConfig(ACCESS_TOKEN, "pre-fetched access token (SI_azBlobAccessToken)", env["SI_azBlobAccessToken"])
    if env.get("AZURE_CLIENT_ID") and env.get("MSI_ENDPOINT") and env.get("MSI_SECRET"):
        return AuthConfig(MANAGED_IDENTITY, "managed identity (AZURE_CLIENT_ID, MSI_ENDPOINT, MSI_SECRET)")
    return AuthConfig(AZURE_IDENTITY, "Azure Identity SDK default credential chain")
