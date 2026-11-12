"""Thin PyTorch conveniences on top of :class:`dacs_fsspec.DacsFileSystem`.

Everything here is a few lines over the public filesystem API; use the
filesystem directly when you need more control.  Requires ``torch``
(``pip install 'dacs-fsspec[torch]'``); :func:`load_safetensors` additionally
needs ``safetensors``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError("dacs_fsspec.torch needs PyTorch: pip install 'dacs-fsspec[torch]'") from exc

from .core import DacsFileSystem

__all__ = ["BlobDataset", "load_safetensors", "torch_load"]


def _filesystem(fs: DacsFileSystem | None, storage_options: dict[str, Any] | None) -> DacsFileSystem:
    return fs if fs is not None else DacsFileSystem(**(storage_options or {}))


class BlobDataset(Dataset):
    """Map-style dataset: one item per blob URL.

    ``dataset[i]`` reads the whole blob through the cache and returns
    ``transform(data)`` (raw ``bytes`` without a transform).  If
    ``with_url=True`` the transform is called as ``transform(data, url)``.

    The dataset holds only URLs and options, so it pickles cheaply into
    ``DataLoader`` workers; each worker creates its own DACS client on first
    use.
    """

    def __init__(
        self,
        urls: Sequence[str],
        transform: Callable[..., Any] | None = None,
        with_url: bool = False,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        self.urls = list(urls)
        self.transform = transform
        self.with_url = with_url
        self.storage_options = dict(storage_options or {})

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, index: int) -> Any:
        url = self.urls[index]
        data = DacsFileSystem(**self.storage_options).cat_file(url)
        if self.transform is None:
            return data
        return self.transform(data, url) if self.with_url else self.transform(data)


def torch_load(
    url: str,
    fs: DacsFileSystem | None = None,
    storage_options: dict[str, Any] | None = None,
    **load_kwargs: Any,
) -> Any:
    """``torch.load`` a checkpoint through the cache (streamed with read-ahead).

    Keyword arguments are passed to ``torch.load`` unchanged, so its defaults
    (including ``weights_only``) apply.
    """
    with _filesystem(fs, storage_options).open(url, "rb") as handle:
        return torch.load(handle, **load_kwargs)


def load_safetensors(
    url: str,
    device: str | torch.device = "cpu",
    fs: DacsFileSystem | None = None,
    storage_options: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Load a ``.safetensors`` file through the cache into a dict of tensors.

    The file is read fully into host memory and then deserialised, so peak
    memory is roughly twice the file size.
    """
    try:
        from safetensors.torch import load as load_bytes
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_safetensors needs safetensors: pip install safetensors") from exc

    tensors: dict[str, torch.Tensor] = load_bytes(_filesystem(fs, storage_options).cat_file(url))
    if str(device) != "cpu":
        tensors = {name: tensor.to(device) for name, tensor in tensors.items()}
    return tensors
