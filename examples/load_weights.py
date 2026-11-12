"""Load model weights through the DACS cache.

python examples/load_weights.py dacs://account/container/models/model.safetensors
python examples/load_weights.py dacs://account/container/models/checkpoint.pt
"""

from __future__ import annotations

import sys

from dacs_fsspec.torch import load_safetensors, torch_load

url = sys.argv[1]

if url.endswith(".safetensors"):
    state = load_safetensors(url, device="cpu")
else:
    state = torch_load(url, map_location="cpu", weights_only=True)

for name, tensor in list(state.items())[:10]:
    print(f"{name:60s} {tuple(tensor.shape)}")
print(f"{len(state)} tensors")
