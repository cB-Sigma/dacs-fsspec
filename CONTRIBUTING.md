# Contributing

Thanks for considering a contribution. This is a small, volunteer-maintained
project; clear bug reports and small focused pull requests are the easiest to
act on.

## The most valuable thing you can do

Run the package against a **real DACS cluster** and tell us what happened. The
maintainers have not been able to do that yet (see "How this is tested, and
what is not" in the README).

```bash
pip install -e ".[dev,adlfs,client]"
python -m dacs_fsspec doctor --probe dacs://account/container/some/blob

DACS_FSSPEC_TEST_URL=dacs://account/container/some/blob \
DACS_FSSPEC_TEST_WRITE_PREFIX=dacs://account/container/scratch \
pytest -m integration -v
```

Open an issue with the output (review `doctor` output for anything you do not
want public first), pass or fail.

## Development setup

```bash
git clone https://github.com/cb-Sigma/dacs-fsspec
cd dacs-fsspec
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,adlfs]"
# On Linux with glibc >= 2.35 you can also install the real client:
pip install dacs-client
```

Run everything CI runs:

```bash
ruff check .
ruff format --check .
mypy
pytest
python -m build && twine check dist/*
```

Coverage: `coverage run -m pytest && coverage report`. (Prefer this over
`pytest --cov`: the test suite's `conftest.py` imports the package before the
pytest-cov plugin starts measuring, which under-reports.)

## How the tests are organised

| Tests | Need |
| --- | --- |
| Most of `tests/` | Nothing but fsspec. They run against `dacs_fsspec.testing.FakeDacs`, an in-memory client with the same ctypes-buffer interface and error style as the real one. |
| `tests/test_real_client.py` | `dacs-client` importable (Linux). No cluster: it introspects the real classes to catch API drift. Skipped elsewhere. |
| `tests/test_fsspec_integration.py`, `tests/test_ecosystem.py`, `tests/test_torch.py` | pandas, pyarrow, dask, datasets, zarr, webdataset, torch, safetensors. Each test is skipped if its library is missing. |
| `tests/test_live_integration.py` | A live cluster. Marked `integration`, skipped unless `DACS_FSSPEC_TEST_URL` is set. |

If you change how the adapter talks to the client, update the fake in
`src/dacs_fsspec/testing.py` **only** to match behaviour you have confirmed
from the client's source or from a real cluster, and say which in the pull
request. The fake encoding a wrong assumption is the main way this project
can be confidently wrong.

## Guidelines

- Keep the adapter thin. Behaviour the DACS client does not have should raise
  `NotImplementedError` with a helpful message, not be emulated in ways that
  surprise people (no read-modify-write "append", no copy-then-delete "move").
- Never import `py_dacs_client`, `adlfs` or `torch` at module import time in
  the core package; they are optional.
- The client must stay behind the protocols in `dacs_fsspec/client.py`, and
  must never be pickled or shared across processes.
- Be honest in documentation about what has and has not been tested.
- Add a test for every behaviour change and a line to `CHANGELOG.md`.
- Type hints everywhere; `mypy --strict` must pass.

## Pull requests

1. Open an issue first for anything larger than a bug fix, so we can agree on
   the approach.
2. Keep commits focused; describe what and why.
3. Make sure the checks above pass locally.

By contributing you agree that your contributions are licensed under the
Apache License 2.0, the project's license.

## Releasing (maintainers)

1. Update `src/dacs_fsspec/_version.py` and `CHANGELOG.md`.
2. Tag `vX.Y.Z` and push the tag.
3. The `release` workflow builds the sdist and wheel, checks that the tag
   matches the package version, and publishes to PyPI with trusted
   publishing. This requires a `pypi` environment in the repository settings
   and a matching trusted publisher configured on PyPI.
