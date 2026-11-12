## What and why

<!-- What does this change, and what problem does it solve? Link the issue if there is one. -->

## How it was tested

<!-- Be specific about the level of evidence. -->

- [ ] Unit tests against the in-memory fake (`pytest`)
- [ ] Real `dacs-client` wheel importable (`tests/test_real_client.py` ran, not skipped)
- [ ] Live DACS cluster (`pytest -m integration`), describe the setup below
- [ ] Not tested beyond static checks (say why)

## Checklist

- [ ] `ruff check .`, `ruff format --check .` and `mypy` pass
- [ ] Tests added or updated
- [ ] `CHANGELOG.md` updated for user-visible changes
- [ ] README updated if behaviour, options or limitations changed
- [ ] If `src/dacs_fsspec/testing.py` changed: the new fake behaviour is confirmed from the client source or a real cluster (say which)
