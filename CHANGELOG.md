# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it
reaches 1.0. Until then, minor versions may contain breaking changes.

## [Unreleased]

## [0.1.0] - not yet released

First public version. Not validated against a live DACS cluster; see
"How this is tested, and what is not" in the README.

### Added

- `DacsFileSystem`, an fsspec filesystem registered as `dacs://` through the
  `fsspec.specs` entry point. Reads (`open`, `cat_file`, `cat_ranges`,
  `get_file`) go through the DACS Python client.
- `DacsFile`, a buffered file object with read-ahead over a single native
  handle, and streaming writes committed on close.
- Metadata backends: the DACS client's own `stat`/`list_dir` (default) or
  Azure Blob Storage through adlfs (`metadata="adlfs"`); custom backends via
  the `MetadataBackend` protocol.
- Opt-in writes mapped to the client's write modes (`sync`, `blob_only`,
  `cache_only`, `cache_blob_async`) and `consistent_mode` for overwrites.
- Opt-in fallback to direct Azure Blob reads when DACS is unavailable
  (`fallback=True` or `DACS_FSSPEC_FALLBACK=1`), with a single
  `DacsFallbackWarning` and a retry window.
- Per-process lazy client construction; the filesystem and read-mode files are
  pickle-safe and usable from forked or spawned workers.
- URL helpers: `BlobPath`, `from_blobfile_url`, `from_adlfs_url`, `to_dacs_url`.
- `dacs_fsspec.torch`: `BlobDataset`, `torch_load`, `load_safetensors`.
- `dacs_fsspec.testing`: an in-memory fake client for cluster-free tests.
- Command line interface (`python -m dacs_fsspec` / `dacs-fsspec`) with `ls`,
  `stat`, `cat`, `cp` and `doctor`.

[Unreleased]: https://github.com/cb-Sigma/dacs-fsspec/commits/main
[0.1.0]: https://github.com/cb-Sigma/dacs-fsspec/commits/main
