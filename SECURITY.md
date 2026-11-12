# Security policy

## Scope

dacs-fsspec is a thin client-side adapter. It does not run a service, store
credentials, or implement authentication: it passes blob addresses to the DACS
Python client and, optionally, to adlfs.

Things that are **out of scope here** and belong upstream:

- Vulnerabilities in DACS itself (cache servers, the native client, the
  `dacs-client` package): report them to the DACS maintainers following
  <https://github.com/Azure/dacs/blob/main/SECURITY.md>.
- Vulnerabilities in fsspec, adlfs or the Azure SDKs: report them to those
  projects.

Please also keep the DACS deployment model in mind. Upstream documents that
DACS currently assumes a single trust domain equal to the whole Kubernetes
cluster and is not suitable for multi-tenant clusters. Using this adapter does
not change that.

## Supported versions

The project is pre-1.0. Only the latest release (and `main`) receive fixes.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub's private reporting instead: on the repository page go to
**Security → Report a vulnerability**
(<https://github.com/cb-Sigma/dacs-fsspec/security/advisories/new>).

Include what you found, how to reproduce it, and the versions of dacs-fsspec,
fsspec and dacs-client involved. You should get an acknowledgement within a
week. This is a volunteer-maintained project, so fix timelines are best
effort; we will tell you what to expect once the report is understood.

## Things worth knowing

- `python -m dacs_fsspec doctor` prints environment variables used by the
  client. Values of secret-bearing variables (`STORAGE_INTCPT_SAS_TOKEN`,
  `SI_azBlobAccessToken`, `MSI_SECRET`) are never printed, only their length.
  Still review the output before pasting it into a public issue: endpoint
  hostnames and client IDs are shown.
- The DACS client writes a per-process configuration file under `/tmp` that
  can contain the SAS or access token it was given. That file is created by
  the client (mode `0600`), not by this package.
- `torch.load` on untrusted files can execute code. `dacs_fsspec.torch.torch_load`
  passes its keyword arguments straight to `torch.load`; keep
  `weights_only=True` for files you do not fully trust.
