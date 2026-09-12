"""Fetch deposited input data by DOI.

Figure scripts should not hardcode paths into someone else's filesystem. They ask
for a *named source* declared in ``config.yaml`` and get back local paths::

    from src.fetch_data import fetch
    paths = fetch("figure_intermediates")
    df = pd.read_csv(paths["fig2_tif_vs_lv.csv"])

The name resolves to a Zenodo **concept DOI**, which always points at the newest
version of a deposit. Files are cached on disk and verified against the MD5 the
Zenodo API reports, so a second run is offline and free.

Tracking "latest" is a deliberate choice: the deposit backing this paper is not
expected to change, and one stable DOI in the Data Availability statement is worth
more than byte-pinning. The resolved *version* DOI is printed on every fetch, so a
run's provenance still ends up in the log even though it is not enforced.

Stdlib only (plus PyYAML for the config), so the released hub has one less thing
that can break.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml

ZENODO_API = "https://zenodo.org/api/records"
SANDBOX_API = "https://sandbox.zenodo.org/api/records"
SANDBOX_PREFIX = "10.5072"  # Zenodo sandbox issues DOIs under this prefix
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_CACHE = REPO_ROOT / "data" / "external"
TIMEOUT = 60
CHUNK = 1 << 20  # 1 MiB


class FetchError(RuntimeError):
    """Raised when a source cannot be resolved or a download fails verification."""


# --------------------------------------------------------------------------- config


def _load_config() -> dict:
    # Through the loader, like every other config read in the repository: data_sources is
    # public, but a second way of reading config.yaml is how the two drift apart.
    from src.config import load_config
    return load_config(CONFIG_PATH)


def _cache_dir(config: dict) -> Path:
    """Cache location: $MVP_DATA_CACHE, else config, else <repo>/data/external."""
    env = os.environ.get("MVP_DATA_CACHE")
    if env:
        return Path(env).expanduser()
    configured = (config.get("data_sources") or {}).get("cache_dir")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path
    return DEFAULT_CACHE


def _source_spec(config: dict, name: str) -> dict:
    sources = (config.get("data_sources") or {}).get("sources") or {}
    if name not in sources:
        known = ", ".join(sorted(sources)) or "none declared"
        raise FetchError(
            "unknown data source {!r}; declared in config.yaml: {}".format(name, known)
        )
    return sources[name]


# --------------------------------------------------------------------------- zenodo


def _bare_doi(doi: str) -> str:
    """Strip any resolver prefix, leaving '<prefix>/zenodo.<id>' or a bare id."""
    text = doi.strip().rstrip("/")
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.lower().startswith(prefix):
            return text[len(prefix) :]
    return text


def record_id_from_doi(doi: str) -> str:
    """Extract the Zenodo record id from a DOI, DOI URL, or bare id.

    Works for both concept and version DOIs -- Zenodo redirects a concept id to
    the latest version server-side, so the caller does not need to know which
    kind it holds.
    """
    text = _bare_doi(doi)
    if "zenodo." in text:
        text = text.rsplit("zenodo.", 1)[1]
    if not text.isdigit():
        raise FetchError(
            "cannot read a Zenodo record id out of {!r}; expected something like "
            "'10.5281/zenodo.12345678'".format(doi)
        )
    return text


def api_base(doi: str) -> str:
    """Pick the production or sandbox API from the DOI prefix.

    Sandbox deposits get 10.5072 DOIs and live on a different host, so a dry run
    against sandbox.zenodo.org needs no config change -- paste the sandbox DOI into
    config.yaml and everything else works unmodified.
    """
    return SANDBOX_API if _bare_doi(doi).startswith(SANDBOX_PREFIX) else ZENODO_API


def resolve_record(doi: str) -> dict:
    """Resolve a (concept) DOI to the latest published record's API JSON."""
    import json

    recid = record_id_from_doi(doi)
    url = "{}/{}".format(api_base(doi), recid)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        # urlopen follows the concept -> latest-version redirect for us.
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            record = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise FetchError(
                "Zenodo has no record {} (from DOI {!r}). If the deposit is still "
                "a draft, publish it or use a reserved DOI.".format(recid, doi)
            )
        if error.code in (401, 403):
            raise FetchError(
                "Zenodo record {} is restricted or embargoed; automated fetching "
                "only works for open records.".format(recid)
            )
        raise FetchError("Zenodo API error {} for {}".format(error.code, url))
    except urllib.error.URLError as error:
        raise FetchError("could not reach Zenodo ({}): {}".format(url, error.reason))
    return record


def record_files(record: dict) -> Dict[str, dict]:
    """Map filename -> {url, size, md5} for a resolved record."""
    entries = record.get("files") or []
    if not entries:
        raise FetchError(
            "Zenodo record {} lists no downloadable files (restricted access, or an "
            "empty deposit).".format(record.get("id"))
        )
    files = {}
    for entry in entries:
        checksum = entry.get("checksum", "")
        files[entry["key"]] = {
            "url": entry["links"]["self"],
            "size": entry.get("size"),
            "md5": checksum.split("md5:", 1)[1] if checksum.startswith("md5:") else None,
        }
    return files


def describe(record: dict) -> str:
    metadata = record.get("metadata") or {}
    return "record {} (DOI {}, version {}, published {})".format(
        record.get("id"),
        record.get("doi"),
        metadata.get("version") or "unversioned",
        metadata.get("publication_date") or "?",
    )


# ------------------------------------------------------------------------ transfer


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _human(size: Optional[int]) -> str:
    if not size:
        return "?"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return "{:.0f} {}".format(value, unit) if unit == "B" else "{:.1f} {}".format(value, unit)
        value /= 1024
    return "{:.1f} TB".format(value)


def _download(url: str, dest: Path, expected_md5: Optional[str], size: Optional[int]) -> None:
    """Stream to a .part file, verify, then move into place atomically."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    show_progress = sys.stderr.isatty() and bool(size)
    downloaded = 0
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response, open(partial, "wb") as handle:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if show_progress:
                    sys.stderr.write(
                        "\r    {:5.1f}%  {}".format(100 * downloaded / size, _human(downloaded))
                    )
                    sys.stderr.flush()
    except (urllib.error.URLError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise FetchError("download failed for {}: {}".format(dest.name, error))
    finally:
        if show_progress:
            sys.stderr.write("\r" + " " * 40 + "\r")

    if expected_md5:
        actual = _md5(partial)
        if actual != expected_md5:
            partial.unlink(missing_ok=True)
            raise FetchError(
                "checksum mismatch for {} (expected md5 {}, got {}); the download was "
                "truncated or corrupted".format(dest.name, expected_md5, actual)
            )
    shutil.move(str(partial), str(dest))


# Records the md5 each cached file was downloaded with, so a new deposit version can be
# recognised without re-hashing the cache on every run.
INDEX_NAME = ".fetch_index.json"


def _read_index(cache_dir: Path) -> Dict[str, str]:
    try:
        with open(str(cache_dir / INDEX_NAME)) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _write_index(cache_dir: Path, index: Dict[str, str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(str(cache_dir / INDEX_NAME), "w") as handle:
            json.dump(index, handle, indent=1, sort_keys=True)
    except OSError:
        pass  # a read-only cache still works, it just re-verifies by hashing


def _is_cached(dest: Path, expected_md5: Optional[str], verify: bool, recorded_md5=None) -> bool:
    """Is the cached copy the file the record currently points at?

    Existence alone is NOT enough. Zenodo records are immutable, but a *concept* DOI
    resolves to the newest version, so the same filename legitimately gains new content
    when a new version is published -- and a stale copy would then be used silently, which
    is a wrong-figure bug, not a slow one. So the md5 the file was downloaded with is
    compared against the md5 the record advertises. Files cached before the index existed
    have no recorded md5 and get hashed once, which then records them.
    """
    if not dest.exists():
        return False
    if not expected_md5:
        return True
    known = recorded_md5 if recorded_md5 is not None else _md5(dest)
    if known != expected_md5:
        return False
    # The index says it matched at download time; verify re-reads the bytes to catch
    # corruption or edits made to the cache since.
    if verify and _md5(dest) != expected_md5:
        return False
    return True


# ---------------------------------------------------------------------------- api


def _fetch_local(
    local_dir: Path,
    source: str,
    files: Optional[Iterable[str]],
    quiet: bool,
) -> Dict[str, Path]:
    """Serve the requested files from a local directory of freshly built intermediates.

    Used only when $MVP_INTERMEDIATES_DIR is set; see fetch() for why that exists.
    Searches the directory recursively, because the producers write into per-figure
    subdirectories, one per figure.
    """
    if not local_dir.is_dir():
        raise FetchError(
            "$MVP_INTERMEDIATES_DIR is set to {}, which is not a directory".format(local_dir)
        )
    if files is None:
        raise FetchError(
            "$MVP_INTERMEDIATES_DIR needs an explicit file list; a local directory has no "
            "record to enumerate"
        )

    index = {}
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            index.setdefault(path.name, path)

    resolved, remaining = {}, []
    for name in files:
        if name in index:
            resolved[name] = index[name]
        else:
            remaining.append(name)

    if not quiet and resolved:
        print("  LOCAL OVERRIDE: {} file(s) for {} taken from {}".format(
            len(resolved), source, local_dir))
        for name, path in resolved.items():
            print("    {}  <-  {}".format(name, path))
        if remaining:
            print("  the remaining {} file(s) still come from the deposit".format(
                len(remaining)))
    return resolved, remaining


def fetch(
    source: str,
    files: Optional[Iterable[str]] = None,
    force: bool = False,
    verify_cached: bool = False,
    quiet: bool = False,
) -> Dict[str, Path]:
    """Fetch a named data source, returning {filename: local path}.

    Parameters
    ----------
    source
        Key under ``data_sources.sources`` in ``config.yaml``.
    files
        Optional subset of filenames; default fetches everything in the record.
    force
        Re-download even if a cached copy exists.
    verify_cached
        Re-hash cached files before trusting them. Off by default -- that is a
        full read of every file, which is slow for multi-GB deposits.

    Local override
    --------------
    Setting ``$MVP_INTERMEDIATES_DIR`` makes this read the named files straight from
    that directory instead of resolving the DOI. That is the development loop: a stage
    00-04 producer writes fresh intermediates, and a 05_figures script must be able to
    build from them BEFORE they are deposited -- otherwise the only way to see a
    regenerated panel is to publish a Zenodo version first. Files not present locally
    still come from the deposit, so a partially regenerated set works.

    This is deliberately an environment variable and not a config key: it is a
    per-invocation development affordance, and it must never be the state a reader of
    the public repository silently inherits. Every override is printed.
    """
    config = _load_config()
    spec = _source_spec(config, source)

    local_dir = os.environ.get("MVP_INTERMEDIATES_DIR")
    local_hits: Dict[str, Path] = {}
    if local_dir:
        local_hits, remaining = _fetch_local(
            Path(local_dir).expanduser(), source, files, quiet
        )
        if not remaining:
            return local_hits
        files = remaining

    kind = spec.get("kind", "zenodo")
    if kind != "zenodo":
        raise FetchError(
            "source {!r} has kind {!r}; only 'zenodo' is implemented".format(source, kind)
        )

    doi = spec.get("concept_doi") or spec.get("doi")
    if not doi:
        raise FetchError("source {!r} declares no concept_doi in config.yaml".format(source))
    if "XXXX" in doi.upper():
        raise FetchError(
            "source {!r} still has a placeholder DOI ({}); fill it in once the Zenodo "
            "deposit is published".format(source, doi)
        )

    record = resolve_record(doi)
    available = record_files(record)
    wanted = list(files) if files is not None else list(available)

    missing = [name for name in wanted if name not in available]
    if missing:
        raise FetchError(
            "{} not in Zenodo {}; it holds: {}".format(
                ", ".join(missing), describe(record), ", ".join(sorted(available))
            )
        )

    target_dir = _cache_dir(config) / source
    if not quiet:
        print("[fetch_data] {} -> Zenodo {}".format(source, describe(record)))
        print("[fetch_data] cache: {}".format(target_dir))

    index = _read_index(target_dir)

    resolved = {}
    for name in wanted:
        info = available[name]
        dest = target_dir / name
        if not force and _is_cached(dest, info["md5"], verify_cached, index.get(name)):
            if not quiet:
                print("  cached  {} ({})".format(name, _human(info["size"])))
        else:
            if not quiet:
                print("  get     {} ({})".format(name, _human(info["size"])))
            _download(info["url"], dest, info["md5"], info["size"])
        if info["md5"]:
            index[name] = info["md5"]
        resolved[name] = dest

    _write_index(target_dir, index)
    # Locally overridden files win over their deposited namesakes.
    resolved.update(local_hits)
    return resolved


def fetch_file(source: str, name: str, **kwargs) -> Path:
    """Fetch a single file from a source and return its local path."""
    return fetch(source, files=[name], **kwargs)[name]


# ---------------------------------------------------------------------------- cli


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch deposited input data by DOI (see config.yaml: data_sources)."
    )
    parser.add_argument("source", nargs="?", help="named source; omit to list all")
    parser.add_argument("--list", action="store_true", help="show record contents, download nothing")
    parser.add_argument("--file", action="append", dest="files", help="fetch only this file (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--verify-cached", action="store_true", help="re-hash cached files")
    parser.add_argument(
        "--paths",
        action="store_true",
        help="fetch, then print 'filename<TAB>path' on stdout and nothing else "
        "(for consumption by other languages, e.g. 05_figures/lib/fetch_data.R)",
    )
    args = parser.parse_args(argv)

    config = _load_config()
    sources = (config.get("data_sources") or {}).get("sources") or {}

    if not args.source:
        if not sources:
            print("No data sources declared in config.yaml.")
            return 1
        print("Declared data sources (cache: {}):\n".format(_cache_dir(config)))
        for name, spec in sorted(sources.items()):
            print("  {:24s} {}".format(name, spec.get("concept_doi", "?")))
            if spec.get("description"):
                print("  {:24s} {}".format("", spec["description"]))
        return 0

    try:
        if args.list:
            spec = _source_spec(config, args.source)
            record = resolve_record(spec.get("concept_doi") or spec.get("doi"))
            print("{}: Zenodo {}".format(args.source, describe(record)))
            print("  {}".format(record.get("links", {}).get("self_html", "")))
            for name, info in sorted(record_files(record).items()):
                print("  {:50s} {:>10s}".format(name, _human(info["size"])))
        elif args.paths:
            # Progress goes to stderr so stdout stays a clean machine-readable table.
            resolved = fetch(
                args.source,
                files=args.files,
                force=args.force,
                verify_cached=args.verify_cached,
                quiet=True,
            )
            for name in sorted(resolved):
                print("{}\t{}".format(name, resolved[name]))
        else:
            fetch(
                args.source,
                files=args.files,
                force=args.force,
                verify_cached=args.verify_cached,
            )
    except FetchError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
