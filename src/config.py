"""Read a project YAML: ``<name>.yaml``, overlaid by ``<name>.local.yaml``.

Used for the two configuration files that carry machine paths -- ``config.yaml`` and
``replication.yaml``. The rule is the file name, so a third one needs no new code.

WHY THERE ARE TWO FILES. ``config.yaml`` carries everything that is *about the study* --
palettes, page geometry, the type band, the multiple-testing burden, the DOIs the figure
scripts fetch through -- and it is published with the code. It used to carry, in the same
breath, 35 absolute paths into ``/HDD``, ``/NVME`` and ``/SSD``, plus a home directory.
Those are facts about this machine, they are useless to a reader, and publishing them maps
the group's storage for anyone who reads the repository. They now live in
``config.local.yaml``, which is gitignored and never leaves this box.

The split is a MERGE, not a fork: the local file holds the same key paths with real values
where the public file holds ``null``. Nothing is duplicated and nothing can drift, because
neither file is a copy of the other -- one supplies the values the other declares.

    from src.config import load_config, require_path

    config = load_config()
    scores = require_path(config, "figure_data.gene_scores_dir")

WHAT A PUBLIC CHECKOUT SEES. ``config.local.yaml`` is absent, every private leaf is
``None``, and a script that needs one stops with a message naming the key and saying where
the data actually lives. That is the intended experience, not a degraded one: the figure
tier reads its inputs from the deposit by DOI and touches none of these keys, so a reader
who hits this message is running something that genuinely needs cohort access.

DO NOT go back to ``yaml.safe_load(open("config.yaml"))``. It still parses, and it will
hand back ``None`` for every private leaf without a word -- which is how a script ends up
writing to ``None/output.csv`` or, worse, quietly reading nothing and reporting zero rows.
Every reader of a private block goes through this module.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
LOCAL_PATH = REPO_ROOT / "config.local.yaml"

# The env var exists for one reason: running a script against a staged public tree, or
# against a second checkout, without editing anything. It is not a profile mechanism --
# `profiles:` inside the config is that.
LOCAL_ENV = "MVP_CONFIG_LOCAL"


def _read(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """`overlay` wins, recursively, except that an explicit null never overwrites a value.

    The asymmetry is deliberate. A merge is applied one way here -- local over public --
    and the public file's job is to DECLARE the keys, with `null` standing for "this is a
    machine path, look in the local file". If null overwrote, a key present in both files
    would blank itself depending on which side happened to carry the null.
    """
    result = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        elif _named_list(value) and _named_list(result.get(key)):
            result[key] = _merge_named(result[key], value)
        elif value is not None or key not in result:
            result[key] = copy.deepcopy(value)
    return result


def _named_list(value):
    """A list whose every item is a dict carrying a `name` -- `replication.yaml: cohorts`."""
    return (isinstance(value, list) and value
            and all(isinstance(item, dict) and "name" in item for item in value))


def _merge_named(base, overlay):
    """Merge two lists of named dicts BY NAME, never by position.

    replication.yaml's `cohorts:` is a list, and the local overlay carries one entry per
    cohort holding just its `base_dir`. Position matching would work today and be a data
    corruption tomorrow: add a cohort to the public file, forget the local one, and every
    overlay below the insertion point lands on the wrong cohort -- CoLaus reading the
    Rotterdam directory, silently, with plausible-looking output. Matching on `name` cannot
    do that; an overlay whose name is not in the public list is appended instead.
    """
    merged = [copy.deepcopy(item) for item in base]
    index = {item["name"]: position for position, item in enumerate(merged)}
    for item in overlay:
        if item["name"] in index:
            merged[index[item["name"]]] = deep_merge(merged[index[item["name"]]], item)
        else:
            merged.append(copy.deepcopy(item))
    return merged


def load_config(path: Optional[Path] = None, local: Optional[Path] = None) -> Dict[str, Any]:
    """The merged configuration. Missing local file is normal, not an error."""
    path = Path(path) if path else CONFIG_PATH
    config = _read(path)
    if local is None:
        env = os.environ.get(LOCAL_ENV)
        # Next to the config that was actually read, not next to this module. A script
        # given `--config some/other/config.yaml` -- a staged public tree, a second
        # checkout -- must not silently pick up this machine's local overlay.
        local = Path(env) if env else path.with_name(path.stem + ".local" + path.suffix)
    local = Path(local)
    if local.exists():
        config = deep_merge(config, _read(local))
    return config


def get(config: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """`get(cfg, "genetic_tools.pascalx.reference_panel")`, without the KeyError dance."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def require_path(config: Dict[str, Any], dotted: str) -> Path:
    """Return a configured path, or stop with a message a stranger can act on.

    The failure this replaces is the one worth naming: `None` flowing into `Path()` or into
    an f-string, and a script writing `None/thing.csv` or reading an empty directory and
    reporting zero rows. A missing machine path is not a bug to be tolerated three function
    calls later; it is the end of the run.
    """
    value = get(config, dotted)
    if value:
        return Path(value)
    raise SystemExit(
        f"{dotted} is not configured.\n"
        f"  It is a machine path, so it lives in {LOCAL_PATH.name} (gitignored), not in "
        f"config.yaml.\n"
        f"  On a public checkout that file does not exist and this key has no value: the "
        f"script you ran needs cohort-level data rather than the published deposit. The "
        f"figure scripts under 05_figures/ fetch their inputs by DOI and need none of "
        f"these keys.\n"
        f"  On our machines: add {dotted} to {LOCAL_PATH}."
    )
