"""Shared configuration reader for the tools wrappers.

Both the PascalX and the LDSC wrappers need the same two things: the column names
of a cohort's summary statistics, and the machine paths of the third-party tool
they drive. Neither should be written down twice.

  - Column names come from ``replication.yaml``, which already declares them per
    cohort -- including the allele-direction convention, which differs between the
    Rotterdam dTIF and mTIF runs and silently flips every signed statistic if
    guessed wrong.
  - Tool paths come from ``config.yaml`` under ``genetic_tools``.
  - Cross-cohort trait pairings come from ``replication.yaml``'s
    ``label_mappings``, which is the only place the dTIF/mTIF and UKB/Rotterdam/
    CoLaus name correspondences are written down.

The shell wrappers consume the same source through ``--emit-shell`` and
``--emit-pairs``, so there is no YAML parsing in bash and no second copy of any
name.

Usage from Python::

    from cohort_config import load_cohort, load_tool, load_label_pairs
    cols = load_cohort("RotterdamStudy")["columns"]
    panel = load_tool("pascalx")["reference_panel"]
    pairs = load_label_pairs("ukb_dtif", "ukb_mtif")

Usage from bash::

    eval "$(python3 cohort_config.py --cohort RotterdamStudy --emit-shell)"
    echo "$COL_PVALUE"   # -> P-value

    python3 cohort_config.py --pairs ukb_dtif:ukb_mtif --emit-pairs
    # -> one "<label_a>\t<label_b>" line per trait
"""

import argparse
import os
import sys

import yaml

# tools/ -> repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLICATION_YAML = os.path.join(REPO_ROOT, "replication.yaml")
CONFIG_YAML = os.path.join(REPO_ROOT, "config.yaml")
sys.path.insert(0, str(REPO_ROOT))
from src.config import load_config  # noqa: E402 -- machine paths: config.local.yaml



def _read_yaml(path):
    # config.yaml goes through the loader so genetic_tools picks up config.local.yaml;
    # replication.yaml has no local overlay and reads the same either way.
    # Both config.yaml and replication.yaml carry machine paths that live in a sibling
    # .local.yaml; load_config() merges whichever exists and is a plain read otherwise.
    return load_config(path)


# The UK Biobank discovery set is not a `cohorts:` entry -- a dozen replication
# scripts iterate that list and UKB is what they replicate, not one of them. It is
# still a cohort as far as these wrappers are concerned, so this name resolves to
# `ukb_discovery_cohort:` instead.
UKB_COHORT_NAME = "UKBiobank"


def _ukb_cohort(cfg):
    """Build a cohort entry for the UKB discovery set from ukb_discovery_cohort:.

    Uses `sumstats_columns:`, which names the columns of the full
    *__gwas_sumstats.tsv -- the file LDSC and PascalX read -- rather than the
    top-hit columns. In that file A1 is the NON-effect allele and A2 is the effect
    allele; LDSC's own guess reads it the other way round and inverts the sign of
    every Z.
    """
    entry = cfg.get("ukb_discovery_cohort") or {}
    columns = entry.get("sumstats_columns")
    if not columns:
        raise KeyError(
            "replication.yaml has ukb_discovery_cohort: but no sumstats_columns: "
            "block, so the columns of *__gwas_sumstats.tsv are undeclared. Without "
            "it LDSC guesses, and its guess reverses the effect allele."
        )
    return {
        "name": UKB_COHORT_NAME,
        "type": "discovery",
        "base_dir": entry.get("sumstats_dir") or entry.get("raw_top_hits_dir", ""),
        "sumstats_subdir": "",
        "columns": columns,
    }


def load_cohort(name, path=REPLICATION_YAML):
    """Return the cohort entry from replication.yaml, or raise with the valid names."""
    cfg = _read_yaml(path)
    if name == UKB_COHORT_NAME:
        return _ukb_cohort(cfg)
    cohorts = cfg.get("cohorts", [])
    for cohort in cohorts:
        if cohort.get("name") == name:
            return cohort
    known = ", ".join([c.get("name", "?") for c in cohorts] + [UKB_COHORT_NAME])
    raise KeyError("Unknown cohort %r. Known cohorts: %s" % (name, known))


def _resolve(value):
    """Make a repo-relative value absolute.

    Everything in ``genetic_tools`` is an absolute machine path except the pixi
    environments, which live inside the checkout. A leading ``./`` marks a value as
    repo-relative so that e.g. ``./.pixi/envs/ldsc/bin/python`` keeps working when a
    wrapper is invoked from an unrelated directory.
    """
    if isinstance(value, str) and value.startswith("./"):
        return os.path.join(REPO_ROOT, value[2:])
    return value


def load_tool(name, path=CONFIG_YAML):
    """Return the genetic_tools.<name> block from config.yaml, paths resolved."""
    cfg = _read_yaml(path)
    tools = (cfg or {}).get("genetic_tools", {})
    if name not in tools:
        raise KeyError(
            "config.yaml has no genetic_tools.%s block (found: %s)"
            % (name, ", ".join(sorted(tools)) or "none")
        )
    return {key: _resolve(value) for key, value in (tools[name] or {}).items()}


def _split_label_set(name):
    """"ukb_dtif" -> ("ukb", "dtif"). The feature kind is always the last field."""
    cohort_key, _, kind = name.rpartition("_")
    if kind not in ("dtif", "mtif") or not cohort_key:
        raise KeyError(
            "Label set %r is not of the form <cohort>_<dtif|mtif> "
            "(e.g. ukb_dtif, rotterdam_mtif)" % name
        )
    return cohort_key, kind


def load_label_pairs(set_a, set_b, path=REPLICATION_YAML):
    """Return [(label_a, label_b), ...] pairing two label sets trait by trait.

    A label set is named ``<cohort>_<dtif|mtif>`` -- ``ukb_dtif``,
    ``rotterdam_mtif``, ... -- and resolves to one column of
    ``replication.yaml``'s ``label_mappings``. The pairing is positional: entry i
    of ``label_mappings.dtif`` and entry i of ``label_mappings.mtif`` are the same
    retinal trait, which is what makes a dTIF-vs-mTIF genetic correlation a
    comparison of like with like rather than a cross product.
    """
    cfg = _read_yaml(path)
    mappings = (cfg or {}).get("label_mappings", {}) or {}

    cohort_a, kind_a = _split_label_set(set_a)
    cohort_b, kind_b = _split_label_set(set_b)

    entries_a = mappings.get(kind_a) or []
    entries_b = mappings.get(kind_b) or []
    if not entries_a or not entries_b:
        raise KeyError(
            "replication.yaml has no label_mappings.%s / label_mappings.%s"
            % (kind_a, kind_b)
        )
    if len(entries_a) != len(entries_b):
        raise ValueError(
            "label_mappings.%s has %d entries but label_mappings.%s has %d -- the "
            "positional pairing they encode is broken"
            % (kind_a, len(entries_a), kind_b, len(entries_b))
        )

    pairs = []
    for index, (entry_a, entry_b) in enumerate(zip(entries_a, entries_b)):
        label_a, label_b = entry_a.get(cohort_a), entry_b.get(cohort_b)
        for label, cohort, kind in ((label_a, cohort_a, kind_a), (label_b, cohort_b, kind_b)):
            if not label:
                raise KeyError(
                    "label_mappings.%s[%d] has no '%s' label (known: %s)"
                    % (kind, index, cohort, ", ".join(sorted(entry_a)) or "none")
                )
        pairs.append((label_a, label_b))
    return pairs


def emit_pairs(set_a, set_b):
    """Print one tab-separated label pair per line, for `read` in a shell wrapper."""
    for label_a, label_b in load_label_pairs(set_a, set_b):
        print("%s\t%s" % (label_a, label_b))


def _shell_quote(value):
    # A missing value becomes the empty string, never the four characters "None". Since
    # the machine paths moved to config.local.yaml, an absent key is the NORMAL case on a
    # public checkout, and `str(None)` would hand the shell a directory named None that
    # every downstream `-d` test would report as simply missing.
    if value is None:
        value = ""
    return "'" + str(value).replace("'", "'\\''") + "'"


def emit_shell(cohort_name, tool_name):
    """Print KEY=value lines for `eval` in a shell wrapper.

    Column names become COL_<KEY>; tool settings become <TOOL>_<KEY>. Absent or
    null values are emitted as the empty string so callers can test with -n/-z.
    """
    lines = []
    if cohort_name:
        cohort = load_cohort(cohort_name)
        lines.append("COHORT_NAME=%s" % _shell_quote(cohort.get("name", "")))
        lines.append("COHORT_TYPE=%s" % _shell_quote(cohort.get("type", "")))
        lines.append("COHORT_BASE_DIR=%s" % _shell_quote(cohort.get("base_dir", "")))
        lines.append(
            "COHORT_SUMSTATS_SUBDIR=%s" % _shell_quote(cohort.get("sumstats_subdir", ""))
        )
        for key, value in sorted((cohort.get("columns") or {}).items()):
            lines.append(
                "COL_%s=%s" % (key.upper(), _shell_quote("" if value is None else value))
            )
    if tool_name:
        for key, value in sorted(load_tool(tool_name).items()):
            lines.append(
                "%s_%s=%s"
                % (tool_name.upper(), key.upper(), _shell_quote("" if value is None else value))
            )
    print("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", help="cohort name as it appears in replication.yaml")
    parser.add_argument("--tool", choices=["pascalx", "ldsc"], help="genetic_tools block")
    parser.add_argument(
        "--pairs",
        metavar="SET_A:SET_B",
        help="two label sets to pair trait by trait, e.g. ukb_dtif:rotterdam_dtif",
    )
    parser.add_argument(
        "--emit-shell",
        action="store_true",
        help="print KEY=value lines suitable for `eval` in a shell wrapper",
    )
    parser.add_argument(
        "--emit-pairs",
        action="store_true",
        help="print tab-separated label pairs, one per line (requires --pairs)",
    )
    args = parser.parse_args(argv)

    if args.emit_pairs:
        if not args.pairs or ":" not in args.pairs:
            parser.error("--emit-pairs needs --pairs SET_A:SET_B")
        set_a, _, set_b = args.pairs.partition(":")
        emit_pairs(set_a, set_b)
        return 0

    if not args.cohort and not args.tool:
        parser.error("give at least one of --cohort / --tool")
    if not args.emit_shell:
        parser.error("give an output mode: --emit-shell or --emit-pairs")

    emit_shell(args.cohort, args.tool)
    return 0


if __name__ == "__main__":
    sys.exit(main())
