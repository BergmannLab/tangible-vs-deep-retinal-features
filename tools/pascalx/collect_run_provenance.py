#!/usr/bin/env python3
"""
collect_run_provenance.py

Collect the settings and the logs of every PascalX scoring run into the two
provenance artefacts of the deposit.

    pixi run python tools/pascalx/collect_run_provenance.py
    pixi run python tools/pascalx/collect_run_provenance.py --check

WHY. The scored values were deposited before the settings that produced them, and the
settings are not one set: gene and pathway scoring differ, the two passes differ from
each other (8 CPUs in the first, 4 and a niceness in the second), and the second pass
did NOT use the gene-set file its own config names. Its `.config.json` says
msigdb.v7.2.symbols.gmt while `<trait>__pathway_scores.pathway_file_used.txt` records
the per-trait `__within_BH.gmt` it actually read -- the config alone would misdescribe
the run it came from. Both are captured here, and the file actually used wins.

The settings also survive in three different formats across the runs: a python
`.config` module (first pass), a `.config.json` (second), and neither for some runs.
Reading all of them into one table is the point -- a reader should not have to know
which generation a given trait belongs to.

WHAT GOES IN THE LOG ARCHIVE, AND WHAT DOES NOT. Every `.out` and `.err` beside a
scoring output, with absolute paths reduced to their basenames. The logs carry real
provenance -- the annotation build, the variant count, which gene-score file was
reused, PascalX's own warnings -- but they also carry the operator's home directory
and this machine's volume layout, which the deposit has no reason to publish. Nothing
else is altered: timings, counts, versions and warnings are all preserved verbatim.

Outputs (deposited with the public data record):
  scoring_settings.csv   one row per scoring run, settings and log facts
  scoring_logs.tgz         the scrubbed .out/.err of every run

The archive is `.tgz` and not `.tar.gz` deliberately: the Zenodo assembler treats any
`*.gz` as a table to scan for participant identifiers, and would try to read a tar
stream as CSV.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.config import load_config  # noqa: E402 -- machine paths: config.local.yaml

DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "natgen_revision" / "intermediate" / "scoring_provenance"

PROVENANCE_FILE = "scoring_settings.csv"
LOG_ARCHIVE = "scoring_logs.tgz"

# Which profile is which pass. The full first-pass screen lives in the preprint
# profile's directories; the active profile holds only the recomputation.
PASSES = [("screen", "preprint"), ("recomputed", None)]

# Settings worth carrying. Everything PascalX's behaviour depends on, plus the sumstats
# column mapping, which is what a reader would need to re-run from the deposited GWAS.
SETTING_KEYS = [
    "GPU", "N_CPU", "NICENESS", "METHOD", "AUTORESCORE", "VARCUTOFF", "MAF", "WINDOW",
    "GENETYPE", "MODE", "N_SAMPLINGS", "FUSE", "MERGEDIST",
    "RSCOL", "PCOL", "BCOL", "DELIMITER", "REFERENCE_COL", "ALTERNATE_COL",
]
# Paths are recorded by basename only -- see the module docstring.
PATH_KEYS = ["REFERENCE_PANEL", "PATHWAYS"]

_ABS_PATH = re.compile(r"/(?:[\w.@+~-]+/)*[\w.@+~-]+")
_LV = re.compile(r"^LV_(\d+)$")


def scrub(text: str) -> str:
    """Absolute paths -> their last component. Everything else untouched."""
    return _ABS_PATH.sub(lambda m: m.group(0).rsplit("/", 1)[-1] or "/", text)


def read_config(path: Path) -> dict:
    """Settings from a `.config.json`, or from the python `.config` module beside it.

    The python form is parsed with `ast`, never executed: it is a config file from an
    earlier generation of the pipeline, it imports socket and branches on the hostname,
    and running it to read six constants would be both unnecessary and unsafe. Only
    module-level assignments of plain literals are taken, which is exactly the block
    that carries the PascalX parameters.
    """
    if path.suffix == ".json":
        with open(str(path), encoding="utf-8") as handle:
            return json.load(handle)

    values = {}
    tree = ast.parse(open(str(path), encoding="utf-8").read())
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
            continue
        except (ValueError, SyntaxError):
            pass
        # `PATHWAYS = PATHWAY_PATH + "msigdb.v7.2.symbols.gmt"`. PATHWAY_PATH is
        # assigned inside a hostname branch, so it is not resolvable here and does not
        # need to be: only the basename is recorded, and that is the literal half.
        if (isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add)
                and isinstance(node.value.right, ast.Constant)
                and isinstance(node.value.right.value, str)):
            values[target.id] = node.value.right.value
    return values


def parse_log(path: Path) -> dict:
    """The facts a scoring log states about its own run."""
    facts = {}
    if not path.is_file():
        return facts
    text = path.read_text(errors="replace")
    patterns = {
        "log_started": r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
        "n_active_genes": r"(\d+) active genes",
        "n_snps_loaded": r"(\d+) SNPs loaded",
        "n_gene_scores_loaded": r"(\d+) scores loaded",
        "n_modules_loaded": r"(\d+) modules loaded",
        "genome_build": r"\[ [^]]* \] \( (\w+) \)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            facts[key] = match.group(1)
    return facts


def count_rows(path: Path) -> int | None:
    if not path.is_file():
        return None
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(str(path), "rt", errors="replace") as handle:
        return sum(1 for _ in handle)


def feature_set_of(stem: str, labels: list) -> tuple:
    """`bifurcations_pred` -> ('dtif', 'bifurcations'); `LV_7` -> ('lv', 'LV_7')."""
    if _LV.match(stem):
        return "lv", stem
    for label in labels:
        for suffix, name in (("pred", "dtif"), ("true", "mtif")):
            if stem == "{}_{}".format(label, suffix):
                return name, label
    return "unknown", stem


def collect(config: dict) -> tuple:
    """(provenance rows, [(archive name, scrubbed text)]) over every scoring run."""
    labels = config["tif_label"]
    rows, logs = [], []

    for pass_name, profile_name in PASSES:
        profile = config["profiles"][profile_name or config["active_profile"]]["paths"]
        for kind, directory in (("tif", Path(profile["tif_dir"])),
                                ("lv", Path(profile["lv_dir"]))):
            if not directory.is_dir():
                print("[WARN] missing directory, skipped: {}".format(directory))
                continue
            for analysis, data_suffix in (("gene", "__gene_scores"),
                                          ("pathway", "__pathway_scores.txt")):
                for data in sorted(directory.glob("*" + data_suffix)):
                    stem = data.name[: -len(data_suffix)]
                    base = directory / (stem + data_suffix.replace(".txt", ""))
                    feature_set, trait = feature_set_of(stem, labels)

                    settings, settings_format = {}, "none"
                    for candidate, fmt in ((Path(str(base) + ".config.json"), "json"),
                                           (Path(str(base) + ".config"), "python")):
                        if candidate.is_file():
                            settings, settings_format = read_config(candidate), fmt
                            break

                    used = Path(str(base) + ".pathway_file_used.txt")
                    gene_set_file = (
                        used.read_text().strip() if used.is_file()
                        else settings.get("PATHWAYS", ""))

                    row = {
                        "pass": pass_name, "analysis": analysis,
                        "feature_set": feature_set, "trait": trait,
                        "output_rows": count_rows(data),
                        "settings_format": settings_format,
                    }
                    row.update({k.lower(): settings.get(k, "") for k in SETTING_KEYS})
                    # N_SAMPLINGS is read only in permutation mode. Every run here is
                    # RANKING, so carrying its configured value per row would state a
                    # parameter that no run applied.
                    if str(row.get("mode", "")).upper() != "PERMUTATION":
                        row["n_samplings"] = ""
                    row.update({k.lower(): Path(str(settings.get(k, ""))).name
                                for k in PATH_KEYS})
                    # Gene scoring reads no gene sets, but shares the config file that
                    # names one, so these two columns stay empty for it rather than
                    # inheriting a setting the run never used.
                    row["gene_set_file"] = (
                        Path(gene_set_file).name
                        if gene_set_file and analysis == "pathway" else "")
                    row["gene_set_selection"] = (
                        "within-trait BH, alpha 0.05" if "__within_BH" in row["gene_set_file"]
                        else "all gene sets" if row["gene_set_file"] else "")
                    row.update(parse_log(Path(str(base) + ".out")))

                    # A sidecar can outlive the run it describes: these directories were
                    # scored more than once, and a later run overwrites .config/.out
                    # while leaving the earlier .txt in place. Where the log says how
                    # many gene sets it loaded, that number must equal the rows the data
                    # file actually holds; where it does not, the settings on this row
                    # belong to a DIFFERENT run and must not be read as describing it.
                    modules = row.get("n_modules_loaded")
                    consistent = ""
                    if analysis == "pathway" and modules not in (None, ""):
                        consistent = str(int(modules) == (row["output_rows"] or -1))
                    row["settings_consistent"] = consistent
                    rows.append(row)

                    for stream in (".out", ".err"):
                        log = Path(str(base) + stream)
                        if log.is_file():
                            logs.append((
                                "{}/{}/{}{}".format(pass_name, kind, stem + "__" + analysis,
                                                    stream),
                                scrub(log.read_text(errors="replace")),
                            ))
    return rows, logs


def write_archive(logs: list, path: Path) -> None:
    """One deterministic .tgz. Sorted, and with mtime/uid/gid zeroed so re-running
    on unchanged logs reproduces the same bytes and the same deposited md5."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, text in sorted(logs):
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    with open(str(path), "wb") as handle:
        handle.write(gzip.compress(raw.getvalue(), mtime=0))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--check", action="store_true",
                        help="do not write; fail if either artefact has drifted")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    rows, logs = collect(config)
    # fillna("") before anything else: a run whose log stated none of the parsed facts
    # simply lacks those keys, so the frame would carry NaN where a CSV round-trip
    # yields "" -- and --check would report drift on every such row for ever.
    frame = pd.DataFrame(rows).fillna("").sort_values(
        ["pass", "analysis", "feature_set", "trait"]).reset_index(drop=True)
    print("[INFO] {} scoring run(s), {} log file(s)".format(len(frame), len(logs)))
    print(frame.groupby(["pass", "analysis", "feature_set"]).size().to_string())

    prov_path = args.out_dir / PROVENANCE_FILE
    arch_path = args.out_dir / LOG_ARCHIVE

    if args.check:
        failures = 0
        if not prov_path.is_file():
            print("[FAIL] {}: not written yet".format(prov_path)); failures += 1
        else:
            existing = pd.read_csv(str(prov_path), keep_default_na=False)
            same = existing.astype(str).equals(frame.astype(str).reset_index(drop=True))
            print("{} {}".format("[ OK ]" if same else "[FAIL]", PROVENANCE_FILE))
            failures += 0 if same else 1
        if not arch_path.is_file():
            print("[FAIL] {}: not written yet".format(arch_path)); failures += 1
        else:
            before = arch_path.read_bytes()
            write_archive(logs, arch_path)
            same = arch_path.read_bytes() == before
            print("{} {}".format("[ OK ]" if same else "[FAIL]", LOG_ARCHIVE))
            failures += 0 if same else 1
        return 1 if failures else 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(str(prov_path), index=False)
    write_archive(logs, arch_path)
    print("\n[INFO] wrote {} ({} rows)".format(prov_path, len(frame)))
    print("[INFO] wrote {} ({} logs, {:.1f} MB)".format(
        arch_path, len(logs), arch_path.stat().st_size / 1024 / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
