#!/usr/bin/env python3
"""
aggregate_pathway_scores.py

Aggregate the per-trait PascalX pathway scores into the tables that are deposited
on Zenodo.

    pixi run python tools/pascalx/aggregate_pathway_scores.py --all
    pixi run python tools/pascalx/aggregate_pathway_scores.py --screen lv --check

THE PATHWAY ANALYSIS WAS RUN IN TWO PASSES, AND BOTH ARE DEPOSITED.

  Pass 1, the screen. Every one of the 31 120 MSigDB v7.2 gene sets scored against
  every trait -- all 34 TIF trait-sets and all 1024 latent variables -- on the
  EARLIER phenotype files, i.e. the `preprint` profile. Complete: 34 + 1024 files of
  31 120 rows each.

  Pass 2, the recomputation. When the phenotypes were corrected (the
  `rebuttal_highQualitySNPs` profile, MAF > 0.01 / INFO > 0.8), pathway scoring was
  NOT repeated in full -- it was repeated only for the pathways that had already
  reached suggestive significance in pass 1. That covers 20 of the 34 TIF trait-sets
  and 611 of the 1024 latent variables, and within those, only the selected pathways:
  560 TIF and 9 974 LV pathway-trait cells.

Figure 3j and k are therefore a hybrid -- a recomputed value where one exists, the
pass-1 value otherwise -- and this is a deliberate, disclosed shortcut rather than an
accident. Depositing both passes is what makes it checkable: a reader can see which
of the two produced any number in the figure, and can measure the disagreement
themselves on the cells where both exist.

What that disagreement is, measured on the 560 overlapping TIF cells: the median
log10 ratio is ~0, so there is no systematic drift between passes, but individual
pathways move by up to 35x, and 14 of the 560 significance calls flip -- 10 gaining
significance and 4 losing it. Because recomputation lowers a p-value more often than
it raises one, the pathways that were never recomputed can only be ones that should
have been reported and were not. The omission cannot manufacture a pathway that
appears in the figure.

Inputs (access-controlled, stage 02 -- not in the public deposit):
  <preprint tif_dir>/<trait>_{true,pred}__pathway_scores.txt   pass 1, 34 files
  <preprint lv_dir>/LV_<i>__pathway_scores.txt                 pass 1, 1024 files
  <active tif_dir>/<trait>_{true,pred}__pathway_scores.txt     pass 2, 20 files
  <active lv_dir>/LV_<i>__pathway_scores.txt                   pass 2, 611 files
      each whitespace-separated, no header: pathway, member genes, genes scored, p

Outputs (deposited with the public data record):
  pathway_scores_dtif_screen.csv     31 120 pathways x 17 dTIFs   (pass 1)
  pathway_scores_mtif_screen.csv     31 120 x 17 mTIFs            (pass 1)
  pathway_scores_lv_screen.csv.gz    31 120 x 1024 LVs            (pass 1)
  pathway_scores_recomputed.csv      long, every pass-2 cell      (pass 2)

gzip mtime is zeroed on the LV table so that re-running on unchanged inputs
reproduces the same bytes and the same md5, which the deposit manifest records.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.config import load_config  # noqa: E402 -- machine paths: config.local.yaml

DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "natgen_revision" / "intermediate" / "pathway_scores"

# The profile the full screen was run in. Not the active one: see the module docstring.
SCREEN_PROFILE = "preprint"

TIF_SUFFIX = {"dtif": "pred", "mtif": "true"}

SCREEN_FILES = {
    "dtif": "pathway_scores_dtif_screen.csv",
    "mtif": "pathway_scores_mtif_screen.csv",
    "lv": "pathway_scores_lv_screen.csv.gz",
}
RECOMPUTED_FILE = "pathway_scores_recomputed.csv"

_LV_INDEX = re.compile(r"^LV_(\d+)__pathway_scores\.txt$")

COLUMNS = ["pathway", "n_genes", "n_scored", "p"]


def read_pathway_scores(path: Path) -> pd.DataFrame:
    """One trait's PascalX pathway output: whitespace-separated, no header.

    A gene-set name never contains whitespace in MSigDB, so splitting on runs of
    whitespace is safe.
    """
    table = pd.read_csv(
        str(path), sep=r"\s+", header=None, names=COLUMNS, dtype={"pathway": str}
    )
    return table.drop_duplicates("pathway").set_index("pathway")


def screen_paths(feature_set: str, tif_dir: Path, lv_dir: Path, labels: list) -> list:
    """(column name, file) for every trait in the feature set, in deposit order.

    LVs are sorted NUMERICALLY -- a lexical sort would put LV_10 next to LV_1 and
    silently scramble 1024 columns.
    """
    if feature_set == "lv":
        found = {}
        for path in lv_dir.glob("LV_*__pathway_scores.txt"):
            match = _LV_INDEX.match(path.name)
            if match:
                found[int(match.group(1))] = path
        return [("LV_{}_p".format(i), found[i]) for i in sorted(found)]

    suffix = TIF_SUFFIX[feature_set]
    return [
        ("{}_p".format(label), tif_dir / "{}_{}__pathway_scores.txt".format(label, suffix))
        for label in labels
    ]


def build_screen(pairs: list, n_pathways_expected: int) -> pd.DataFrame:
    """Wide pathway x trait matrix of p-values, aligned on the pathway index.

    `n_genes` and `n_scored` are properties of the RUN, not of the trait, so they are
    carried once at the front of the table when every trait agrees on them -- and
    dropped, loudly, if they do not, rather than picking one trait's copy and implying
    the others matched.
    """
    missing = [str(path) for _, path in pairs if not path.is_file()]
    if missing:
        raise SystemExit(
            "{} pathway-score file(s) missing, first is {}".format(len(missing), missing[0])
        )

    tables = {name: read_pathway_scores(path) for name, path in pairs}
    matrix = pd.DataFrame({name: t["p"] for name, t in tables.items()})
    matrix.index.name = "pathway"

    if len(matrix) != n_pathways_expected:
        raise SystemExit(
            "aggregated {} pathways but {} gene sets were loaded.\n"
            "  The per-trait files disagree about which pathways were scored, so the "
            "join widened the index and padded the difference with NaN.".format(
                len(matrix), n_pathways_expected
            )
        )

    first = next(iter(tables.values()))
    counts = first[["n_genes", "n_scored"]]
    if all(t[["n_genes", "n_scored"]].reindex(counts.index).equals(counts)
           for t in tables.values()):
        matrix = pd.concat([counts, matrix], axis=1)
    else:
        print("[WARN] n_genes/n_scored differ between traits; omitting both columns")
    return matrix


def build_recomputed(config: dict, tif_dir: Path, lv_dir: Path, labels: list) -> pd.DataFrame:
    """Every pass-2 cell as one row: which trait, which pathway, what it scored.

    Long rather than wide, because pass 2 is sparse by construction -- a wide table
    would be mostly empty and would invite reading a blank as a missing result rather
    than as a pathway that was never selected for recomputation.
    """
    rows = []
    for feature_set, suffix in TIF_SUFFIX.items():
        for label in labels:
            path = tif_dir / "{}_{}__pathway_scores.txt".format(label, suffix)
            if not path.is_file():
                continue
            table = read_pathway_scores(path).reset_index()
            table.insert(0, "trait", label)
            table.insert(0, "feature_set", feature_set)
            rows.append(table)

    for path in sorted(lv_dir.glob("LV_*__pathway_scores.txt"),
                       key=lambda p: int(_LV_INDEX.match(p.name).group(1))):
        table = read_pathway_scores(path).reset_index()
        table.insert(0, "trait", path.name.split("__")[0])
        table.insert(0, "feature_set", "lv")
        rows.append(table)

    if not rows:
        raise SystemExit("no pass-2 pathway files found under {} or {}".format(tif_dir, lv_dir))
    return pd.concat(rows, ignore_index=True)[["feature_set", "trait"] + COLUMNS]


def write_table(frame: pd.DataFrame, path: Path, index: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz"):
        # mtime=0 -- see the module docstring.
        frame.to_csv(str(path), index=index, compression={"method": "gzip", "mtime": 0})
    else:
        frame.to_csv(str(path), index=index)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--screen", choices=sorted(SCREEN_FILES), action="append",
                        help="build the pass-1 matrix for this feature set; repeatable")
    parser.add_argument("--recomputed", action="store_true", help="build the pass-2 table")
    parser.add_argument("--all", action="store_true", help="all four tables")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--check", action="store_true",
                        help="do not write; fail if a table differs from its inputs")
    args = parser.parse_args(argv)

    screens = sorted(SCREEN_FILES) if args.all else (args.screen or [])
    recomputed = args.all or args.recomputed
    if not screens and not recomputed:
        parser.error("give --all, --recomputed, or at least one --screen")

    config = load_config(args.config)
    screen = config["profiles"][SCREEN_PROFILE]["paths"]
    active = config["profiles"][config["active_profile"]]["paths"]
    labels = config["tif_label"]
    n_pathways = config["gwas"]["n_pathways_scored"]

    print("[INFO] pass 1 (screen)     profile {}: {}".format(SCREEN_PROFILE, screen["tif_dir"]))
    print("[INFO] pass 2 (recomputed) profile {}: {}".format(
        config["active_profile"], active["tif_dir"]))

    failures = 0
    jobs = [(SCREEN_FILES[s], s, True) for s in screens]
    if recomputed:
        jobs.append((RECOMPUTED_FILE, None, False))

    for name, feature_set, is_screen in jobs:
        if is_screen:
            frame = build_screen(
                screen_paths(feature_set, Path(screen["tif_dir"]), Path(screen["lv_dir"]),
                             labels), n_pathways)
        else:
            frame = build_recomputed(config, Path(active["tif_dir"]),
                                     Path(active["lv_dir"]), labels)
        out_path = args.out_dir / name

        if args.check:
            if not out_path.is_file():
                print("[FAIL] {}: not written yet".format(out_path))
                failures += 1
                continue
            existing = pd.read_csv(str(out_path), index_col=0 if is_screen else None,
                                   float_precision="round_trip")
            if is_screen:
                existing.index = existing.index.astype(str)
                same = existing.reindex(frame.index).equals(frame)
            else:
                same = existing.equals(frame)
            print("{} {}: {} x {}".format("[ OK ]" if same else "[FAIL]", name, *frame.shape))
            failures += 0 if same else 1
            continue

        write_table(frame, out_path, index=is_screen)
        print("[INFO] wrote {}  ({} x {}, {:.1f} MB)".format(
            out_path, frame.shape[0], frame.shape[1],
            out_path.stat().st_size / 1024 / 1024))

    if failures:
        print("\n[ERROR] {} table(s) do not match their inputs.\n"
              "        Re-run without --check to rewrite them.".format(failures),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
