#!/usr/bin/env python3
"""
select_pathways_bh.py

Reproduce the pathway selection that decided which gene sets were rescored on the
corrected phenotypes -- the second pass of the two-pass pathway analysis.

    pixi run python tools/pascalx/select_pathways_bh.py --all --out-dir <gmt dir>
    pixi run python tools/pascalx/select_pathways_bh.py --all --verify

THE RULE. A gene set was selected for a trait if it passed
Benjamini-Hochberg at alpha = 0.05 **within that trait** -- one BH correction per
column of the first-pass screen, over the gene sets scored for that trait, never
pooled across traits. Selected sets were written to a per-trait `.gmt` named
`<trait>__within_BH.gmt`, and run_pathway_scoring.sh --pathway-file was pointed at it.

--verify checks the rebuilt selection against the `.gmt` files the original run used; it
reproduces all 20 tangible-feature and all 611 latent-variable selections exactly.

Selecting on the screen and rescoring only the selection is a deliberate, disclosed
shortcut -- the data record's README gives what it costs and why the direction of the
error is the conservative one.

NOTE ON THE .gmt FILES. The written gene sets carry MSigDB membership, which is
third-party and is NOT redistributed by this project -- the `.gmt` files are a local
intermediate, never a deposit item. What IS deposited is the selection itself, as the
(feature_set, trait, pathway) rows of pathway_scores_recomputed.csv:
the names, not MSigDB's gene lists.

Inputs:
  the first-pass screen tables built by aggregate_pathway_scores.py
  <genetic_tools.pascalx.pathway_file>   the master MSigDB .gmt, to copy lines from

Outputs:
  <out-dir>/<trait>__within_BH.gmt       one per trait with at least one selection
  <out-dir>/selection_summary.csv        trait, gene sets tested, selected, cutoff
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.config import load_config  # noqa: E402 -- machine paths: config.local.yaml

SCREEN_DIR = REPO_ROOT / "outputs" / "natgen_revision" / "intermediate" / "pathway_scores"

SCREENS = {
    "dtif": ("pathway_scores_dtif_screen.csv", "pred"),
    "mtif": ("pathway_scores_mtif_screen.csv", "true"),
    "lv": ("pathway_scores_lv_screen.csv.gz", None),
}

ALPHA = 0.05
METHOD = "fdr_bh"


def trait_stem(column: str, suffix: str | None) -> str:
    """Screen column name -> the trait stem the .gmt is named after.

    `bifurcations_p` + `pred` -> `bifurcations_pred`; `LV_7_p` -> `LV_7`. The `_p` is
    the screen table's marker that a column holds p-values, not part of the trait.
    """
    base = column[:-2] if column.endswith("_p") else column
    return base if suffix is None else "{}_{}".format(base, suffix)


def select(screen: pd.DataFrame, suffix: str | None) -> dict:
    """{trait stem: selected gene sets} by within-trait BH at ALPHA.

    NaN is dropped per column before correcting, so a gene set that was not scored for
    a trait does not enter that trait's multiple-testing burden. Doing it the other way
    would make one trait's selection depend on gene sets it never tested.
    """
    out = {}
    for column in screen.columns:
        scores = screen[column].dropna()
        if scores.empty:
            continue
        reject, *_ = multipletests(scores.values, alpha=ALPHA, method=METHOD)
        chosen = list(scores.index[reject])
        if chosen:
            out[trait_stem(column, suffix)] = chosen
    return out


def load_gmt_lines(path: Path) -> dict:
    """{gene set name: its whole line} from the master .gmt, kept verbatim.

    The line is copied through rather than rebuilt so the written .gmt is byte-identical
    to the master's formatting -- description field, gene order and all.
    """
    lines = {}
    with open(str(path), encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line:
                lines[line.split("\t", 1)[0]] = line
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--feature-set", choices=sorted(SCREENS), action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--screen-dir", type=Path, default=SCREEN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where to write the .gmt files; omit with --verify")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--verify", type=Path, nargs="*", default=None,
                        help="compare against the .gmt directories of an existing run "
                             "instead of writing; give the directories explicitly")
    args = parser.parse_args(argv)

    sets = sorted(SCREENS) if args.all else (args.feature_set or [])
    if not sets:
        parser.error("give --all or at least one --feature-set")
    if args.verify is None and args.out_dir is None:
        parser.error("give --out-dir, or --verify to check against the original .gmt files")

    config = load_config(args.config)

    selections = {}
    for feature_set in sets:
        name, suffix = SCREENS[feature_set]
        path = args.screen_dir / name
        if not path.is_file():
            raise SystemExit(
                "first-pass screen missing: {}\n"
                "  Build it with aggregate_pathway_scores.py --screen {}".format(
                    path, feature_set))
        print("[INFO] {}: reading {}".format(feature_set, path.name))
        screen = pd.read_csv(str(path), index_col=0)
        screen.index = screen.index.astype(str)
        chosen = select(screen, suffix)
        selections[feature_set] = chosen
        print("[INFO] {}: {} trait(s) with at least one selected gene set, {} cell(s)"
              .format(feature_set, len(chosen), sum(len(v) for v in chosen.values())))

    if args.verify is not None:
        # The two directories our own run wrote to were hardcoded here and are now
        # asked for, because they were machine paths in a published file and because a
        # default that exists on exactly one computer is not a default.
        if not args.verify:
            raise SystemExit(
                "--verify needs the directories to compare against: the .gmt output "
                "directories of the run you are checking, e.g.\n"
                "  --verify <out-dir>/retfound_rebuttal_tifs <out-dir>/retfound_rebuttal"
            )
        dirs = [Path(d) for d in args.verify]
        actual = {}
        for directory in dirs:
            for gmt in directory.glob("*__within_BH.gmt"):
                stem = gmt.name.replace("__within_BH.gmt", "")
                actual[stem] = {l.split("\t", 1)[0] for l in open(str(gmt)) if l.strip()}
        if not actual:
            raise SystemExit("no *__within_BH.gmt found in: {}".format(
                ", ".join(str(d) for d in dirs)))

        ok = bad = 0
        for chosen in selections.values():
            for stem, names in chosen.items():
                if stem not in actual:
                    continue          # trait was never rescored; nothing to compare
                if set(names) == actual[stem]:
                    ok += 1
                else:
                    bad += 1
                    print("[FAIL] {}: rebuilt {} vs original {}".format(
                        stem, len(names), len(actual[stem])))
        print("\n{} {} of {} original selections reproduced exactly".format(
            "[ OK ]" if bad == 0 else "[FAIL]", ok, ok + bad))
        return 1 if bad else 0

    master = Path(config["genetic_tools"]["pascalx"]["pathway_file"])
    if not master.is_file():
        raise SystemExit("master gene-set file not found: {}".format(master))
    lines = load_gmt_lines(master)
    print("[INFO] master gene sets: {}".format(len(lines)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for feature_set, chosen in selections.items():
        for stem, names in sorted(chosen.items()):
            present = [n for n in names if n in lines]
            absent = [n for n in names if n not in lines]
            target = args.out_dir / "{}__within_BH.gmt".format(stem)
            with open(str(target), "w", encoding="utf-8") as handle:
                for name in present:
                    handle.write(lines[name] + "\n")
            if absent:
                # Loud, and recorded: a selected gene set the master no longer contains
                # means the screen and this .gmt are different MSigDB versions.
                print("[WARN] {}: {} selected gene set(s) absent from the master .gmt"
                      .format(stem, len(absent)))
            summary.append({
                "feature_set": feature_set, "trait": stem,
                "n_selected": len(names), "n_written": len(present),
                "n_absent_from_master": len(absent),
                "alpha": ALPHA, "method": METHOD,
            })
    frame = pd.DataFrame(summary)
    frame.to_csv(str(args.out_dir / "selection_summary.csv"), index=False)
    print("\n[INFO] wrote {} .gmt file(s) + selection_summary.csv -> {}".format(
        len(frame), args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
