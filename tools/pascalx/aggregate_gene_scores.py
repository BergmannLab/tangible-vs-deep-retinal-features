#!/usr/bin/env python3
"""
aggregate_gene_scores.py

Aggregate the per-trait PascalX gene scores of one feature set into the wide
gene x trait matrix that is deposited on Zenodo.

    pixi run python tools/pascalx/aggregate_gene_scores.py --all
    pixi run python tools/pascalx/aggregate_gene_scores.py --feature-set dtif --check

The matrices are built from the per-trait PascalX output in the active profile's own GWAS
directory, and this script is the one place that turns it into a matrix. It is wired into
run_gene_scoring.sh (--aggregate), so a rescoring run rewrites the deposited table instead
of leaving it behind the scores it holds.

Inputs (access-controlled, stage 02 -- not in the public deposit):
  <profile tif_dir>/<trait>_{true,pred}__gene_scores   one file per TIF trait, per set
  <profile lv_dir>/LV_<i>__gene_scores                 one file per latent variable
      each a two-column, tab-separated `gene<TAB>p` table written by score_genes.py,
      with no header

Outputs (deposited with the public data record):
  gene_scores_dtif.csv      25 819 genes x 17 deep tangible image features
  gene_scores_mtif.csv      25 819 genes x 17 measured tangible image features
  gene_scores_lv.csv.gz     25 819 genes x 1024 RETFound latent variables

The LV matrix is gzipped because it is ~500 MB as plain text. gzip's mtime field is
zeroed so that re-running on unchanged inputs reproduces the same bytes and the same md5 --
the deposit manifest records that md5, and a checksum that changed on every run would be
worthless as a provenance check.
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

DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "natgen_revision" / "intermediate" / "gene_scores"

# `true` is the measured TIF and `pred` the deep one -- the file naming predates the
# mTIF/dTIF vocabulary.
TIF_SUFFIX = {"dtif": "pred", "mtif": "true"}

FEATURE_SETS = {
    "dtif": "gene_scores_dtif.csv",
    "mtif": "gene_scores_mtif.csv",
    "lv": "gene_scores_lv.csv.gz",
}

_LV_INDEX = re.compile(r"^LV_(\d+)__gene_scores$")


def read_gene_scores(path: Path) -> pd.Series:
    """One trait's PascalX output: a two-column `gene<TAB>p` table with no header.

    Identical to the pipeline's gene-venn reader, including
    the duplicate handling: a gene appears twice when the genome annotation carries
    duplicate entries for it, the two scores agree, and keeping the first occurrence
    keeps the index unique.
    """
    table = pd.read_csv(
        str(path), sep="\t", header=None, names=["gene", "p"], dtype={"gene": str}
    )
    return table.drop_duplicates("gene").set_index("gene")["p"]


def score_paths(feature_set: str, tif_dir: Path, lv_dir: Path, labels: list) -> list:
    """(column name, file) for every trait in the feature set, in deposit order.

    TIFs follow config.yaml's `tif_label` order, which is the order every other figure
    and table uses. LVs are sorted NUMERICALLY -- a lexical sort puts LV_10 next to LV_1
    and would silently scramble 1024 columns.
    """
    if feature_set == "lv":
        found = {}
        for path in lv_dir.glob("LV_*__gene_scores"):
            match = _LV_INDEX.match(path.name)
            if match:
                found[int(match.group(1))] = path
        return [("LV_{}_p".format(i), found[i]) for i in sorted(found)]

    suffix = TIF_SUFFIX[feature_set]
    return [
        ("{}_p".format(label), tif_dir / "{}_{}__gene_scores".format(label, suffix))
        for label in labels
    ]


def build_matrix(pairs: list, n_genes_expected: int) -> pd.DataFrame:
    """Wide gene x trait matrix, aligned on the gene index.

    Built by aligning each trait's own Series rather than assuming the per-trait files
    share a row order -- they are written in scoring order, which is not the same order
    twice. The row count is checked against config.yaml's gwas.n_genes_scored: an
    aligned join of files that disagree about which genes exist would otherwise widen
    the index silently and pad the difference with NaN.
    """
    missing = [str(path) for _, path in pairs if not path.is_file()]
    if missing:
        raise SystemExit(
            "{} gene-score file(s) missing, first is {}".format(len(missing), missing[0])
        )

    matrix = pd.DataFrame({name: read_gene_scores(path) for name, path in pairs})
    matrix.index.name = "gene"

    if len(matrix) != n_genes_expected:
        raise SystemExit(
            "aggregated {} genes but config.yaml says gwas.n_genes_scored = {}.\n"
            "  The per-trait files disagree about which genes were scored, so the join "
            "widened the index and padded the difference with NaN.".format(
                len(matrix), n_genes_expected
            )
        )
    return matrix


def write_matrix(matrix: pd.DataFrame, path: Path) -> None:
    """CSV, gzipped when the name says so, with a reproducible gzip header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".gz"):
        # mtime=0: see the module docstring. Without it the gzip header carries the
        # current time and the file's md5 changes on every run.
        matrix.to_csv(str(path), compression={"method": "gzip", "mtime": 0})
    else:
        matrix.to_csv(str(path))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), action="append",
                        help="repeatable; defaults to all three when --all is given")
    parser.add_argument("--all", action="store_true", help="all three feature sets")
    parser.add_argument("--tif-dir", type=Path, default=None,
                        help="overrides the active profile's paths.tif_dir")
    parser.add_argument("--lv-dir", type=Path, default=None,
                        help="overrides the active profile's paths.lv_dir")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--check", action="store_true",
                        help="do not write; fail if the existing table differs from the "
                             "per-trait files it claims to aggregate")
    args = parser.parse_args(argv)

    sets = sorted(FEATURE_SETS) if args.all else (args.feature_set or [])
    if not sets:
        parser.error("give --all or at least one --feature-set")

    config = load_config(args.config)
    profile = config["profiles"][config["active_profile"]]["paths"]
    tif_dir = args.tif_dir or Path(profile["tif_dir"])
    lv_dir = args.lv_dir or Path(profile["lv_dir"])
    n_genes = int(config["gwas"]["n_genes_scored"])
    labels = config["tif_label"]

    print("[INFO] profile:  {}".format(config["active_profile"]))
    print("[INFO] tif dir:  {}".format(tif_dir))
    print("[INFO] lv dir:   {}".format(lv_dir))

    failures = 0
    for feature_set in sets:
        pairs = score_paths(feature_set, tif_dir, lv_dir, labels)
        matrix = build_matrix(pairs, n_genes)
        out_path = args.out_dir / FEATURE_SETS[feature_set]

        if args.check:
            if not out_path.is_file():
                print("[FAIL] {}: not written yet".format(out_path))
                failures += 1
                continue
            # float_precision="round_trip" is required, and only here. pandas writes a
            # double as its shortest round-tripping repr but reads it back with a fast
            # parser that can land one ULP away, so a byte-perfect file compares unequal
            # in 81 of 438 923 cells. The fast parser stays in read_gene_scores(): it is
            # what produced the matrices the published figures were drawn from, and the
            # deposit must match those rather than be silently re-rounded.
            existing = pd.read_csv(str(out_path), index_col=0, float_precision="round_trip")
            existing.index = existing.index.astype(str)
            same = (existing.shape == matrix.shape
                    and list(existing.columns) == list(matrix.columns)
                    and existing.reindex(matrix.index).equals(matrix))
            print("{} {}: {} genes x {} traits".format(
                "[ OK ]" if same else "[FAIL]", out_path.name, *matrix.shape))
            failures += 0 if same else 1
            continue

        write_matrix(matrix, out_path)
        print("[INFO] wrote {}  ({} genes x {} traits, {:.1f} MB)".format(
            out_path, matrix.shape[0], matrix.shape[1],
            out_path.stat().st_size / 1024 / 1024))

    if failures:
        print("\n[ERROR] {} table(s) do not match the per-trait scores they aggregate.\n"
              "        Re-run without --check to rewrite them.".format(failures),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
