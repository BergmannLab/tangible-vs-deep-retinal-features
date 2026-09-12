"""Score genes from GWAS summary statistics with PascalX.

PascalX aggregates SNP-level p-values into gene-level p-values, accounting for LD
from a reference panel. It is third-party software (github.com/BergmannLab/PascalX)
and is cited in the Methods; this script is only the wrapper that drives it with
the parameters used for the published analyses.

Everything variable is read, never hardcoded:
  - scoring parameters and the pinned tool version  -> params.yaml (next to this file)
  - per-cohort column names                         -> replication.yaml
  - reference panel / annotation paths              -> config.yaml, genetic_tools.pascalx

Usage:
    score_genes.py --sumstats <file> --cohort <name> --output-dir <dir>

Outputs, per trait, into --output-dir:
    <trait>__gene_scores         gene <TAB> p-value, one line per gene
    <trait>__gene_scores.p       pickled (scores, failed_genes) as returned by PascalX
    <trait>__gene_scores.params.json   every parameter this run actually used

The third file carries the provenance with the scores.

The trait label is everything before the first "__" in the input filename, matching
the convention used across the pipeline.
"""

import argparse
import datetime
import json
import os
import platform
import sys

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GENETIC_DIR = os.path.dirname(HERE)
sys.path.insert(0, GENETIC_DIR)

from cohort_config import load_cohort, load_tool  # noqa: E402

DEFAULT_PARAMS = os.path.join(HERE, "params.yaml")


def load_params(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def check_version(expected):
    """Fail loudly on a version mismatch.

    Scoring with a different PascalX version produces plausible numbers that need
    not match the published ones.
    """
    import PascalX

    actual = getattr(PascalX, "__version__", "unknown")
    if str(actual) != str(expected):
        raise SystemExit(
            "[ERROR] PascalX version mismatch: params.yaml pins %s, imported %s\n"
            "        imported from: %s\n"
            "        Check PYTHONPATH / easy-install.pth ordering."
            % (expected, actual, os.path.dirname(PascalX.__file__))
        )
    return actual


def resolve_columns(sumstats_path, columns, delimiter):
    """Map the cohort's column NAMES to the positional indices PascalX wants."""
    header = pd.read_csv(sumstats_path, sep=delimiter, nrows=0).columns.tolist()
    resolved = {}
    for role, name in (
        ("rscol", columns.get("rsid")),
        ("pcol", columns.get("pvalue")),
        ("bcol", columns.get("effect")),
    ):
        if not name:
            raise SystemExit(
                "[ERROR] The cohort config has no column for '%s'. Add it to "
                "replication.yaml rather than guessing here." % role
            )
        if name not in header:
            raise SystemExit(
                "[ERROR] Column %r (%s) not found in %s\n        Columns present: %s"
                % (name, role, os.path.basename(sumstats_path), ", ".join(header))
            )
        resolved[role] = header.index(name)
    # PascalX misparses the final column when it is the rsid column.
    if resolved["rscol"] == len(header) - 1:
        raise SystemExit(
            "[ERROR] The rsid column is last in %s; PascalX will not parse it "
            "correctly. Reorder the columns upstream." % os.path.basename(sumstats_path)
        )
    return resolved, header


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sumstats", required=True, help="summary statistics file")
    parser.add_argument("--cohort", required=True, help="cohort name in replication.yaml")
    parser.add_argument("--output-dir", required=True, help="where to write the scores")
    parser.add_argument("--params", default=DEFAULT_PARAMS, help="parameter YAML")
    parser.add_argument("--delimiter", default="\t", help="sumstats field separator")
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--gpu", dest="gpu", action="store_true", default=None)
    gpu.add_argument("--no-gpu", dest="gpu", action="store_false", default=None)
    parser.add_argument("--n-cpu", type=int, help="override params.yaml runtime.n_cpu")
    args = parser.parse_args(argv)

    params = load_params(args.params)
    scoring = params["scoring"]
    runtime = params["runtime"]
    tool = params["tool"]

    use_gpu = runtime["gpu"] if args.gpu is None else args.gpu
    n_cpu = args.n_cpu if args.n_cpu else runtime["n_cpu"]

    sumstats = os.path.abspath(args.sumstats)
    if not os.path.isfile(sumstats):
        raise SystemExit("[ERROR] No such sumstats file: %s" % sumstats)
    trait = os.path.basename(sumstats).split("__")[0]

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_base = os.path.join(output_dir, trait + "__gene_scores")

    paths = load_tool("pascalx")
    cohort = load_cohort(args.cohort)
    columns = cohort.get("columns") or {}

    for label, path in (
        ("reference panel", paths["reference_panel"] + ".chr1.db"),
        ("genome annotation", paths["genome_annotation"]),
    ):
        if not os.path.exists(path):
            raise SystemExit("[ERROR] %s not found: %s" % (label, path))

    version = check_version(tool["version"])
    import PascalX
    from PascalX import genescorer

    resolved, header = resolve_columns(sumstats, columns, args.delimiter)

    print("=" * 70)
    print("trait:       %s" % trait)
    print("cohort:      %s (%s)" % (cohort.get("name"), cohort.get("type")))
    print("sumstats:    %s" % sumstats)
    print("output:      %s" % out_base)
    print("PascalX:     %s  (%s)" % (version, os.path.dirname(PascalX.__file__)))
    print("python:      %s" % sys.version.split()[0])
    print("columns:     rsid=%s[%d] p=%s[%d] beta=%s[%d]"
          % (columns.get("rsid"), resolved["rscol"],
             columns.get("pvalue"), resolved["pcol"],
             columns.get("effect"), resolved["bcol"]))
    print("parameters:  method=%s varcutoff=%s window=%s autorescore=%s"
          % (scoring["method"], scoring["varcutoff"], scoring["window"],
             scoring["autorescore"]))
    print("runtime:     gpu=%s n_cpu=%s" % (use_gpu, n_cpu))
    print("=" * 70)

    start = datetime.datetime.now()

    print("[INFO] Loading scorer")
    scorer = genescorer.chi2sum(
        varcutoff=scoring["varcutoff"], window=scoring["window"], gpu=use_gpu
    )

    print("[INFO] Loading reference panel")
    scorer.load_refpanel(paths["reference_panel"], parallel=n_cpu)

    print("[INFO] Loading genome annotation")
    scorer.load_genome(paths["genome_annotation"])

    print("[INFO] Loading GWAS summary statistics")
    scorer.load_GWAS(
        sumstats,
        rscol=resolved["rscol"],
        pcol=resolved["pcol"],
        bcol=resolved["bcol"],
        delimiter=args.delimiter,
        header=True,
    )

    print("[INFO] Scoring genes")
    result = scorer.score_all(
        parallel=n_cpu,
        method=scoring["method"],
        nobar=True,
        autorescore=scoring["autorescore"],
    )
    failed = result[1]
    if failed:
        print("[WARN] %d gene(s) failed to score:" % len(failed))
        print("       %s" % failed)
    else:
        print("[INFO] All genes scored")

    print("[INFO] Saving results")
    scorer.save_scores(out_base)
    import pickle

    with open(out_base + ".p", "wb") as handle:
        pickle.dump(result, handle)

    elapsed = datetime.datetime.now() - start
    snapshot = {
        "trait": trait,
        "cohort": cohort.get("name"),
        "cohort_type": cohort.get("type"),
        "sumstats": sumstats,
        "sumstats_columns": {k: columns.get(k) for k in ("rsid", "pvalue", "effect")},
        "sumstats_column_indices": resolved,
        "tool": {
            "name": tool["name"],
            "version": str(version),
            "repository": tool.get("repository"),
            "commit": tool.get("commit"),
            "module_path": os.path.dirname(PascalX.__file__),
        },
        "scoring": scoring,
        "runtime": {"gpu": use_gpu, "n_cpu": n_cpu},
        "paths": {
            "reference_panel": paths["reference_panel"],
            "genome_annotation": paths["genome_annotation"],
        },
        "environment": {
            "python": sys.version.split()[0],
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "n_genes_failed": len(failed),
        "genes_failed": list(failed),
        "started": start.isoformat(timespec="seconds"),
        "elapsed_seconds": int(elapsed.total_seconds()),
    }
    with open(out_base + ".params.json", "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)

    print("[INFO] Completed in %dh %dm %ds"
          % (elapsed.seconds // 3600, (elapsed.seconds // 60) % 60, elapsed.seconds % 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
