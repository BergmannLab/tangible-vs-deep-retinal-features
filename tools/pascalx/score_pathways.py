"""Score pathways (gene sets) from GWAS summary statistics with PascalX.

A pathway score aggregates the gene scores of its members, so this is the same
scorer as score_genes.py with a second stage on top: chi2rank (RANKING) or
chi2perm (PERMUTATION). Everything variable is read, never hardcoded:

  - scoring and pathway parameters      -> params.yaml (next to this file)
  - per-cohort column names             -> replication.yaml
  - reference panel / annotation / .gmt -> config.yaml, genetic_tools.pascalx

Usage:
    score_pathways.py --sumstats <file> --cohort <name> --output-dir <dir>
                      [--gene-scores <file>] [--pathway-file <gmt>]
                      [--mode ranking|permutation]

Outputs, per trait, into --output-dir:
    <trait>__pathway_scores.txt          pathway, n_genes, n_scored, p-value
    <trait>__pathway_scores.p            pickled result as returned by PascalX
    <trait>__pathway_scores.params.json  every parameter this run actually used

The .txt layout is whitespace-separated.

Gene scores are reused when they already exist next to the output, because scoring
25 771 genes is the expensive half and pathway scoring on top of it is minutes. Pass
--rescore-genes to force them to be recomputed.

The tool version is asserted at run time, and n_samplings is only consulted in
permutation mode.
"""

import argparse
import datetime
import json
import os
import pickle
import platform
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
GENETIC_DIR = os.path.dirname(HERE)
sys.path.insert(0, GENETIC_DIR)

from cohort_config import load_cohort, load_tool  # noqa: E402
from score_genes import check_version, load_params, resolve_columns  # noqa: E402

DEFAULT_PARAMS = os.path.join(HERE, "params.yaml")


def write_scores_txt(result, path):
    """pathway, member genes, genes actually scored, p-value -- one line each.

    result[0] entries are (name, genes, gene_scores, p). The count of non-NaN
    gene scores is what distinguishes a pathway that was scored from one whose
    members mostly fell outside the annotation, and it is the reason this column
    exists rather than just the size.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for entry in result[0]:
            scores = np.asarray(entry[2], dtype=float)
            n_total = len(scores)
            n_scored = int(n_total - np.isnan(scores).sum())
            handle.write("%s %s %s %s\n" % (entry[0], n_total, n_scored, entry[3]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sumstats", required=True, help="summary statistics file")
    parser.add_argument("--cohort", required=True, help="cohort name in replication.yaml")
    parser.add_argument("--output-dir", required=True, help="where to write the scores")
    parser.add_argument("--params", default=DEFAULT_PARAMS, help="parameter YAML")
    parser.add_argument(
        "--gene-scores",
        help="existing <trait>__gene_scores to reuse "
             "(default: that file in --output-dir, if present)",
    )
    parser.add_argument(
        "--rescore-genes",
        action="store_true",
        help="recompute gene scores even if a file is available",
    )
    parser.add_argument(
        "--pathway-file",
        help="override genetic_tools.pascalx.pathway_file (a .gmt)",
    )
    parser.add_argument(
        "--genome-annotation",
        help="override genetic_tools.pascalx.genome_annotation. Which annotation is "
             "used decides how many members of each pathway exist at all, so an old "
             "result can only be reproduced with the annotation it was scored against.",
    )
    parser.add_argument(
        "--mode",
        choices=["ranking", "permutation"],
        help="override params.yaml pathways.mode",
    )
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
    pathway_params = params.get("pathways")
    if not pathway_params:
        raise SystemExit("[ERROR] %s has no `pathways:` block" % args.params)

    mode = (args.mode or pathway_params["mode"]).lower()
    use_gpu = runtime["gpu"] if args.gpu is None else args.gpu
    n_cpu = args.n_cpu if args.n_cpu else runtime["n_cpu"]

    sumstats = os.path.abspath(args.sumstats)
    if not os.path.isfile(sumstats):
        raise SystemExit("[ERROR] No such sumstats file: %s" % sumstats)
    trait = os.path.basename(sumstats).split("__")[0]

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_base = os.path.join(output_dir, trait + "__pathway_scores")
    gene_scores_path = os.path.abspath(args.gene_scores) if args.gene_scores else \
        os.path.join(output_dir, trait + "__gene_scores")

    paths = load_tool("pascalx")
    pathway_file = args.pathway_file or paths.get("pathway_file")
    if not pathway_file:
        raise SystemExit(
            "[ERROR] No pathway file. Set genetic_tools.pascalx.pathway_file in "
            "config.yaml or pass --pathway-file."
        )
    pathway_file = os.path.abspath(pathway_file)
    genome_annotation = os.path.abspath(
        args.genome_annotation or paths["genome_annotation"]
    )

    cohort = load_cohort(args.cohort)
    columns = cohort.get("columns") or {}

    for label, path in (
        ("reference panel", paths["reference_panel"] + ".chr1.db"),
        ("genome annotation", genome_annotation),
        ("pathway file", pathway_file),
    ):
        if not os.path.exists(path):
            raise SystemExit("[ERROR] %s not found: %s" % (label, path))

    version = check_version(tool["version"])
    import PascalX
    from PascalX import genescorer, pathway

    resolved, _ = resolve_columns(sumstats, columns, args.delimiter)
    reuse_genes = os.path.isfile(gene_scores_path) and not args.rescore_genes

    print("=" * 70)
    print("trait:       %s" % trait)
    print("cohort:      %s (%s)" % (cohort.get("name"), cohort.get("type")))
    print("sumstats:    %s" % sumstats)
    print("annotation:  %s" % genome_annotation)
    print("pathways:    %s" % pathway_file)
    print("output:      %s" % out_base)
    print("PascalX:     %s  (%s)" % (version, os.path.dirname(PascalX.__file__)))
    print("gene scores: %s" % (gene_scores_path if reuse_genes else "compute (none to reuse)"))
    print("parameters:  mode=%s fuse=%s mergedist=%s method=%s%s"
          % (mode, pathway_params["fuse"], pathway_params["mergedist"], scoring["method"],
             "" if mode != "permutation" else
             " samples=%s" % pathway_params["n_samplings"]))
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
    scorer.load_genome(genome_annotation)

    print("[INFO] Loading GWAS summary statistics")
    scorer.load_GWAS(
        sumstats,
        rscol=resolved["rscol"],
        pcol=resolved["pcol"],
        bcol=resolved["bcol"],
        delimiter=args.delimiter,
        header=True,
    )

    if reuse_genes:
        print("[INFO] Loading gene scores from %s" % gene_scores_path)
        scorer.load_scores(gene_scores_path)
        gene_scores_source = gene_scores_path
    else:
        print("[INFO] Scoring genes (no existing scores to reuse)")
        gene_result = scorer.score_all(
            parallel=n_cpu,
            method=scoring["method"],
            nobar=True,
            autorescore=scoring["autorescore"],
        )
        if gene_result[1]:
            print("[WARN] %d gene(s) failed to score" % len(gene_result[1]))
        scorer.save_scores(gene_scores_path)
        gene_scores_source = gene_scores_path + " (computed here)"

    print("[INFO] Scoring pathways in %s mode" % mode.upper())
    if mode == "ranking":
        scorer_pathway = pathway.chi2rank(
            scorer, fuse=pathway_params["fuse"], mergedist=pathway_params["mergedist"]
        )
    elif mode == "permutation":
        scorer_pathway = pathway.chi2perm(
            scorer, fuse=pathway_params["fuse"], mergedist=pathway_params["mergedist"]
        )
    else:
        raise SystemExit("[ERROR] Unknown pathway mode: %r" % mode)

    modules = scorer_pathway.load_modules(
        pathway_file,
        ncol=pathway_params["name_col"],
        fcol=pathway_params["first_gene_col"],
    )
    print("[INFO] %d pathway(s) loaded" % len(modules))

    score_kwargs = {"parallel": n_cpu, "nobar": True, "method": scoring["method"]}
    if mode == "permutation":
        score_kwargs["samples"] = pathway_params["n_samplings"]
    result = scorer_pathway.score(modules, **score_kwargs)

    print("[INFO] Saving results")
    with open(out_base + ".p", "wb") as handle:
        pickle.dump(result, handle)
    write_scores_txt(result, out_base + ".txt")

    elapsed = datetime.datetime.now() - start
    snapshot = {
        "trait": trait,
        "cohort": cohort.get("name"),
        "cohort_type": cohort.get("type"),
        "sumstats": sumstats,
        "sumstats_columns": {k: columns.get(k) for k in ("rsid", "pvalue", "effect")},
        "sumstats_column_indices": resolved,
        "gene_scores": gene_scores_source,
        "tool": {
            "name": tool["name"],
            "version": str(version),
            "repository": tool.get("repository"),
            "commit": tool.get("commit"),
            "module_path": os.path.dirname(PascalX.__file__),
        },
        "scoring": scoring,
        "pathways": {
            "mode": mode,
            "fuse": pathway_params["fuse"],
            "mergedist": pathway_params["mergedist"],
            "name_col": pathway_params["name_col"],
            "first_gene_col": pathway_params["first_gene_col"],
            "n_samplings": pathway_params["n_samplings"] if mode == "permutation" else None,
            "n_modules": len(modules),
        },
        "runtime": {"gpu": use_gpu, "n_cpu": n_cpu},
        "paths": {
            "reference_panel": paths["reference_panel"],
            "genome_annotation": genome_annotation,
            "pathway_file": pathway_file,
        },
        "environment": {
            "python": sys.version.split()[0],
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
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
