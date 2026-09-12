#!/usr/bin/env python3
"""
emit_scoring_provenance.py

Recover the gene-scoring provenance that the deposited score matrices drop, and write it
as three small deposit files.

    pixi run python tools/pascalx/emit_scoring_provenance.py --all
    pixi run python tools/pascalx/emit_scoring_provenance.py --all --check

WHY THIS EXISTS. `aggregate_gene_scores.py` turns one file per trait into a wide
gene x trait matrix of p-values, which is the right shape for a flat Zenodo record. But
PascalX writes more than a p-value, and the matrix keeps only that. Beside each
`<trait>__gene_scores` text file sits a `<trait>__gene_scores.p` pickle holding three
lists:

    scored   [gene, p, n_snps]                 <- n_snps is nowhere in the matrix
    failed   [gene, code, diagnostics]         <- scoring raised, for this trait
    skipped  [gene, 'No SNPs']                 <- no variants in the gene's window

So a reader of the deposit sees an empty cell and cannot tell "no SNPs in the window"
from "scoring failed here" -- two different facts, one of which is about the gene and
one about the trait. Recording *why* a value is missing, rather than only that it is,
is the part of missing-data practice that a bare NaN throws away.

The scoring settings emitted here come from `params.yaml`, the Methods-facing statement of
the run, rather than from the per-trait `<trait>__gene_scores.config` files, which carry
machine paths and a `MAF = 0.05` that the scorer never receives -- `chi2sum()` takes only
varcutoff, window and gpu, and the real filter is MAF > 0.01 / INFO > 0.8 applied upstream
when the summary statistics are prepared.

WHAT MAKES THIS THREE SMALL FILES AND NOT 1 058. Measured, not assumed (the script
re-measures every run and refuses to assert what it has not checked):

  - `n_snps` is a property of the GENE, not of the trait. Every trait is scored against
    the same variant set, so the count agrees across traits wherever both scored the gene.
  - the skipped set is likewise identical from trait to trait -- the same genes have no
    variants in their window whatever the phenotype.
  - only failures are genuinely per trait, and they are rare (0-1 per trait).

So the gene-level facts collapse into one table of ~26 000 rows, and the trait-level
facts into a table of a few dozen. Depositing the per-trait sidecars instead would add
~1 000 files to a flat listing to carry the same content.

OUTPUTS (deposited; see the source-data plan for the naming grammar)

  gene_scoring_provenance.yaml  settings, tool, reference data, coverage,
                                           missing-value conventions, and a field
                                           dictionary for the two tables below
  gene_annotation.csv           gene, n_snps, status -- one row per gene
                                           considered, trait-invariant
  gene_scoring_exceptions.csv   feature_set, trait, gene, reason, detail --
                                           the per-trait failures

The names carry the PIPELINE STAGE that produced them, not a counter. A counter would be
arrival order, which is meaningless to a reader, collides with the manuscript's numbered
Supplementary Tables, and makes two people adding a table at once race for a number.

Checksums are deliberately NOT repeated here: MANIFEST.tsv in the deposit already
records one per file and is the single authority. Two copies would drift.

PROVENANCE IS ALSO A STATEMENT ABOUT ITSELF. The `.p` sidecars were not retained for
every trait -- at the time of writing 989 of the 1 024 latent-variable traits have one,
and 1 011 have a `.config`. The matrices cover all 1 024 because they are built from the
text output, which is complete. Rather than quietly describe 989 traits as though they
were all of them, the coverage block records what fraction of each feature set the
provenance was actually derived from.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.config import load_config  # noqa: E402 -- machine paths: config.local.yaml

PARAMS = Path(__file__).resolve().parent / "params.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "natgen_revision" / "intermediate" / "gene_scores"

PROVENANCE_FILE = "gene_scoring_provenance.yaml"
ANNOTATION_FILE = "gene_annotation.csv"
EXCEPTIONS_FILE = "gene_scoring_exceptions.csv"

# Same vocabulary as aggregate_gene_scores.py: `true` is the measured TIF, `pred` the
# deep one. Kept in step deliberately -- these two scripts walk the same files.
TIF_SUFFIX = {"dtif": "pred", "mtif": "true"}
FEATURE_SETS = ("dtif", "mtif", "lv")
_LV_INDEX = re.compile(r"^LV_(\d+)__gene_scores\.p$")

# The status vocabulary. Closed set, documented in the emitted field dictionary, because
# an undocumented status column is as opaque as the blank cell it replaces.
#
# The skip reasons are READ from the sidecars and mapped here, not assumed: PascalX writes
# a free-text reason as the second element, and every skipped row in this run says
# 'No SNPs' (verified: 458 genes, the same set in every trait). If a future run introduces
# a second reason, an unmapped one stops the emit rather than being folded into no_snps and
# quietly misdescribing the data.
STATUS_SCORED = "scored"
STATUS_NO_SNPS = "no_snps"
SKIP_REASONS = {"No SNPs": STATUS_NO_SNPS}


def sidecar_paths(feature_set: str, tif_dir: Path, lv_dir: Path, labels: list) -> list:
    """(trait, pickle path) for every trait of the feature set that HAS a sidecar.

    Missing sidecars are not an error -- they are the coverage gap the docstring
    describes, and the caller reports them.
    """
    if feature_set == "lv":
        found = {}
        for path in lv_dir.glob("LV_*__gene_scores.p"):
            match = _LV_INDEX.match(path.name)
            if match:
                found[int(match.group(1))] = path
        return [("LV_{}".format(i), found[i]) for i in sorted(found)]

    suffix = TIF_SUFFIX[feature_set]
    pairs = []
    for label in labels:
        path = tif_dir / "{}_{}__gene_scores.p".format(label, suffix)
        if path.is_file():
            pairs.append(("{}_{}".format(label, suffix), path))
    return pairs


def expected_trait_count(feature_set: str, tif_dir: Path, lv_dir: Path, labels: list) -> int:
    """How many traits the MATRIX covers, which is the text output, not the sidecars."""
    if feature_set == "lv":
        return len(list(lv_dir.glob("LV_*__gene_scores")))  # excludes .p/.config by glob
    return len(labels)


class Accumulator:
    """Folds every trait's sidecar into gene-level facts, checking the invariants.

    Deliberately does not hold per-trait data: 1 000 traits x 26 000 genes would be a
    gigabyte for facts that are the same in every trait. What it keeps is the first
    observation of each gene plus counters for anything that disagrees with it.
    """

    def __init__(self):
        self.n_snps = {}
        self.status = {}
        self.n_snps_conflicts = Counter()
        self.status_conflicts = Counter()
        self.exceptions = []
        self.unknown_skip_reasons = Counter()
        self.traits_read = Counter()

    def add(self, feature_set: str, trait: str, payload) -> None:
        scored, failed, skipped = payload
        self.traits_read[feature_set] += 1

        for gene, _p, n_snps in scored:
            self._observe(gene, int(n_snps), STATUS_SCORED)
        for row in skipped:
            gene = row[0]
            reason = str(row[1]) if len(row) > 1 else ""
            if reason not in SKIP_REASONS:
                self.unknown_skip_reasons[reason] += 1
                continue
            self._observe(gene, 0, SKIP_REASONS[reason])

        for row in failed:
            gene = row[0]
            detail = row[1:] if len(row) > 1 else ""
            self.exceptions.append({
                "feature_set": feature_set,
                "trait": trait,
                "gene": gene,
                "reason": "scoring_failed",
                "detail": repr(detail) if detail != "" else "",
            })

    def _observe(self, gene: str, n_snps: int, status: str) -> None:
        if gene not in self.n_snps:
            self.n_snps[gene] = n_snps
            self.status[gene] = status
            return
        if self.n_snps[gene] != n_snps:
            self.n_snps_conflicts[gene] += 1
        if self.status[gene] != status:
            self.status_conflicts[gene] += 1


def write_annotation(acc: Accumulator, path: Path) -> int:
    rows = sorted(acc.n_snps)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("gene,n_snps,status\n")
        for gene in rows:
            handle.write("{},{},{}\n".format(gene, acc.n_snps[gene], acc.status[gene]))
    return len(rows)


def write_exceptions(acc: Accumulator, path: Path) -> int:
    ordered = sorted(acc.exceptions, key=lambda r: (r["feature_set"], r["trait"], r["gene"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("feature_set,trait,gene,reason,detail\n")
        for row in ordered:
            detail = row["detail"].replace('"', "'")
            handle.write('{},{},{},{},"{}"\n'.format(
                row["feature_set"], row["trait"], row["gene"], row["reason"], detail))
    return len(ordered)


def build_provenance(acc: Accumulator, params: dict, config: dict, coverage: dict,
                     n_genes: int, n_exceptions: int) -> dict:
    """The YAML record.

    Structure follows the shape provenance records are expected to take -- what was run,
    with which software at which version, under which parameters, over which reference
    data, producing which outputs -- without pulling in a JSON-LD vocabulary that a CSV
    deposit has no other use for. The `resources` block is a field dictionary in the
    Frictionless field style (name, type, description), so it can be lifted into a
    `datapackage.json` unchanged if the deposit ever gains one.

    Nothing here may carry a filesystem path, a hostname or a username. The deposit's
    disclosure scan enforces that; this docstring is why it should not have to.
    """
    annotation = config.get("genetic_tools", {}).get("pascalx", {}).get("genome_annotation", "")
    return {
        "describes": [
            "gene_scores_dtif.csv",
            "gene_scores_mtif.csv",
            "gene_scores_lv.csv.gz",
        ],
        "summary": (
            "Gene-level association scores were computed with PascalX from the GWAS "
            "summary statistics of each trait. This record states the software, the "
            "parameters and the reference data used, and accompanies two tables that "
            "carry the per-gene facts the score matrices do not: how many variants "
            "entered each gene, and why a score is absent where it is absent."
        ),
        "tool": dict(params["tool"]),
        "parameters": {
            "scoring": dict(params["scoring"]),
            "runtime": {
                "gpu": params["runtime"]["gpu"],
                # n_cpu is a throughput setting and does not affect the result; recorded
                # because "how was it run" is part of provenance, flagged so nobody reads
                # it as a scientific parameter.
                "n_cpu": params["runtime"]["n_cpu"],
                "note": "runtime settings; they change speed, not results",
            },
            "not_a_scoring_parameter": {
                # Stops a reader taking MAF = 0.05 for a scoring setting.
                "maf": (
                    "MAF = 0.05 appears in some PascalX configuration files but is not a "
                    "gene-scoring parameter: chi2sum() takes varcutoff, window and gpu "
                    "only. Allele frequency and imputation filtering happen upstream, when "
                    "the summary statistics are prepared."
                ),
            },
        },
        "input_summary_statistics": {
            "cohort": "UK Biobank, right eye; a population of mostly European "
                      "ancestry, with no ancestry filter applied",
            "genome_build": "GRCh37",
            "variant_filter": "MAF > 0.01 and INFO > 0.8",
            "deposited_separately": "NHGRI-EBI GWAS Catalog; this record does not duplicate them",
        },
        "reference_data": {
            "ld_reference_panel": {"name": "UK10K", "genome_build": "GRCh37"},
            # Basename only. The directory it lives in is ours and is not anybody's
            # business; the filename carries the build, the gene classes and the date,
            # which is the part that identifies the annotation.
            "genome_annotation": Path(annotation).name if annotation else "",
        },
        "coverage": coverage,
        "missing_values": {
            "convention": (
                "An empty cell in a score matrix means the gene has no score for that "
                "trait. It is never zero and never a p-value of 1."
            ),
            "how_to_resolve": (
                "Look the gene up in {annotation}. status = {no_snps} means no variant "
                "fell in the gene's window under the parameters above, so no trait has a "
                "score for it. status = {scored} means the gene was scorable, and an "
                "empty cell is then a per-trait failure: find it in {exceptions}."
            ).format(annotation=ANNOTATION_FILE, no_snps=STATUS_NO_SNPS,
                     scored=STATUS_SCORED, exceptions=EXCEPTIONS_FILE),
        },
        "resources": [
            {
                "name": ANNOTATION_FILE,
                "description": (
                    "One row per gene considered by the scorer. Trait-invariant: every "
                    "trait is scored against the same variant set, which this script "
                    "verifies across every trait it reads rather than assuming."
                ),
                "note_on_unscored_genes": (
                    "The genes with status {} carry no variant passing the upstream "
                    "MAF/INFO filter anywhere in their window. Inspected rather than "
                    "assumed: about half are clone-based identifiers from the annotation, "
                    "and the remainder are dominated by multi-copy paralogue families and "
                    "pseudogenes (AGAP4/8/9/10, ANKRD20A1-4, BMS1P17/18, ARL17A and the "
                    "like) -- regions where imputation quality is characteristically low, "
                    "so an INFO threshold removes their variants. They are absent from "
                    "every trait's scores by construction, not by failure."
                ).format(STATUS_NO_SNPS),
                "rows": n_genes,
                "fields": [
                    {"name": "gene", "type": "string",
                     "description": "Gene symbol as carried by the genome annotation above. "
                                    "Some are Vega/Havana clone-based identifiers rather "
                                    "than approved HGNC symbols."},
                    {"name": "n_snps", "type": "integer",
                     "description": "Variants assigned to the gene, i.e. falling within its "
                                    "span extended by the window parameter. 0 where status "
                                    "is " + STATUS_NO_SNPS + "."},
                    {"name": "status", "type": "string",
                     "enum": [STATUS_SCORED, STATUS_NO_SNPS],
                     "description": STATUS_SCORED + ": the gene was scorable. "
                                    + STATUS_NO_SNPS + ": no variant fell in its window, so "
                                    "it has no score for any trait."},
                ],
            },
            {
                "name": EXCEPTIONS_FILE,
                "description": (
                    "Per-trait scoring failures: a gene that was scorable in general but "
                    "for which the scorer raised on this trait. Rare -- typically none or "
                    "one per trait."
                ),
                "rows": n_exceptions,
                "fields": [
                    {"name": "feature_set", "type": "string",
                     "enum": list(FEATURE_SETS),
                     "description": "Which score matrix the trait belongs to."},
                    {"name": "trait", "type": "string",
                     "description": "Trait column name, without the trailing _p used in the "
                                    "score matrices."},
                    {"name": "gene", "type": "string", "description": "Gene symbol."},
                    {"name": "reason", "type": "string", "enum": ["scoring_failed"],
                     "description": "Why the score is absent."},
                    {"name": "detail", "type": "string",
                     "description": "Scorer diagnostics as recorded at run time, verbatim. "
                                    "Free text; no schema is promised."},
                ],
            },
        ],
        "generated_by": {
            "script": "tools/pascalx/emit_scoring_provenance.py",
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": (
                "Derived from the scorer's own per-trait output. Per-file checksums are "
                "in MANIFEST.tsv, which is the single authority for them."
            ),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[2])
    parser.add_argument("--feature-set", choices=FEATURE_SETS, action="append",
                        help="repeatable; default is --all")
    parser.add_argument("--all", action="store_true", help="all three feature sets")
    parser.add_argument("--tif-dir", type=Path, default=None)
    parser.add_argument("--lv-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--check", action="store_true",
                        help="report what would be written and exit without writing")
    args = parser.parse_args(argv)

    sets = list(FEATURE_SETS) if args.all else (args.feature_set or [])
    if not sets:
        parser.error("give --all or at least one --feature-set")

    config = load_config(args.config)
    with open(str(PARAMS), encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    profile = config["profiles"][config["active_profile"]]["paths"]
    tif_dir = args.tif_dir or Path(profile["tif_dir"])
    lv_dir = args.lv_dir or Path(profile["lv_dir"])
    labels = config["tif_label"]

    print("[INFO] profile: {}".format(config["active_profile"]))

    acc = Accumulator()
    coverage = {}
    for feature_set in sets:
        pairs = sidecar_paths(feature_set, tif_dir, lv_dir, labels)
        expected = expected_trait_count(feature_set, tif_dir, lv_dir, labels)
        print("[INFO] {}: reading {} of {} sidecars".format(feature_set, len(pairs), expected))
        for trait, path in pairs:
            with open(str(path), "rb") as handle:
                acc.add(feature_set, trait, pickle.load(handle))
        coverage[feature_set] = {
            "traits_in_matrix": expected,
            "traits_with_provenance": len(pairs),
            "complete": len(pairs) == expected,
        }
        if len(pairs) != expected:
            print("[WARN] {}: {} trait(s) have no .p sidecar; gene-level facts are "
                  "unaffected (they are trait-invariant), per-trait failures for those "
                  "traits are unknown".format(feature_set, expected - len(pairs)))

    # The invariants the three-file design rests on. Reported, never assumed -- if they
    # ever break, the honest output is a per-trait table, not a quietly wrong one.
    if acc.unknown_skip_reasons:
        print("[FAIL] the scorer reported skip reason(s) this script has no mapping for:")
        for reason, n in acc.unknown_skip_reasons.most_common():
            print("         {!r}: {} gene-trait pair(s)".format(reason, n))
        print("       Add them to SKIP_REASONS with a status of their own. Folding an "
              "unknown reason into '{}' would state something the run did not "
              "say.".format(STATUS_NO_SNPS))
        return 1

    if acc.n_snps_conflicts or acc.status_conflicts:
        print("[FAIL] the gene-level facts are NOT trait-invariant: "
              "{} gene(s) disagree on n_snps, {} on status.".format(
                  len(acc.n_snps_conflicts), len(acc.status_conflicts)))
        for gene, n in list(acc.n_snps_conflicts.most_common(5)):
            print("         n_snps: {} ({} disagreeing traits)".format(gene, n))
        print("       A single gene-level table would misrepresent this. Emit per-trait "
              "provenance instead, or explain the exception before depositing.")
        return 1

    n_scored = sum(1 for g in acc.status if acc.status[g] == STATUS_SCORED)
    n_no_snps = sum(1 for g in acc.status if acc.status[g] == STATUS_NO_SNPS)
    print("[INFO] genes: {} total, {} scorable, {} without variants in window".format(
        len(acc.status), n_scored, n_no_snps))
    print("[INFO] per-trait failures: {}".format(len(acc.exceptions)))
    print("[INFO] trait-invariance verified across {} sidecar(s)".format(
        sum(acc.traits_read.values())))

    if args.check:
        print("[INFO] --check: nothing written")
        return 0

    n_genes = write_annotation(acc, args.out_dir / ANNOTATION_FILE)
    n_exc = write_exceptions(acc, args.out_dir / EXCEPTIONS_FILE)
    record = build_provenance(acc, params, config, coverage, n_genes, n_exc)
    with open(str(args.out_dir / PROVENANCE_FILE), "w", encoding="utf-8") as handle:
        yaml.safe_dump(record, handle, sort_keys=False, allow_unicode=True, width=88)

    for name in (PROVENANCE_FILE, ANNOTATION_FILE, EXCEPTIONS_FILE):
        print("[INFO] wrote {}".format(args.out_dir / name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
