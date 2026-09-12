# tools/pascalx — gene and pathway scoring

PascalX aggregates SNP-level summary statistics into gene-level p-values, accounting for
LD from a reference panel. It is **third-party software** and is cited in the Methods; this
directory contains only our wrapper and the parameters we ran, so that the scoring is
reproducible rather than merely described.

## Tool version

| | |
|---|---|
| Package | PascalX **0.0.5** |
| Source | <https://github.com/BergmannLab/PascalX> |
| Commit | `79ecec69cfc119dca2a4f19c3c43ae0fef57767b` |

0.0.5 is the newest tag on that repository and identical to its `main` HEAD (checked
13 Aug 2026).

## Installation

PascalX is not on PyPI and has no wheel: its numerics are a C++ core (Davies, Ruben and
weighted chi-square sums in boost multiprecision) built by the upstream Makefile, reached from
python through cffi. So it cannot be a line in `pixi.toml`. This is:

```bash
pixi install                                  # the repo's environment
pixi run tools/pascalx/install_pascalx.sh # clone at the pinned commit, make, pip install
```

The script clones into `.pascalx/` (gitignored), refuses to build a commit other than the one
`params.yaml` pins, builds against the **boost headers in the pixi environment** so the host
does not need `libboost-dev`, and installs the python package into `.pixi/envs/default`.
It then prints the `build/lib` path to record as `genetic_tools.pascalx.library_path` in
`config.yaml`. Host prerequisites are only `git` and `g++` (with `libquadmath`).

PascalX runs in the repository's ordinary pixi environment — the pins there
(numpy 1.24.4 < 1.25, scipy 1.10.1 < 1.11, matplotlib 3.7.5, cffi, sortedcontainers,
fastnumbers, progressbar, tqdm, seaborn) already satisfy it, and `cupy-cuda117 10.6.0` is the
CUDA stack the published scores came from. Only LDSC, being python 2.7, gets an environment of
its own.

Its compiled libraries must be on `LD_LIBRARY_PATH` **before python starts**, which is why
`run_gene_scoring.sh` and not `score_genes.py` sets it, from
`genetic_tools.pascalx.library_path`.

`score_genes.py` **aborts if the imported version is not the pinned one**, rather than
scoring with a different PascalX.

## Scoring parameters as run

Declared in `params.yaml`, snapshotted next to every output.

| Parameter | Value |
|---|---|
| method | `saddle` |
| autorescore | true |
| variance cutoff | 0.99 |
| gene window | 50 000 bp |
| gene types | `protein_coding`, `lincRNA` |
| reference panel | UK10K, GRCh37 |

**There is no MAF parameter here.** PascalX's `chi2sum()` takes only `varcutoff`, `window`
and `gpu`. Allele-frequency and imputation-quality filtering happens **upstream**, when the
summary statistics are prepared (MAF > 0.01, INFO > 0.8 for the UK Biobank discovery set).

## Inputs

Summary statistics with an **rsid column that is not the last column** (PascalX misparses a
trailing rsid column; `score_genes.py` checks and refuses). Column names are read per cohort
from `replication.yaml`, so no file needs editing to score a different cohort.

For Rotterdam that means `*__with_rsids.tbl`, not the raw METAL `.tbl`, which has no rsid.

## Usage

Everything runs under `pixi run`, which is what puts PascalX and its pinned dependencies on
`sys.path`.

```bash
# one trait
pixi run python tools/pascalx/score_genes.py \
    --sumstats <trait>__with_rsids.tbl --cohort RotterdamStudy --output-dir <dir>

# every trait in a directory, skipping those already scored
pixi run tools/pascalx/run_gene_scoring.sh --cohort RotterdamStudy \
    --input-dir <sumstats dir> --output-dir <dir> --gpu 0

# pathways, reusing the gene scores that run left in --output-dir
pixi run tools/pascalx/run_pathway_scoring.sh --cohort RotterdamStudy \
    --input-dir <sumstats dir> --output-dir <dir> --gpu 0
```

## Outputs

| File | Contents |
|---|---|
| `<trait>__gene_scores` | gene ⟶ p-value, tab-separated, one line per gene |
| `<trait>__gene_scores.p` | pickled `(scores, failed_genes)` as PascalX returned them |
| `<trait>__gene_scores.params.json` | every parameter, path, version and host this run used |

The `.params.json` carries the provenance with the scores, so it is not left behind when
files are moved.

Gene scores are consumed by `04_replication/` and by the gene-level panels in `05_figures/`.
They cross the pipeline–figure boundary and are therefore part of the public data deposit.

## Pathway scoring

A pathway score aggregates the gene scores of its members, so `score_pathways.py` is the same
scorer with a second stage on top. The gene scores are the expensive half — the wrapper
**reuses** `<trait>__gene_scores` when it is already in `--output-dir` and computes (and saves)
them only when it is not, so gene and pathway scoring compose instead of duplicating work.

| Parameter | Value | |
|---|---|---|
| mode | `ranking` | `pathway.chi2rank`; `permutation` selects `chi2perm` |
| fuse | true | genes closer than `mergedist` are merged into one unit, so a gene-dense locus is not counted many times |
| mergedist | 100 000 bp | PascalX's own default |
| gene sets | MSigDB v7.2, symbols | `genetic_tools.pascalx.pathway_file`; third-party, not redistributed |
| .gmt layout | name in column 0, members from column 2 | the MSigDB layout, passed to `load_modules` |

`n_samplings` (10⁶) is read **only** in permutation mode; the ranking runs never use it.

Output is `<trait>__pathway_scores.txt` — `pathway  n_genes  n_scored  p` — plus the pickled
result and a `.params.json`, in the same shape as gene scoring.

### Reproducing an earlier run

Re-scoring `pred_tau1a` from the published July-2024 run (its own gene scores, its own genome
annotation, a 100-pathway MSigDB subset) reproduces it: **membership counts identical for all
100 pathways**, p-values agreeing to median |Δlog₁₀p| = 1.3 × 10⁻³, max 1.5 × 10⁻². The residual
is numerical drift — chi2rank scores the fused meta-genes on the fly, and those come out
fractionally different on this boost/CUDA stack.

Which annotation is used is not a detail: with the current 2025-10-01 annotation instead of the
one that run used, 74 of the 100 pathways gain or lose members and the scores move accordingly.
Hence `--genome-annotation`, for reproducing an older result against the annotation it was
scored with.

## Not here

Circular-shift null permutations and gene-score QQ plotting are separate from scoring; QQ
rendering belongs in `05_figures/`.
