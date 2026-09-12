#!/bin/bash
# Reproduce the two-pass gene and pathway analysis end to end.
#
# This is the pipeline behind the gene and pathway tables of the Zenodo record and
# behind Figure 3 panels d, e, f, j and k. It is written to be READ as much as run: the whole thing is
# weeks of GPU time on 1058 traits x 2 passes, so it prints every command and runs
# nothing unless you pass --run.
#
# Usage:
#   run_two_pass_pathway_analysis.sh [--run] [--gpu <id>]
#                                    [--stage 1|2|3|4|5|6|all]
#
#   --run     actually execute. Without it, every command is printed and nothing runs.
#   --gpu     CUDA device id passed through to scoring (default 0)
#   --stage   run one stage only; default all. Stages are ordered and each depends on
#             the one before.
#
# THE TWO PASSES, and why the pipeline has this shape.
#
#   Pass 1 scored every gene and every one of the 31 120 MSigDB v7.2 gene sets against
#   every trait, on an earlier version of the phenotype files.
#
#   When the phenotypes were corrected, GENE scoring was repeated in full, but PATHWAY
#   scoring was not: it was repeated only for the gene sets that had already passed
#   Benjamini-Hochberg within their trait in pass 1. Figure 3j,k therefore shows a
#   recomputed value where one exists and a pass-1 value otherwise.
#
# The data record's README gives the measured cost of that shortcut and why its error
# runs in the conservative direction. Stage 4, the selection, is the step that decides
# which cells get recomputed, so reproducing the analysis means reproducing that choice.
#
# NOTE. The two passes read DIFFERENT profiles of config.yaml: pass 1 the `preprint`
# profile, pass 2 the active one. The scripts below take their directories from those
# profiles, so nothing here hardcodes a path.

set -euo pipefail

CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
GENETIC_DIR=$(dirname "$CODE_DIR")
REPO_ROOT=$(dirname "$GENETIC_DIR")
. "$GENETIC_DIR/env.sh"

do_run=0
which_gpu=0
stage="all"

while [ $# -gt 0 ]; do
	case "$1" in
		--run)   do_run=1; shift ;;
		--gpu)   which_gpu="$2"; shift 2 ;;
		--stage) stage="$2"; shift 2 ;;
		-h|--help) sed -n '2,36p' "$0"; exit 0 ;;
		*) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
	esac
done

cfg() { "$PY3" -c "
import yaml,sys
c=yaml.safe_load(open('$REPO_ROOT/config.yaml'))
p=c['profiles']['$1' if '$1'!='active' else c['active_profile']]['paths']
print(p['$2'])"; }

SCREEN_TIF=$(cfg preprint tif_dir)
SCREEN_LV=$(cfg preprint lv_dir)
ACTIVE_TIF=$(cfg active tif_dir)
ACTIVE_LV=$(cfg active lv_dir)
GMT_DIR="$REPO_ROOT/outputs/natgen_revision/intermediate/pathway_selection"

step() {
	echo ""
	echo "----------------------------------------------------------------------"
	echo "  $1"
	echo "----------------------------------------------------------------------"
	shift
	printf '  %q' "$@"; echo
	if [ "$do_run" -eq 1 ]; then "$@"; else echo "  (not run; pass --run)"; fi
}

want() { [ "$stage" = "all" ] || [ "$stage" = "$1" ]; }

# --- 1. gene scoring, both passes -------------------------------------------------
# Pathway scoring reuses the gene scores in its output directory, so genes come first
# in each pass or every pathway run pays for them again.
if want 1; then
	for d in "$SCREEN_TIF" "$SCREEN_LV" "$ACTIVE_TIF" "$ACTIVE_LV"; do
		step "1. gene scoring: $(basename "$d")" \
			"$CODE_DIR/run_gene_scoring.sh" --cohort ukb --input-dir "$d" \
			--output-dir "$d" --gpu "$which_gpu"
	done
fi

# --- 2. the deposited gene tables -------------------------------------------------
# Only the active pass is deposited: the paper's gene-level results are the corrected
# ones, and unlike the pathways the gene scores were redone in full.
if want 2; then
	step "2. aggregate gene scores (active pass) -> deposited gene-score matrices" \
		"$PY3" "$CODE_DIR/aggregate_gene_scores.py" --all
fi

# --- 3. pass 1: pathway scoring over all gene sets --------------------------------
if want 3; then
	for d in "$SCREEN_TIF" "$SCREEN_LV"; do
		step "3. pass-1 pathway scoring: $(basename "$d")" \
			"$CODE_DIR/run_pathway_scoring.sh" --cohort ukb --input-dir "$d" \
			--output-dir "$d" --gpu "$which_gpu"
	done
	step "3. aggregate the pass-1 screen -> deposited screen tables" \
		"$PY3" "$CODE_DIR/aggregate_pathway_scores.py" --all
fi

# --- 4. the selection -------------------------------------------------------------
# The step that decides which cells pass 2 recomputes. --verify checks the rebuilt
# selection against the .gmt files the original run used.
if want 4; then
	step "4. select gene sets by within-trait BH -> per-trait .gmt" \
		"$PY3" "$CODE_DIR/select_pathways_bh.py" --all --out-dir "$GMT_DIR"
	step "4. verify the selection against the original .gmt files" \
		"$PY3" "$CODE_DIR/select_pathways_bh.py" --all --verify
fi

# --- 5. pass 2: pathway scoring over the selection only ---------------------------
if want 5; then
	for d in "$ACTIVE_TIF" "$ACTIVE_LV"; do
		step "5. pass-2 pathway scoring: $(basename "$d")" \
			"$CODE_DIR/run_pathway_scoring.sh" --cohort ukb --input-dir "$d" \
			--output-dir "$d" --gpu "$which_gpu" --pathway-dir "$GMT_DIR"
	done
	step "5. aggregate the pass-2 recomputation -> deposited recomputation table" \
		"$PY3" "$CODE_DIR/aggregate_pathway_scores.py" --recomputed
fi

# --- 6. settings and logs ---------------------------------------------------------
if want 6; then
	step "6. collect settings and logs -> deposited settings and logs" \
		"$PY3" "$CODE_DIR/collect_run_provenance.py"
fi

echo ""
if [ "$do_run" -eq 1 ]; then
	echo "[INFO] done. Stage the deposit with:"
else
	echo "[INFO] nothing was run. Re-run with --run, or stage the current tables with:"
fi
echo "         the analysis pipeline's deposit-assembly step, with --force"
