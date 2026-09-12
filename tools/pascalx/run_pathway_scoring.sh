#!/bin/bash
# Score pathways for every trait in a directory.
#
# Same shape as run_gene_scoring.sh -- discover traits, skip those already scored,
# put PascalX's compiled libraries on LD_LIBRARY_PATH, score, log -- and it reuses
# the gene scores that run_gene_scoring.sh left in --output-dir, because scoring the
# genes is the expensive half. Run gene scoring first, or let score_pathways.py
# compute them (it writes them to the same place, so the next trait's run and any
# rerun pick them up).
#
# Usage:
#   run_pathway_scoring.sh --cohort <name> --input-dir <dir> --output-dir <dir>
#                          [--suffix <tail>] [--gpu <id>] [--traits <pattern>]
#                          [--mode ranking|permutation] [--pathway-file <gmt>]
#                          [--pathway-dir <dir>] [--aggregate <set>] [--force]
#
#   --suffix        input filename tail (default __with_rsids.tbl)
#   --gpu           CUDA device id (default 0)
#   --traits        shell pattern selecting trait labels, e.g. 'meta_tau1*' (quote it)
#   --mode          override params.yaml pathways.mode
#   --pathway-file  override genetic_tools.pascalx.pathway_file, for every trait
#   --pathway-dir   PER-TRAIT gene-set files: <dir>/<trait>__within_BH.gmt. This is how
#                   the second pass was run -- each trait scored against only the gene
#                   sets it had already passed BH on. A trait with no file there is
#                   SKIPPED, which is correct: no file means nothing was selected for it.
#                   Build the directory with select_pathways_bh.py. Mutually exclusive
#                   with --pathway-file.
#   --aggregate     after scoring, rebuild the deposited pathway tables from the
#                   per-trait outputs (aggregate_pathway_scores.py). Takes `screen` for
#                   a full run or `recomputed` for a selected one.
#   --force         rescore traits that already have pathway output

set -euo pipefail

CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
GENETIC_DIR=$(dirname "$CODE_DIR")
. "$GENETIC_DIR/env.sh"

cohort=""
input_dir=""
output_dir=""
suffix="__with_rsids.tbl"
which_gpu=0
traits="*"
mode=""
pathway_file=""
pathway_dir=""
aggregate=""
force=0

while [ $# -gt 0 ]; do
	case "$1" in
		--cohort)       cohort="$2"; shift 2 ;;
		--input-dir)    input_dir="$2"; shift 2 ;;
		--output-dir)   output_dir="$2"; shift 2 ;;
		--suffix)       suffix="$2"; shift 2 ;;
		--gpu)          which_gpu="$2"; shift 2 ;;
		--traits)       traits="$2"; shift 2 ;;
		--mode)         mode="$2"; shift 2 ;;
		--pathway-file) pathway_file="$2"; shift 2 ;;
		--pathway-dir)  pathway_dir="$2"; shift 2 ;;
		--aggregate)    aggregate="$2"; shift 2 ;;
		--force)        force=1; shift ;;
		-h|--help)      sed -n '2,22p' "$0"; exit 0 ;;
		*) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
	esac
done

[ -n "$cohort" ]     || { echo "[ERROR] --cohort is required" >&2; exit 2; }
[ -n "$input_dir" ]  || { echo "[ERROR] --input-dir is required" >&2; exit 2; }
[ -n "$output_dir" ] || { echo "[ERROR] --output-dir is required" >&2; exit 2; }
if [ -n "$pathway_file" ] && [ -n "$pathway_dir" ]; then
	echo "[ERROR] --pathway-file and --pathway-dir are mutually exclusive: one gene-set" >&2
	echo "        file for every trait, or one per trait. Pick which pass this is." >&2
	exit 2
fi
[ -z "$pathway_dir" ] || [ -d "$pathway_dir" ] || {
	echo "[ERROR] No such --pathway-dir: $pathway_dir" >&2; exit 1; }

[ -d "$input_dir" ] || { echo "[ERROR] No such input directory: $input_dir" >&2; exit 1; }
input_dir=$(realpath "$input_dir")
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")

# PascalX is built from source; its compiled libraries have to be on the dynamic
# linker path before python starts, which is why this cannot live in the python code.
eval "$("$PY3" "$GENETIC_DIR/cohort_config.py" --tool pascalx --emit-shell)"
if [ -n "${PASCALX_LIBRARY_PATH:-}" ]; then
	case ":${LD_LIBRARY_PATH:-}:" in
		*":$PASCALX_LIBRARY_PATH:"*) ;;
		*) export LD_LIBRARY_PATH="${PASCALX_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
	esac
fi

shopt -s nullglob
inputs=("$input_dir"/${traits}${suffix})
shopt -u nullglob

echo "[INFO] cohort:     $cohort"
echo "[INFO] input dir:  $input_dir"
echo "[INFO] output dir: $output_dir"
echo "[INFO] pattern:    ${traits}${suffix}  ->  ${#inputs[@]} file(s)"
echo "[INFO] pathways:   ${pathway_file:-${PASCALX_PATHWAY_FILE:-<config.yaml>}}"
echo "[INFO] GPU:        $which_gpu"
echo "[INFO] LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"

[ "${#inputs[@]}" -gt 0 ] || { echo "[ERROR] No inputs matching ${traits}${suffix} in $input_dir" >&2; exit 1; }

extra=()
[ -n "$mode" ] && extra+=(--mode "$mode")
[ -n "$pathway_file" ] && extra+=(--pathway-file "$pathway_file")

scored=0
skipped=0
failed=0

for input in "${inputs[@]}"; do
	base=$(basename "$input")
	trait=${base%%__*}
	target="$output_dir/${trait}__pathway_scores.txt"

	if [ "$force" -eq 0 ] && [ -f "$target" ]; then
		echo "[SKIP] $trait already scored"
		skipped=$((skipped + 1))
		continue
	fi

	# Per-trait gene-set file, the second pass's whole mechanism. Absent means the
	# trait had nothing selected, which is a reason to skip it and not an error.
	trait_extra=()
	if [ -n "$pathway_dir" ]; then
		gmt="$pathway_dir/${trait}__within_BH.gmt"
		if [ ! -f "$gmt" ]; then
			echo "[SKIP] $trait has no selected gene sets in $pathway_dir"
			skipped=$((skipped + 1))
			continue
		fi
		trait_extra+=(--pathway-file "$gmt")
	fi

	echo ""
	echo "[INFO] === $trait === $(date '+%Y-%m-%d %H:%M:%S')"
	# -u: unbuffered, so progress reaches the log while a long trait is running.
	if CUDA_VISIBLE_DEVICES="$which_gpu" "$PY3" -u "$CODE_DIR/score_pathways.py" \
		--sumstats "$input" --cohort "$cohort" --output-dir "$output_dir" \
		"${extra[@]}" ${trait_extra[@]+"${trait_extra[@]}"}; then
		scored=$((scored + 1))
	else
		echo "[WARN] pathway scoring failed for $trait" >&2
		failed=$((failed + 1))
	fi
done

echo ""
echo "[INFO] scored=$scored skipped=$skipped failed=$failed"
[ "$failed" -eq 0 ] || exit 1

# Rebuild the deposited tables from what was just scored. After the failure check, for
# the same reason as in run_gene_scoring.sh: a table built over a run in which a trait
# failed would carry a stale or absent column with nothing to show it.
if [ -n "$aggregate" ]; then
	echo ""
	echo "[INFO] aggregating pathway scores ($aggregate)"
	case "$aggregate" in
		screen)     "$PY3" "$CODE_DIR/aggregate_pathway_scores.py" --all ;;
		recomputed) "$PY3" "$CODE_DIR/aggregate_pathway_scores.py" --recomputed ;;
		*) echo "[ERROR] --aggregate takes screen or recomputed, not '$aggregate'" >&2; exit 2 ;;
	esac
	"$PY3" "$CODE_DIR/collect_run_provenance.py"
fi
