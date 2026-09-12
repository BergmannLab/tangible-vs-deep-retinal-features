#!/bin/bash
# Score genes for every trait in a directory.
#
# Thin loop around score_genes.py: discover traits, skip those already scored,
# set up the environment PascalX needs, score, log. Circular-shift null generation,
# QQ plotting and output relocation are separate steps, not part of gene scoring.
#
# Usage:
#   run_gene_scoring.sh --cohort <name> --input-dir <dir> --output-dir <dir>
#                       [--suffix <tail>] [--gpu <id>] [--traits <pattern>] [--force]
#                       [--aggregate <dtif|mtif|lv>]
#
#   --suffix   input filename tail (default __with_rsids.tbl)
#   --gpu      CUDA device id (default 0); PascalX only uses a non-default GPU when
#              CUDA_VISIBLE_DEVICES is set explicitly
#   --traits   shell pattern selecting trait labels, e.g. 'meta_tau1*'
#              (quote it, or the shell expands it first)
#   --force    rescore traits that already have output
#   --aggregate  after scoring, rebuild that feature set's deposited gene x trait matrix
#              from the per-trait outputs (aggregate_gene_scores.py). Repeatable.
#              Pass it for the three UK Biobank discovery sets, whose matrices are
#              Zenodo deposit items -- without it a rescore leaves the deposited table
#              behind the scores it claims to hold. Omit it for cohorts that have no
#              such table (Rotterdam, CoLaus, the left-eye replication).

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
force=0
aggregate=()

while [ $# -gt 0 ]; do
	case "$1" in
		--cohort)     cohort="$2"; shift 2 ;;
		--input-dir)  input_dir="$2"; shift 2 ;;
		--output-dir) output_dir="$2"; shift 2 ;;
		--suffix)     suffix="$2"; shift 2 ;;
		--gpu)        which_gpu="$2"; shift 2 ;;
		--traits)     traits="$2"; shift 2 ;;
		--force)      force=1; shift ;;
		--aggregate)  aggregate+=("$2"); shift 2 ;;
		-h|--help)    sed -n '2,29p' "$0"; exit 0 ;;
		*) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
	esac
done

[ -n "$cohort" ]     || { echo "[ERROR] --cohort is required" >&2; exit 2; }
[ -n "$input_dir" ]  || { echo "[ERROR] --input-dir is required" >&2; exit 2; }
[ -n "$output_dir" ] || { echo "[ERROR] --output-dir is required" >&2; exit 2; }

[ -d "$input_dir" ] || { echo "[ERROR] No such input directory: $input_dir" >&2; exit 1; }
input_dir=$(realpath "$input_dir")
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")

# PascalX is built from source; its compiled libraries have to be on the dynamic
# linker path before python starts, which is why this cannot live in the python code.
eval "$("$PY3" "$GENETIC_DIR/cohort_config.py" --tool pascalx --emit-shell)"
if [ -n "${PASCALX_LIBRARY_PATH:-}" ]; then
	# Idempotent: re-running in a shell that already exported it must not stack copies.
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
echo "[INFO] GPU:        $which_gpu"
echo "[INFO] LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"

[ "${#inputs[@]}" -gt 0 ] || { echo "[ERROR] No inputs matching ${traits}${suffix} in $input_dir" >&2; exit 1; }

scored=0
skipped=0
failed=0

for input in "${inputs[@]}"; do
	base=$(basename "$input")
	trait=${base%%__*}
	target="$output_dir/${trait}__gene_scores"

	if [ "$force" -eq 0 ] && [ -f "$target" ]; then
		echo "[SKIP] $trait already scored"
		skipped=$((skipped + 1))
		continue
	fi

	echo ""
	echo "[INFO] === $trait === $(date '+%Y-%m-%d %H:%M:%S')"
	# -u: unbuffered, so progress reaches the log while a multi-hour trait is running
	# rather than arriving all at once when it finishes.
	if CUDA_VISIBLE_DEVICES="$which_gpu" "$PY3" -u "$CODE_DIR/score_genes.py" \
		--sumstats "$input" --cohort "$cohort" --output-dir "$output_dir"; then
		scored=$((scored + 1))
	else
		echo "[WARN] scoring failed for $trait" >&2
		failed=$((failed + 1))
	fi
done

echo ""
echo "[INFO] scored=$scored skipped=$skipped failed=$failed"
[ "$failed" -eq 0 ] || exit 1

# Rebuild the deposited gene x trait matrix from what was just scored. Deliberately
# after the failure check: aggregating a run in which a trait failed would produce a
# matrix with a stale or absent column and no sign that anything was wrong.
#
# The feature set is read out of THIS run's --output-dir rather than out of the active
# profile, so the matrix holds the scores this invocation produced. Rescoring a profile
# that is not the active one therefore cannot quietly rewrite the active profile's table.
if [ "${#aggregate[@]}" -gt 0 ]; then
	for set_name in "${aggregate[@]}"; do
		case "$set_name" in
			dtif|mtif) dir_flag="--tif-dir" ;;
			lv)        dir_flag="--lv-dir" ;;
			*) echo "[ERROR] --aggregate takes dtif, mtif or lv, not '$set_name'" >&2; exit 2 ;;
		esac
		echo ""
		echo "[INFO] aggregating $set_name gene scores"
		"$PY3" "$CODE_DIR/aggregate_gene_scores.py" \
			--feature-set "$set_name" "$dir_flag" "$output_dir"
	done
fi
