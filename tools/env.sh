# Shared interpreter resolution for the tools bash wrappers. Source it:
#
#     GENETIC_DIR=$(dirname "$CODE_DIR")
#     . "$GENETIC_DIR/env.sh"
#     "$PY3" "$GENETIC_DIR/cohort_config.py" --tool ldsc --emit-shell
#
# The helpers these wrappers call (cohort_config.py, collect_h2.py, score_genes.py)
# are python 3 and need pyyaml. Leaving that to whatever `python3` happens to be on
# PATH means the wrappers work here and fail on a machine whose system python has no
# pyyaml -- so prefer the repository's own pixi environment when it exists, and say
# which interpreter was chosen rather than deciding silently.
#
# PYTHONNOUSERSITE mirrors pixi.toml's [activation.env]: it must be set even when the
# environment is addressed directly rather than through `pixi run`, or ~/.local
# shadows the pinned packages and the environment becomes cosmetic.

_genetic_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
_repo_root=$(dirname "$_genetic_dir")

if [ -x "$_repo_root/.pixi/envs/default/bin/python" ]; then
	PY3="$_repo_root/.pixi/envs/default/bin/python"
	export PYTHONNOUSERSITE=1
else
	PY3=$(command -v python3 || true)
	[ -n "$PY3" ] || { echo "[ERROR] no python3 found; run 'pixi install'" >&2; exit 1; }
fi
export PY3

unset _genetic_dir _repo_root
