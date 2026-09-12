#!/bin/bash
# Build and install PascalX into this repository's pixi environment.
#
# PascalX ships no wheel: its numerics are a C++ core (Davies, Ruben, weighted
# chi-square sums in boost multiprecision) built by the upstream Makefile, plus a
# python package that reaches those libraries through cffi. So it cannot be a line
# in pixi.toml -- this script is the line in pixi.toml's place, and it pins the
# same commit that params.yaml asserts at run time.
#
# What it does:
#   1. clone github.com/BergmannLab/PascalX at the pinned commit (or reuse a
#      checkout you already have, via --src)
#   2. make   -- builds libruben / libdavies / libwchissum into <src>/build/lib
#   3. pip install <src>/python into .pixi/envs/default
#   4. print the library path to record in config.yaml
#
# Prerequisites: git, g++ (with libquadmath), and `pixi install` already run.
# boost headers come from the pixi environment, not the host.
#
# Usage:
#   pixi run tools/pascalx/install_pascalx.sh [--src <dir>] [--allow-mismatch]
#
#   --src              where to clone/find PascalX (default: <repo>/.pascalx/PascalX)
#   --allow-mismatch   proceed even if the checkout is not at the pinned commit

set -euo pipefail

CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(dirname "$(dirname "$CODE_DIR")")

src="$REPO_ROOT/.pascalx/PascalX"
allow_mismatch=0

while [ $# -gt 0 ]; do
	case "$1" in
		--src)             src="$2"; shift 2 ;;
		--allow-mismatch)  allow_mismatch=1; shift ;;
		-h|--help)         sed -n '2,26p' "$0"; exit 0 ;;
		*) echo "[ERROR] Unknown argument: $1" >&2; exit 2 ;;
	esac
done

# Mirror pixi.toml's [activation.env] explicitly, because this script addresses the
# environment's interpreter directly and so does not get pixi's activation when run
# bare. Without it the build can resolve setuptools out of the user site, and an
# inherited PYTHONPATH can do the same kind of damage.
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# The default pixi environment, addressed directly rather than through `pixi run`,
# so this script behaves the same whether or not it was invoked under pixi.
ENV_PREFIX="$REPO_ROOT/.pixi/envs/default"
ENV_PYTHON="$ENV_PREFIX/bin/python"
[ -x "$ENV_PYTHON" ] || {
	echo "[ERROR] $ENV_PYTHON not found. Run 'pixi install' first." >&2
	exit 1
}

# Single source of truth for the version/commit: the same file score_genes.py
# checks against, so the pin cannot drift between install time and run time.
read -r want_version want_commit < <("$ENV_PYTHON" - "$CODE_DIR/params.yaml" <<-'PY'
	import sys, yaml
	tool = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["tool"]
	print(tool["version"], tool["commit"])
PY
)
echo "[INFO] PascalX pin: version $want_version, commit $want_commit"

if [ -d "$src/.git" ]; then
	echo "[INFO] Reusing checkout: $src"
else
	echo "[INFO] Cloning into $src"
	mkdir -p "$(dirname "$src")"
	git clone https://github.com/BergmannLab/PascalX.git "$src"
	git -C "$src" checkout --quiet "$want_commit"
fi

have_commit=$(git -C "$src" rev-parse HEAD)
if [ "$have_commit" != "$want_commit" ]; then
	echo "[WARN] checkout is at $have_commit, pin is $want_commit" >&2
	[ "$allow_mismatch" -eq 1 ] || {
		echo "[ERROR] refusing to build a different version. Pass --allow-mismatch to override," >&2
		echo "        or 'git -C $src checkout $want_commit'." >&2
		exit 1
	}
fi

echo "[INFO] Building C++ core (boost headers from $ENV_PREFIX/include)"
# The upstream Makefile hardcodes its own CFLAGS and does not use them on every
# rule, so an -I cannot be injected through make variables. CPLUS_INCLUDE_PATH
# reaches every g++ invocation without patching upstream.
CPLUS_INCLUDE_PATH="$ENV_PREFIX/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" \
	make -C "$src"

lib_dir="$src/build/lib"
for lib in libruben libdavies libwchissum; do
	[ -f "$lib_dir/$lib.so" ] || { echo "[ERROR] $lib.so was not built" >&2; exit 1; }
done

echo "[INFO] Installing the python package into $ENV_PREFIX"
# --no-deps: pixi.toml already pins every runtime dependency at the versions the
# published results used. Letting pip resolve them here would silently move them.
CPLUS_INCLUDE_PATH="$ENV_PREFIX/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" \
	"$ENV_PYTHON" -m pip install --no-deps --no-build-isolation "$src/python"

echo ""
echo "[INFO] Verifying"
LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
	"$ENV_PYTHON" -c "
import PascalX, sys
print('  PascalX  ', PascalX.__version__)
print('  from     ', PascalX.__file__)
assert PascalX.__version__ == '$want_version', 'version mismatch after install'
assert PascalX.__file__.startswith('$ENV_PREFIX'), 'PascalX is NOT the one in the pixi environment'
"

echo ""
echo "[INFO] Done. Record the library path in config.yaml:"
echo ""
echo "    genetic_tools:"
echo "      pascalx:"
if [ "${lib_dir#$REPO_ROOT/}" != "$lib_dir" ]; then
	echo "        library_path: ./${lib_dir#$REPO_ROOT/}"
else
	echo "        library_path: $lib_dir"
fi
