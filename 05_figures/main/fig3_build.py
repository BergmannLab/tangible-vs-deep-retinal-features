#!/usr/bin/env python
"""Build Figure 3: the four strips, then the assembly. One command, in parallel.

    pixi run python 05_figures/main/fig3_build.py            # everything
    pixi run python 05_figures/main/fig3_build.py --only k    # just panel k, then assemble
    pixi run python 05_figures/main/fig3_build.py --png       # ... and the 300 dpi preview
    pixi run python 05_figures/main/fig3_build.py --double-log  # log2-log2 QQs (b, e)

Author: Michael Beyeler (github.com/mjbeyeler)

WHY THIS EXISTS. A full rebuild is four scripts and an assembly. Measured 9 Sep 2026:
snp 42.6 s, gene 20.9 s, pathway 7.4 s, polygenicity 5.4 s, then 15.4 s of Ghostscript
crop and a tenth of a second to merge -- about 95 s in sequence, 61 s here.

The strips are independent: each writes its own directory under
the results directory, and only the assembly reads all four. So they run
concurrently here, which takes the strip phase down to the slowest single script -- the
SNP strip, which is therefore the only thing worth optimising -- and the 300 dpi PNG
companion is off unless asked for. The Ghostscript crop cannot be overlapped: it runs on
the merged file, so it is a quarter of the wall clock by construction.

WHAT THIS DELIBERATELY DOES NOT DO is decide for itself which strips are stale. A cache
keyed on file times or on a guess at which config keys a strip reads is exactly the kind
of thing that silently ships a figure built from a stale panel; `--only` puts that
judgement in the hands of whoever knows what they changed, and the default is to rebuild
everything. The assembly always runs.

The two R strips need the `r` pixi environment and the two Python ones the default
environment, so each is launched through its own `pixi run`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# key -> (what it draws, the command that draws it)
STRIPS = {
    "snp": ("a b c", ["pixi", "run", "-e", "r", "Rscript",
                      "05_figures/main/fig3_snp.R"]),
    "gene": ("d e f", ["pixi", "run", "-e", "r", "Rscript",
                       "05_figures/main/fig3_gene.R"]),
    "polygenicity": ("g h i j", ["pixi", "run", "python",
                                 "05_figures/main/fig3_polygenicity.py"]),
    "pathway": ("k", ["pixi", "run", "python",
                      "05_figures/main/fig3_pathway.py"]),
}
# The strips carrying a QQ panel, and so the only ones --double-log means anything to.
QQ_STRIPS = {"snp", "gene"}

# Shorthands, so a panel letter picks its strip.
ALIASES = {"a": "snp", "b": "snp", "c": "snp",
           "d": "gene", "e": "gene", "f": "gene",
           "g": "polygenicity", "h": "polygenicity", "i": "polygenicity",
           "j": "polygenicity", "k": "pathway"}


def run(key, double_log=False):
    """Run one strip script, returning (key, seconds, returncode, output)."""
    panels, command = STRIPS[key]
    if double_log and key in QQ_STRIPS:
        command = command + ["--double-log"]
    started = time.time()
    done = subprocess.run(command, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return key, time.time() - started, done.returncode, done.stdout + done.stderr


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", default="",
        help="comma-separated strips or panel letters to rebuild (default: all). "
             "The assembly always runs, so `--only k` reassembles with the other three "
             "strips exactly as they are on disk.")
    parser.add_argument("--png", action="store_true",
                        help="also write the 300 dpi PNG companion")
    parser.add_argument("--jobs", type=int, default=4,
                        help="how many strips to build at once (default: all four)")
    parser.add_argument("--double-log", action="store_true",
                        help="draw the QQ panels (b, e) on log2-log2 axes instead of "
                             "linear ones. Recommended -- the bulk of the distribution is "
                             "legible and the y ticks match the Manhattans -- but not the "
                             "default, which reproduces the published panels.")
    args = parser.parse_args(argv)

    if args.only:
        wanted, unknown = [], []
        for name in (n.strip() for n in args.only.split(",") if n.strip()):
            key = ALIASES.get(name, name)
            if key not in STRIPS:
                unknown.append(name)
            elif key not in wanted:
                wanted.append(key)
        if unknown:
            raise SystemExit("unknown strip(s): {}. Use one of {} or a panel letter "
                             "a-k.".format(", ".join(unknown), ", ".join(STRIPS)))
    else:
        wanted = list(STRIPS)

    print("Building {} of Figure 3's 4 strips, {} at a time:".format(
        len(wanted), min(args.jobs, len(wanted))))
    for key in wanted:
        print("  {:14s} {}".format(key, STRIPS[key][0]))
    print()

    started = time.time()
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for key, seconds, code, output in pool.map(
                partial(run, double_log=args.double_log), wanted):
            status = "ok" if code == 0 else "FAILED"
            print("  {:14s} {:6.1f} s  {}".format(key, seconds, status))
            if code != 0:
                failed.append(key)
                print(output.rstrip())
    if failed:
        raise SystemExit("\n{} strip(s) failed: {}".format(len(failed), ", ".join(failed)))
    print("  strips         {:6.1f} s total (wall clock)\n".format(time.time() - started))

    assemble = ["pixi", "run", "python", "05_figures/main/fig3.py"]
    if not args.png:
        assemble.append("--no-png")
    done = subprocess.run(assemble, cwd=str(REPO_ROOT))
    print("\nWhole build: {:.1f} s".format(time.time() - started))
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
