# 05_figures/main — main-text figures

One script per main-text figure, named for the figure it produces. Each fetches its inputs
from the data deposit and writes the figure as PDF and PNG.

Figure 3 is built as four strips — `fig3_snp.R` (a–c), `fig3_gene.R` (d–f),
`fig3_polygenicity.py` (g–j) and `fig3_pathway.py` (k) — that `fig3.py` stacks into one page;
`fig3_build.py` runs all five.

**Figure 1** is the one figure with nothing to compute: it is hand-drawn artwork by
**Dennis Bontempi** ([@denbonte](https://github.com/denbonte)), carried here with his
permission. `fig1.py` renders it anyway, so that every main figure answers the same command
with the same two files. See `fig1_overview/README.md`.
