# 05_figures — figure generation

**Tier: figures. This is the directory most readers want.**

Everything here reads **only** the deposited intermediate tables — no access-controlled data,
no genotypes, no GPU, no cluster. Download the data deposit, point the configuration at it,
and every panel in the paper regenerates in seconds on a laptop.

That boundary is deliberate. `00_cohort/` through `04_replication/` do the analysis and write
the numbers; this directory renders them. No script here fits a model, scores a gene, or reads
a participant-level row.

## Layout

- `main/` — main-text figures
- `lib/` — shared helpers the figure scripts source or shell out to

Scripts here are a mix of R and Python, and the R ones shell out to Python for two things:
fetching the deposit (`lib/fetch_data.R` → `src/fetch_data.py`) and drawing the feature-set
Venn from its deposited counts (`lib/render_venn3.py`). With `pixi` the right interpreter is
found automatically; otherwise set `MVP_PYTHON` to one carrying this project's dependencies.

## Conventions

Every figure in the paper follows the same rules. They are set in `config.yaml` and read
through `src/plot_style.py`.

| | |
|:--|:--|
| **Page** | Display items are 179 mm wide at most and 260 mm tall, the journal's limit. Widths come from `plot_styles.natgen`. |
| **Type** | One sans face throughout, Nimbus Sans, which is metric-compatible with the Helvetica the journal asks for and ships in `lib/fonts/`. Body text sits in a 5–7 pt band; panel letters are 8 pt bold. Mathtext is set in the same face. |
| **Colour** | The four feature-set colours are set in `config.yaml → colors`. They mean *feature set* and nothing else. |
| **p-values** | Typeset as `$m \times 10^{e}$` to one significant digit, never `3.2e-09`. |
| **Colour bars** | One size paper-wide, a thin outline, end labels, and tick marks on the label side. One switch, `plot_styles.colorbar_ticks`. |
| **Vector** | Every figure is written as PDF alongside its PNG, with SVG inputs embedded as paths rather than rasterised. |

**R panels** were re-verified under R 4.4 with the packages `pixi.toml` pins, which is a newer
stack than the one that produced the submitted figures (most visibly ggplot2 4.0.3 against
3.5.1); the differences were checked figure by figure rather than assumed away.

## Relationship to the data deposit

The inputs consumed by this directory *are* the public data deposit — the deposit is defined
by exactly this dependency edge, and doubles as the per-figure source data accompanying the
paper. If a figure here cannot find its input, the input is missing from the deposit; that is
a packaging bug, not a data-access limitation.
