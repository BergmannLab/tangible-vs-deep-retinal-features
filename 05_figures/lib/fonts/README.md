# Nimbus Sans, carried with the code

The four faces the figures are typeset in, vendored so that a clone renders correctly on a
machine that has never heard of them. On a Mac it had to be installed by hand before this
directory existed, and the failure was silent: matplotlib substitutes DejaVu Sans without a
word, which is a different figure — different metrics, visibly heavier ink — that still looks
plausible enough to submit.

Nature Genetics asks for Helvetica or Arial. **Nimbus Sans is metric-compatible with
Helvetica**, which is why it is the family in `config.yaml` and why substituting it changes
nothing about the layout while installing it changes nothing about the licence of the figures.

| | |
|:--|:--|
| Upstream | [ArtifexSoftware/urw-base35-fonts](https://github.com/ArtifexSoftware/urw-base35-fonts) |
| Licence | AGPL-3.0 **with the font exception** — see `LICENSE-urw-base35.txt` |
| Faces | Regular, Bold, Italic, BoldItalic (~340 kB total) |

The font exception is the part that matters here: it permits including these programs in a
PostScript or PDF file "regardless of the conditions or license applying to the document
itself". Every figure this repository builds embeds them, and the exception is what keeps the
figure's own licence its own business.

## How they are found

- **Python** — `src/plot_style.py` registers this directory with matplotlib's font manager
  directly. matplotlib does not consult fontconfig, so nothing else would work.
- **R / cairo** — `pixi.toml` points `XDG_DATA_HOME` at `05_figures/lib`, and fontconfig
  looks for a `fonts/` directory under it. That is why this folder is named `fonts` and why
  it sits where it does; moving it silently returns you to DejaVu.
- **Neither, and the run stops.** `load_plot_style()` verifies that the family it asked for
  is the family matplotlib resolved, and fails with instructions rather than drawing.
