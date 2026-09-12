# tools — our wrappers around third-party software

**Not a pipeline stage.** The numbered directories `00_cohort` … `04_replication` are the
analysis in the order the paper runs it. This one is orthogonal to them: it holds what we had
to write to drive somebody else's software, and the parameters we drove it with.

| | |
|:--|:--|
| `pascalx/` | Gene and pathway scoring. Runs on summary statistics, so it needs no cohort access, and `params.yaml` carries every parameter the published scoring used. |
| `ldsc/` | Heritability and genetic correlation. Needs a python 2.7 environment and an LD reference panel we do not redistribute, so it ships with the pipeline release rather than this one. |
| `cohort_config.py`, `env.sh` | What both wrappers read: tool paths from `config.yaml`, per-cohort column names from `replication.yaml`, and the python each shell wrapper should call. |

## Third-party tools

`ldsc/` and `pascalx/` hold **our wrappers and the exact parameters we ran**, not the tools
themselves. LDSC and PascalX are published third-party software, cited in the Methods and
installed as dependencies. See each subdirectory's README.

Getting both installed:

```bash
pixi install                                    # PascalX's dependencies, and everything else
pixi run tools/pascalx/install_pascalx.sh  # build PascalX at its pinned commit
pixi install -e ldsc                            # LDSC's python 2.7 environment
```

PascalX runs in the repository's ordinary environment; LDSC, being python 2.7, gets the one
standalone environment. `env.sh` resolves the python 3 interpreter for the wrappers' helpers,
so no wrapper here needs a `pixi run` prefix. Machine paths (reference panel, genome
annotation, MSigDB gene sets, LD scores, LDSC checkout) live in `config.yaml` under
`genetic_tools`; scientific
parameters live next to the code in `pascalx/params.yaml`, because they are published in the
Methods and must not vary by machine.

## What does not belong here

- Replication in independent cohorts → `04_replication/`
- Manhattan, QQ, Venn and forest *rendering* → `05_figures/`. This stage writes the plotting
  data; it does not draw.

## Outputs

Gene and pathway scores, heritability and genetic-correlation estimates, and the settings and
logs of every scoring run. All aggregate; the scores and the scoring provenance are in the
public data deposit. The per-trait summary statistics are deposited separately — see the
top-level README.
