<div align="center">

# Comparing tangible retinal image characteristics with deep learning features reveals their complementarity for gene association and disease prediction

Code by **Michael Beyeler** [@mjbeyeler](https://github.com/mjbeyeler) and
**David Presby** [@presbyd](https://github.com/presbyd)

This work is embedded in the [**VascX Consortium**](#more-work-from-the-vascx-consortium), which studies the vasculature across scales, from capillaries to conduit arteries.

[![Preprint](https://img.shields.io/badge/medRxiv-Preprint-BD2736?style=flat-square)](https://doi.org/10.1101/2024.12.23.24319548)
[![Data](https://img.shields.io/badge/Zenodo-Figure%20source%20data-1682D4?style=flat-square)](https://doi.org/10.5281/zenodo.22688789)
[![Summary statistics](https://img.shields.io/badge/Summary%20statistics-1%E2%80%AF109%20studies-0A6EAA?style=flat-square)](https://drive.google.com/drive/folders/1JxgQlr_tAQkBa50OanwA7Ke6XWXLg3o8)
[![License](https://img.shields.io/badge/Code-GPL--3.0-6E7781?style=flat-square)](LICENSE)
[![Powered by Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square)](https://pixi.sh)

</div>

---

Everything behind the Article. Every main figure regenerates from one deposited data record,
and every tool, model and deposit that produced the results is linked and pinned below.
Reproduce it, alter it, or build on top.

## Quick start

<a href="https://pixi.sh/latest/installation/"><img src="https://raw.githubusercontent.com/prefix-dev/pixi/main/docs/assets/pixi-logo.svg" alt="Install pixi" height="30" align="right"></a>

```bash
git clone https://github.com/BergmannLab/tangible-vs-deep-retinal-features
cd tangible-vs-deep-retinal-features
pixi install                                     # locked; nothing to resolve
pixi run python 05_figures/main/fig3_build.py    # fetches its data, writes results/
```

**Not a pixi user?** It resolves from the same conda channels, and the Python figure
stack is a short list of pinned wheels. Create a Python 3.8
environment with conda, mamba, uv or venv and `pip install -r requirements.txt` for the
same versions — `pixi.lock` is simply the one path that also pins them by build.

R panels run the same way:

```bash
pixi run -e r Rscript 05_figures/main/fig3_snp.R
```

---

## What is here

```
05_figures/
├── main/            one script per main figure (Python and R)
└── lib/             shared helpers the figure scripts call
tools/pascalx/       gene and pathway scoring: our PascalX wrappers, the installer that
                     builds it from the pinned commit, and params.yaml, which carries every
                     parameter the published scoring ran with
results/
└── main_figures/    vectorized publication figures
src/                 plot style, label placement, data fetching, configuration
config.yaml          palettes, page geometry, type sizes, data locations
pixi.toml pixi.lock  the environment, pinned to the published versions
```

---

## The data

| | What | Where |
|:--|:--|:--|
| 📊 | **Figure source data and result tables** — every panel's values, and past them a **25 819-gene × 1 058-trait** score matrix and **31 120 scored pathways**. Look up your gene, test your pathway, mine it for targets — no GWAS of your own required | Zenodo [`10.5281/zenodo.22688789`](https://doi.org/10.5281/zenodo.22688789) |
| 🧬 | **Full GWAS summary statistics** — 1 109 genome-wide studies: 17 measured traits, 17 deep traits and 1 024 latent dimensions. Genetic correlation, Mendelian randomisation or colocalisation against any of them. GRCh37, *n* = 41 526 | [Google Drive](https://drive.google.com/drive/folders/1JxgQlr_tAQkBa50OanwA7Ke6XWXLg3o8) · GWAS Catalog accession to follow |
| ✨ | **Fine-tuned RETFound weights** — extracts 17 interpretable vascular traits and 1 024 latent dimensions from a colour fundus photograph: the feature layer our blood pressure and BMI models are built on, and a starting point for endpoints we never examined | [🤗 PresbyD/RETFOUND_TIFs](https://huggingface.co/PresbyD/RETFOUND_TIFs) |

Individual-level data — images, per-participant traits, covariates — is governed by the
UK Biobank, Rotterdam Study and CoLaus agreements; apply to each cohort.

> [!TIP]
> **These GWAS share their participants.** The measured traits, the deep traits and the
> 1 024 latent dimensions are all measured in the same UK Biobank eyes, so genetic
> correlations among them are within-sample, Mendelian randomisation across them is not
> two-sample, and they overlap with any other UK Biobank study you pair them with. The
> released weights are the way around it: run them on your own genotyped cohort, get the
> same 1 024 dimensions, and replicate or meta-analyse there.

---

## The software behind the results

Every step is published software, pinned to the version we ran.

#### Retinal feature extraction

| Software | Role | Pinned at |
|:--|:--|:--|
| [**Retina-phenotypes**](https://doi.org/10.1038/s41467-024-52334-1) | segmentation and morphometry → mTIFs — described in Ortín Vela, Beyeler *et al.*, *Nat Commun* **15** | 2024 |
| [**lwnet**](https://github.com/mjbeyeler/lwnet) — our fork of [agaldran/lwnet](https://github.com/agaldran/lwnet) | artery/vein masks | `d1beff2` |
| [**optic-nerve-cnn**](https://github.com/mjbeyeler/optic-nerve-cnn) — our fork of [seva100/optic-nerve-cnn](https://github.com/seva100/optic-nerve-cnn) | optic-disc centre and size | `403aefa` |
| [**ARIA**](https://doi.org/10.1371/journal.pone.0032435) | centrelines, diameters, tortuosity — vendored in the pipeline, needs MATLAB | `328853d` |

> [!NOTE]
> The two forks above are ours only in the packaging. Please cite the upstream work, not the forks.

#### Genomics

| Software | Role | Pinned at |
|:--|:--|:--|
| [**BGENIE**](https://jmarchini.org/software/) | the GWAS | `v1.3` |
| [**LDSC**](https://github.com/bulik/ldsc) | heritability, genetic correlation | `v1.0.1` · `aa33296` |
| [**PascalX**](https://github.com/BergmannLab/PascalX) | gene and pathway scoring | `0.0.5` · `79ecec69` |
| [**logarithmic-thinning**](https://github.com/BergmannLab/logarithmic-thinning) | Manhattan and QQ point reduction | `v1.0.0` |
| [**MSigDB**](https://www.gsea-msigdb.org/) · **UK10K** | gene sets · LD reference panel | `v7.2` · GRCh37 |

Model and analysis parameters are in the Article's Methods; machine-specific locations are in
`config.yaml`.

---

## Full author list

Michael Beyeler ([@mjbeyeler](https://github.com/mjbeyeler)), Olga Trofimova ([@ot710](https://github.com/ot710)), Dennis Bontempi ([@denbonte](https://github.com/denbonte)), José Vargas Quiros ([@josedvq](https://github.com/josedvq)), Ilenia Meloni, Bart Liefers, Adham Elwakil, Sacha Bors ([@SachaBors](https://github.com/SachaBors)), Ian Quintas ([@laquinte1906](https://github.com/laquinte1906)), Leah Böttger ([@LeahBoettger](https://github.com/LeahBoettger)), Ilaria Iuliani ([@ilaria-iuliani](https://github.com/ilaria-iuliani)), Sofía Ortín Vela ([@OVSofia](https://github.com/OVSofia)), Federica Conedera, Mattia Tomasoni ([@mattiat](https://github.com/mattiat)), Ciara Bergin, Reinier Schlingemann, Caroline Klaver, [VascX Consortium](#more-work-from-the-vascx-consortium), David Presby ([@presbyd](https://github.com/presbyd)), Sven Bergmann ([@ftsven](https://github.com/ftsven)).

## More work from the VascX Consortium

Adjacent studies from the same collaboration.

| | | |
|:--|:--|:--|
| 🕸️ | **VascX vessel models.** Improved pixel-wise segmentation of arteries, veins, optic disc and fovea, and the vascular feature extraction built on it. *Vargas Quiros et al.*, TVST 2025 | [**Paper**](https://doi.org/10.1167/tvst.14.7.19) · [**Models**](https://github.com/Eyened/rtnls_vascx_models) |
| ⏳ | **Retinal biological age, and the sex-specific age gap.** The residual of predicted against chronological age as an ageing marker, with its clinical and genetic signatures resolved separately in women and men. *Trofimova et al.*, Nat Commun 2026 | [**Paper**](https://doi.org/10.1038/s41467-026-77102-1) · [**Code**](https://github.com/ot710/retinal-age) |
| ♀️♂️ | **Sex-specific disease association and genetic architecture.** Retinal vascular traits analysed per sex rather than pooled, where averaging the sexes conceals the effect. *Böttger et al.* 2025 | [**Preprint**](https://doi.org/10.1101/2025.07.16.665150) · [**Code**](https://github.com/BergmannLab/SexStrat) |
| 🔬 | **Capillary-level phenotyping by OCT-A.** Optical coherence tomography angiography resolves capillary networks that fundus photography cannot, and this establishes how reproducibly AI-derived measurements of them transfer between devices and centres. *De Clerck et al.* 2026, Jules-Gonin | [**Preprint**](https://doi.org/10.21203/rs.3.rs-9106265/v1) |
| 📡 | **From microvasculature to a conduit artery.** Carotid lumen diameter segmented from B-mode ultrasound, then its genetic architecture and cardiovascular associations, at a vessel scale the retina cannot reach. *Ortín Vela et al.* 2025 | [**Preprint**](https://doi.org/10.1101/2025.01.07.25320106) |

---

## Citation

Please cite the Article. Author affiliations and machine-readable metadata are in
[`CITATION.cff`](CITATION.cff).

## License

Copyright © 2026 Michael Beyeler, David Presby and Sven Bergmann — free software, copyleft
under **GPL-3.0** ([`LICENSE`](LICENSE)).

> [!IMPORTANT]
> The fine-tuned weights inherit upstream *RETFound*'s **CC BY-NC 4.0** — attribution
> required, **non-commercial use only**. They are linked here, not redistributed.
