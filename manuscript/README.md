# Manuscript Workspace

This directory contains a LaTeX workspace for drafting a manuscript about the
current KL-TE clustering method.

## Files

- `main.tex`: manuscript entry point
- `sections/`: section-level draft files
- `references.bib`: BibTeX database
- `build.sh`: build helper with engine autodetection
- `Makefile`: convenience targets

## Build

If one of the supported TeX engines is installed, run:

```bash
cd manuscript
make pdf
```

If you want a repo-local compiler without installing a system-wide TeX
distribution, run:

```bash
cd manuscript
./install_tectonic.sh
make pdf
```

Supported build paths:

- `./tools/tectonic`
- `latexmk`
- `tectonic`
- `pdflatex` + `bibtex`

If no engine is installed, `build.sh` exits with a clear error message and the
source tree remains ready to compile once TeX tooling is added.

## Draft Status

The current draft is intentionally front-loaded toward method description.
Results, figures, and the full calibration study still need to be added.
