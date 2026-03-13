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

## VS Code / LaTeX Workshop

If you use VS Code with LaTeX Workshop, open the repository root rather than
only the `manuscript/` directory. The workspace settings in
`../.vscode/settings.json` configure LaTeX Workshop to:

- build through `manuscript/build.sh`
- use `manuscript/build` as the output directory
- resolve section files back to `main.tex`

With that setup, building from `main.tex` or from any file in `sections/`
should target the same manuscript PDF.

## Draft Status

The current draft is intentionally front-loaded toward method description.
Results, figures, and the full calibration study still need to be added.
