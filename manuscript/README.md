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

## Prose Lint

This workspace also supports manuscript-only prose linting with Vale.

To install a repo-local Vale binary:

```bash
cd manuscript
./install_vale.sh
```

To sync styles and lint the manuscript prose:

```bash
cd manuscript
make prose-lint
```

The current starter setup uses:

- `write-good`
- `alex`
- local `KLTE` terminology and weak-phrase rules

The local `KLTE` rules intentionally flag repository-internal labels such as
`Gate 2`, `Gate 3`, `TreeBH`, `JL`, and `erank` so the manuscript can replace
them with clearer prose. The packaged styles are intentionally narrowed for
LaTeX math writing, so the lint output focuses on terminology and weak phrasing
instead of generic passive-voice or spelling noise.

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
