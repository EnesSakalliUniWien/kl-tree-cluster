#!/bin/sh
set -eu

cd "$(dirname "$0")"

mkdir -p build

if [ -x "./tools/tectonic" ]; then
  ./tools/tectonic --keep-logs --synctex --outdir build main.tex
  exit 0
fi

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -file-line-error -outdir=build main.tex
  exit 0
fi

if command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --synctex --outdir build main.tex
  exit 0
fi

if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
  (
    cd build
    bibtex main
  )
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
  exit 0
fi

echo "No LaTeX engine found." >&2
echo "Install one of: ./install_tectonic.sh, latexmk, tectonic, or pdflatex+bibtex." >&2
exit 1
