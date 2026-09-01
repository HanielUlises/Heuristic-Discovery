#!/bin/sh
# Rebuilds the figures used by the README.
#
# The TikZ sources are the originals; the SVGs beside them are generated and
# committed so that the README renders without a LaTeX toolchain. Regenerate
# after editing any .tex file.
#
# Requires pdflatex (with the tikz and standalone packages) and dvisvgm.
set -eu
cd "$(dirname "$0")"

for figure in admissibility landmarks partition; do
  pdflatex -interaction=nonstopmode -halt-on-error "${figure}.tex" >/dev/null
  dvisvgm --pdf --no-fonts --optimize=all --precision=3 \
          -o "${figure}.svg" "${figure}.pdf" >/dev/null 2>&1
  echo "${figure}.svg"
done

rm -f ./*.aux ./*.log ./*.pdf
