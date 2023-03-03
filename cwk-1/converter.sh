# shellcheck disable=SC2045
# shellcheck disable=SC2006
# converts .bmp to .jpg to be put into LaTeX files
# shellcheck disable=SC2035
# shellcheck disable=SC2086
for i in *.bmp; do convert "${i}" ${i%bmp}jpg; done