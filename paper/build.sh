# sort the bib file
biber --tool --output_align --output_indent=2 --output_fieldcase=lower paper.bib -O paper.bib

docker run --rm \
  --volume $PWD:/data \
  --user $(id -u):$(id -g) \
  --env JOURNAL=joss \
  openjournals/inara \
  -o pdf,preprint \
  paper.md

latexmk -pdf paper.preprint.tex
rm paper.preprint.aux
rm paper.preprint.fdb_latexmk
rm paper.preprint.fls
rm paper.preprint.log
rm paper.preprint.tex
