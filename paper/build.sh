# old way
# see https://joss.readthedocs.io/en/latest/submitting.html#docker
# docker run --rm \
#  --volume $PWD:/data \
#  --user $(id -u):$(id -g) \
#  --env JOURNAL=joss \
#  openjournals/paperdraft

# new way, not documented by JOSS yet
# see https://github.com/openjournals/inara/tree/main
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
