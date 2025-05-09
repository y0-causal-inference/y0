# $Y_0$ Manuscript

This folder contains source material for a
[Journal of Open Source Software](https://joss.theoj.org/) submission:

1. [paper.md](paper.md): the main manuscript
2. [paper.bib](paper.bib): BibTex citations for the manuscript

## TODO

- mention HCM data structure from `[@weinstein2024hierarchicalcausalmodels]`,
  _only after_ https://github.com/y0-causal-inference/y0/pull/236 is finished
- @Jeremy reference other PNNL use cases (even if they're not published)
- @Jeremy there's a note for you to fill in a sentence in the future work
  paragraph
- Get Richard and Pruthvi's ORCIDs

## Linting

The markdown files here are linted with:

```shell
npx prettier --prose-wrap always --check "**/*.md" --write
```

## Build

Follow the instructions at
https://joss.readthedocs.io/en/latest/submitting.html#docker:

```shell
sh build.sh
```

## Submission

Follow the instructions at https://joss.theoj.org/papers/new.

## Authorship Concept

- All material contributors to the $Y_0$ package are eligible for authorship.
- Lead authorship will be shared between Charles Tapley Hoyt and Jeremy Zucker
- Remaining authors will be listed alphabetically by last name
- JOSS's
  [authorship policy](https://joss.readthedocs.io/en/latest/submitting.html#authorship)
  states:

  > Purely financial (such as being named on an award) and organizational (such
  > as general supervision of a research group) contributions are not considered
  > sufficient for co-authorship of JOSS submissions, but active project
  > direction and other forms of non-code contributions are.

## Extras

Some references that didn't make it into the manuscript:

- Good and bad controls [@cinelli2022crash] (could go in future work)
