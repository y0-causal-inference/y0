---
title: Causal inference with $Y_0$
bibliography: paper.bib
repository: y0-causal-inference/y0

tags:
  - causal inference
  - network science
  - causal artificial intelligence
  - causal machine learning
  - mathematics
  - experimental design
  - interventional trials
  - observational studies

has-equal-contributors: true
authors:
  - name: Charles Tapley Hoyt
    orcid: 0000-0003-4423-4370
    affiliation: 1
    equal-contrib: true
    corresponding: true
    email: cthoyt@gmail.com
    roles:
      - type: software
        degree: equal
      - type: conceptualization
        degree: equal
      - type: writing-original-draft
        degree: lead
  - name: Richard J. Callahan
    affiliation: 2
    orcid: 0009-0006-6041-5517
    roles:
      - type: software
        degree: supporting
  - name: Joseph Cottam
    orcid: 0000-0002-3097-5998
    affiliation: 2
    roles:
      - type: software
        degree: supporting
  - name: Benjamin M. Gyori
    orcid: 0000-0001-9439-5346
    affiliation: 3
    roles:
      - type: software
        degree: supporting
      - type: supervision
        degree: supporting
  - name: Haley Hummel
    orcid: 0009-0004-5405-946X
    affiliation: 4
    roles:
      - type: software
        degree: supporting
  - name: Nathaniel Merrill
    orcid: 0000-0002-1998-0980
    affiliation: 2
    roles:
      - type: software
        degree: supporting
  - name: Sara Mohammad Taheri
    orcid: 0000-0002-6554-9083
    affiliation: 3
    roles:
      - type: software
        degree: supporting
  - name: Pruthvi Prakash Navada
    affiliation: 3
    roles:
      - type: software
        degree: supporting
  - name: Marc-Antoine Parent
    orcid: 0000-0003-4159-7678
    affiliation: 5
    roles:
      - type: software
        degree: supporting
  - name: Adam Rupe
    affiliation: 2
    orcid: 0000-0003-0105-8987
    roles:
      - type: software
        degree: supporting
  - name: Olga Vitek
    affiliation: 3
    orcid: 0000-0003-1728-1104
    roles:
      # See pre-submission inquiry https://github.com/openjournals/joss/issues/1363
      - type: supervision
        degree: supporting
  - name: Jeremy Zucker
    orcid: 0000-0002-7276-9009
    affiliation: 2
    equal-contrib: true
    corresponding: true
    email: jeremy.zucker@pnnl.gov
    roles:
      - type: software
        degree: equal
      - type: conceptualization
        degree: equal
      - type: writing-original-draft
        degree: supporting
      - type: supervision
        degree: lead

affiliations:
  - name: RWTH Aachen University
    index: 1
    ror: 04xfq0f34
  - name: Pacific Northwest National Laboratory
    index: 2
    ror: 05h992307
  - name: Northeastern University
    index: 3
    ror: 04t5xt781
  - name: Oregon State University
    index: 4
    ror: 00ysfqy60
  - name: Conversence
    index: 5

date: 9 May 2025
---

# Summary

Causal inference is the process of determining if and how one variable
influences another. Many algorithms take a graphical model representing causal
dependencies between variables and enable asking counterfactual questions on
observational data. This is useful when acquiring interventional data might be
unethical or otherwise intractable.

The $Y_0$ Python package implements a domain-specific language for representing
probabilistic expressions, a generic data structure for representing graphical
models, several _identification_ algorithms that return an estimand for
different kinds of causal queries (e.g., what is the effect of treatment $X$ on
outcome $Y$?) that serve as the core of causal inference workflows, and an
assortment of related algorithms and workflows useful for doing causal
inference.

# State of the field

Several open source packages in the Python programming language have implemented
the most simple identification algorithm (`ID`) from @shpitser2006id including
[Ananke](https://gitlab.com/causal/ananke) [@lee2023ananke],
[pgmpy](https://github.com/pgmpy/pgmpy) [@ankan2015pgmpy],
[DoWhy](https://github.com/py-why/dowhy) [@sharma2020dowhy], and
[causaleffect-py](https://github.com/pedemonte96/causaleffect)
[@pedemonte2021causalefffectpy]. Further, Ananke and DoWhy implement algorithms
that consume the estimand returned by `ID` and observational data in order to
estimate the average causal effect of an intervention on the outcome. However,
these methods are limited in their generalization to causal queries that include
multiple interventions, multiple outcomes, conditionals, or interventions.

In the R programming language, the
[causaleffect](https://github.com/santikka/causaleffect) package
[@tikka2017causaleffectr] implements `ID`, `IDC` [@shpitser2007idc], surrogate
outcomes (`TRSO`) [@tikka2019trso], and transport [@correa2020transport]. The
[cfid](https://github.com/santikka/cfid) package from the same authors
[@RJ-2023-053] implements `ID*` [@shpitser2012idstar] and `IDC*`
[@shpitser2012idstar]. However, these packages are challenging to use and
extend.

Finally, [CausalFusion](https://www.causalfusion.net) is a web application that
implements many identification and estimation algorithms, but is neither open
source, has open registration, nor provides documentation.

Causal inference remains an active research area where new identification
algorithms are regularly published (see the recent review from @JSSv099i05), but
often without a reference implementation. This motivates the implementation of a
modular framework with reusable data structures and workflows to support the
implementation of both previously published and future algorithms and workflows.

# Implementation

**Probabilistic Expressions** $Y_0$ implements an internal domain-specific
language that can capture variables, counterfactual variables, population
variables, and probabilistic expressions in which they appear. It covers three
levels of Pearl's Causal Hierarchy [@bareinboim2022], including the probability
of sufficient causation $P(Y_X \mid X^*, Y^*)$, necessary causation
$P(Y^*_{X^*} \mid X, Y)$, and necessary and sufficient causation
$P(Y_X, Y^*_{X^*})$. Expressions can be converted to SymPy [@meurer2017sympy],
LaTeX expressions, and be rendered in Jupyter notebooks.

**Data Structure** $Y_0$ builds on NetworkX [@hagberg2008networkx] to implement
an (acyclic) directed mixed graph data structure, used in many identification
algorithms, and the latent variable graph structure described by
@evans2016simplification. It includes a suite of generic graph operations, graph
simplification workflows such as the one proposed by Evans, and conversion
utilities for Ananke, CausalFusion, pgmpy, and causaleffect.

**Falsification** $Y_0$ implements several workflows for checking the
consistency of graphical models against observational data. First, it implements
D-separation [@Pearl_2009], M-separation [@drton2004mseparation],
$\sigma$-separation [@forre2018sigmaseparation] that are applicable to
increasingly more generic mixed graphs. Then, it implements a workflow for
identifying conditional independencies [@Pearl1989] and falsification
[@eulig2023falsifyingcausalgraphsusing]. Finally, it provides a wrapper around
`causaleffect` through [`rpy2`](https://github.com/rpy2/rpy2) for calculating
Verma constraints [@tian2012verma].

**Identification** $Y_0$ has the most complete suite of identification
algorithms of any causal inference package. It implements `ID`
[@shpitser2006id], `IDC` [@shpitser2007idc], `ID*` [@shpitser2012idstar], `IDC*`
[@shpitser2012idstar], surrogate outcomes (`TRSO`) [@tikka2019trso], `tian-ID`
[@tian2010identifying], transport [@correa2020transport], counterfactual
transport [@correa2022cftransport], and identification for causal queries over
hierarchical causal models [@weinstein2024hierarchicalcausalmodels].

# Case Study

We present a case study regarding the effect of how smoking relates to cancer.
First, we construct a graphical model (\autoref{cancer}A) representing the
following prior knowledge:

1. Smoking causes an accumulation of tar in the lungs
2. Accumulation of tar in the lungs increase the risk of cancer
3. Smoking itself also increases the risk of cancer

![**A**) A simplified acyclic directed graph model representing prior knowledge on smoking and cancer and **B**) a more complex acyclic directed mixed graph that explicitly represents confounding variables.](figures/cancer_tar.pdf){#cancer height="100pt"}

The ID algorithm [@shpitser2006id] estimates the effect of smoking on the risk
of cancer in \autoref{cancer}A as
$\sum_{Tar} P(Cancer | Smoking, Tar) P(Tar | Smoking)$. However, the model in
\autoref{cancer}A is inaccurate because it does not represent confounders
between smoking and tar accumulation, such as the choice to smoke tar-free
cigarettes. Therefore, we add a _bidirected_ edge in \autoref{cancer}B.
Unfortunately, the ID algorithm can not produce an estimand for
\autoref{cancer}B, which motivates the usage of an alternative algorithm that
incorporates observational and/or interventional data. For example, if data from
an observational study ($\pi^{\ast}$) and data from an interventional trial on
smoking ($\pi_1$) are available, the TRSO algorithm [@tikka2019trso] estimates
the effect of smoking on the risk of cancer in \autoref{cancer}B as
$\sum_{Tar} P^{\pi^{\ast}}(Cancer | Smoking, Tar) P_{\text{Smoking}}^{{\pi_1}}(Tar)$.
Code and a more detailed description of this case study can be found in the
following
[Jupyter notebook](https://github.com/y0-causal-inference/y0/blob/main/notebooks/Surrogate%20Outcomes.ipynb).

We provide a second case study demonstrating the transport
[@correa2020transport] and counterfactual transport [@correa2022cftransport]
algorithms for epidemiological studies in COVID-19 in this
[Jupyter notebook](https://github.com/y0-causal-inference/y0/blob/main/notebooks/Counterfactual%20Transportability.ipynb).

We highlight several which used (and motivated further development of) $Y_0$:

- @mohammadtaheri2022experimentaldesigncausalquery used $Y_0$ to develop an
  automated experimental design workflow.
- @taheri2023adjustment used $Y_0$ for falsification against experimental and
  simulated data for several biological signaling pathways.
- @taheri2024eliater used $Y_0$ and Ananke to implement an automated causal
  workflow for simple causal queries compatible with `ID`.
- @ness_causal_2024 used $Y_0$ as a teaching tool for identification and the
  causal hierarchy

# Future direction

There remain several high value identification algorithms to include in $Y_0$ in
the future. For example, the cyclic ID (`ioID`)
[@forré2019causalcalculuspresencecycles] is important to work with more
realistic graphs that contain cycles, such as how biomolecular signaling
pathways often contain feedback loops. Further, missing data identification
algorithms can handle when data is missing not at random (MNAR) by modeling the
underlying missingness mechanism [@mohan2021]. Several algorithms noted in the
review by @JSSv099i05, such as generalized ID (`gID`) [@lee2019general] and
generalized counterfactual ID (`gID*`) [@correa2021counterfactual], can be
formulated as special cases of counterfactual transportability. Therefore, we
plan to improve the user experience by exposing more powerful algorithms like
counterfactual transport through a simplified APIs corresponding to special
cases like `gID` and `gID*`. Similarly, we plan to implement probabilistic
expression simplification [@tikka2017b] to improve the consistency of the
estimands output from identification algorithms.

It remains an open research question on how to estimate the causal effect for an
arbitrary estimand produced by an algorithm more sophisticated than `ID`. Two
potential avenues for overcoming this might be a combination of the Pyro
probabilistic programming langauge [@bingham2018pyro] and its causal inference
extension [ChiRho](https://github.com/BasisResearch/chirho). Tractable circuits
[@darwiche2022causalinferenceusingtractable] also present a new paradigm for
generic estimation. Such a generalization would be a lofty achievement and
enable the automation of downstream applications in experimental design.

# Availability and usage

`y0` is available as a package on [PyPI](https://pypi.org/project/y0) with the
source code available at
[https://github.com/y0-causal-inference/y0](https://github.com/y0-causal-inference/y0),
archived to Zenodo at
[doi:10.5281/zenodo.4432901](https://zenodo.org/doi/10.5281/zenodo.4432901), and
documentation available at
[https://y0.readthedocs.io](https://y0.readthedocs.io). The repository also
contains an interactive Jupyter notebook tutorial and notebooks for the case
studies described above.

# Acknowledgements

The development of $Y_0$ has been partially supported by the following grants:

- DARPA award HR00111990009
  ([Automating Scientific Knowledge Extraction](https://www.darpa.mil/program/automating-scientific-knowledge-extraction))
- PNNL Data Model Convergence Initiative award 90001
  ([Causal Inference and Machine Learning Methods for Analysis of Security Constrained Unit Commitment](https://web.archive.org/web/20240518030340/https://www.pnnl.gov/projects/dmc/converged-applications))
- DARPA award HR00112220036
  ([Automating Scientific Knowledge Extraction and Modeling](https://www.darpa.mil/program/automating-scientific-knowledge-extraction-and-modeling))

The authorship of this manuscript lists the primary contributors as the first
and last authors and all remaining authors in alphabetical order by family name.

# References
