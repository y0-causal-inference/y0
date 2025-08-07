---
title: Causal identification with $Y_0$
bibliography: paper.bib
repository: y0-causal-inference/y0

tags:
  - causal inference
  - causal identification
  - causal estimation
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
  - name: Craig Bakker
    email: craig.bakker@pnnl.gov
    orcid: 0000-0002-0083-4000
    affiliation: 2
    roles:
      - type: software
        degree: supporting
  - name: Richard J. Callahan
    affiliation: 2
    email: richardcallahan@protonmail.com
    orcid: 0009-0006-6041-5517
    roles:
      - type: software
        degree: supporting
  - name: Joseph Cottam
    email: joseph.cottam@pnnl.gov
    orcid: 0000-0002-3097-5998
    affiliation: 2
    roles:
      - type: software
        degree: supporting
  - name: August George
    orcid: 0000-0001-7876-4359
    affiliation: 2
    email: august.george@pnnl.gov
    roles:
      - type: software
        degree: supporting
  - name: Benjamin M. Gyori
    email: b.gyori@northeastern.edu
    orcid: 0000-0001-9439-5346
    affiliation: 3
    roles:
      - type: software
        degree: supporting
      - type: supervision
        degree: supporting
  - name: Haley M. Hummel
    email: haley.hummel@oregonstate.edu
    orcid: 0009-0004-5405-946X
    affiliation: [2, 4]
    roles:
      - type: software
        degree: supporting
  - name: Nathaniel Merrill
    orcid: 0000-0002-1998-0980
    email: merrill@battelle.org
    affiliation: 5
    roles:
      - type: software
        degree: supporting
  - name: Sara Mohammad Taheri
    email: srtaheri66@gmail.com
    orcid: 0000-0002-6554-9083
    affiliation: 3
    roles:
      - type: software
        degree: supporting
  - name: Pruthvi Prakash Navada
    email: navada.p@northeastern.edu
    orcid: 0009-0008-8505-1670
    affiliation: 3
    roles:
      - type: software
        degree: supporting
  - name: Marc-Antoine Parent
    email: maparent@acm.org
    orcid: 0000-0003-4159-7678
    affiliation: 6
    roles:
      - type: software
        degree: supporting
  - name: Adam Rupe
    email: adam.rupe@pnnl.gov
    affiliation: 2
    orcid: 0000-0003-0105-8987
    roles:
      - type: software
        degree: supporting
  - name: Olga Vitek
    email: o.vitek@northeastern.edu
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
  - name: Battelle Memorial Institute
    index: 5
    ror: 01h5tnr73
  - name: Conversence
    index: 6

date: 21 June 2025
---

# Summary

Researchers are often interested in investigating whether one thing causes
another, such as whether a medication effectively treats a disease or whether
education improves income. Randomized controlled trials provide the most direct
evidence for causal relationships, but they are often logistically impossible,
unethical, or prohibitively expensive to conduct. Causal inference comprises
statistical methods that provide indirect evidence for causal relationships
based on whatever data is available, whether it comes from a (randomized)
controlled trial, an observational study, or a combination of both. However,
both the qualitative and quantitative investigation of causation remains
challenging in the presence of (unknown) confounding variables—a converse to the
old adage that correlation does not imply causation.

A key step in causal inference is **causal identification** during which it is
determined whether it is theoretically possible to estimate a causal effect from
available data, given prior knowledge about relationships between variables and
a causal query, such as:

1. **Interventional Query**, which asks: _what will happen if we intervene?_ For
   example, what would be the average effect if everyone received treatment?
2. **Counterfactual Query**, which asks: _what would have happened had we done
   something different?_ For example, would a given patient, who recovered after
   receiving treatment, have recovered anyway without treatment?.
3. **Transportability Query**, which asks: _can causal findings from one
   population be validly applied to another, and if so, how can evidence from
   multiple studies or populations be combined to draw conclusions about a
   target group of interest?_

We present the $Y_0$ Python package, which implements causal identification
algorithms that apply interventional, counterfactual, and transportability
queries to data from (randomized) controlled trials, observational studies, or
mixtures thereof. $Y_0$ focuses on the qualitative investigation of causation,
helping researchers determine _whether_ a causal relationship can be estimated
from available data before attempting to estimate _how strong_ that relationship
is. Furthermore, $Y_0$ provides guidance on how to transform the causal query
into a symbolic estimand that can be non-parametrically estimated from the
available data. $Y_0$ provides a domain-specific language for representing
causal queries and estimands as symbolic probabilistic expressions, tools for
representing causal graphical models with unobserved confounders, such as
acyclic directed mixed graphs (ADMGs), and implementations of numerous
identification algorithms from the recent causal inference literature.

# Statement of Need

Several open source Python packages have implemented the simplest identification
algorithm (`ID`) from @shpitser2006id including
[Ananke](https://gitlab.com/causal/ananke) [@lee2023ananke],
[pgmpy](https://github.com/pgmpy/pgmpy) [@ankan2015pgmpy],
[DoWhy](https://github.com/py-why/dowhy) [@sharma2020dowhy], and
[causaleffect-py](https://github.com/pedemonte96/causaleffect)
[@pedemonte2021causalefffectpy]. Further, Ananke and DoWhy implement algorithms
that consume the estimand returned by `ID` and observational data in order to
estimate the average causal effect of an intervention on the outcome. However,
these methods are limited in their generalization when causal queries include
multiple outcomes, conditionals, or interventions.

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
source, available for registration of new users, nor provides documentation.

Causal inference remains an active research area where new algorithms are
regularly published (see the recent review from @JSSv099i05), but often without
a reference implementation. We therefore implemented the $Y_0$ Python package in
order to address the need for open source implementations of existing algorithms
as well as to provide a modular framework that can support the implementation of
future algorithms and workflows.

# Implementation

**Probabilistic Expressions** $Y_0$ implements an internal domain-specific
language that can capture variables, counterfactual variables, population
variables, and probabilistic expressions in which they appear. It covers the
three levels of Pearl's Causal Hierarchy [@bareinboim2022], including
association $P(Y=y \mid
X=x^\ast)$, represented as \texttt{P(Y | \textasciitilde
X)}, interventions $P_{do(X=x^\ast)}(Y=y, Z=z)$, represented as
\texttt{P[\textasciitilde X](Y, Z)} and counterfactuals
$P(Y_{do(X=x^\ast)}=y^\ast\mid X=x, Y=y)$, represented as
\texttt{P(\textasciitilde Y @ \textasciitilde X | X, Y)}. Expressions can be
converted to SymPy [@meurer2017sympy] or LaTeX expressions and can be rendered
in Jupyter notebooks.

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
[@tian2010identifying], transport [@correa2020transport], and counterfactual
transport [@correa2022cftransport].

# Case Study

We present a case study regarding the effect of how smoking relates to cancer.
First, we construct a graphical model (\autoref{cancer}A) representing the
following prior knowledge:

1. Smoking causes an accumulation of tar in the lungs.
2. Accumulation of tar in the lungs increases the risk of cancer.
3. Smoking also increases the risk of cancer directly.

![**A**) A simplified acyclic directed graph model representing prior knowledge on smoking and cancer and **B**) a more complex acyclic directed mixed graph that explicitly represents confounding variables.](figures/cancer_tar.pdf){#cancer height="100pt"}

The identification algorithm (`ID`) [@shpitser2006id] estimates the effect of
smoking on the risk of cancer in \autoref{cancer}A as
$\sum_{Tar} P(Cancer | Smoking, Tar) P(Tar | Smoking)$. However, the model in
\autoref{cancer}A is inaccurate because it does not represent confounders
between smoking and tar accumulation, such as the choice to smoke tar-free
cigarettes. Therefore, we add a _bidirected_ edge in \autoref{cancer}B.
Unfortunately, `ID` can not produce an estimand for \autoref{cancer}B, which
motivates the usage of an alternative algorithm that incorporates observational
and/or interventional data. For example, if data from an observational study
associating smoking with tar and cancer ($\pi^{\ast}$) and data from a
randomized trial studying the causal effect of smoking on tar buildup in the
lungs ($\pi_1$) are available, the surrogate outcomes algorithm (`TRSO`)
[@tikka2019trso] estimates the effect of smoking on the risk of cancer in
\autoref{cancer}B as
$\sum_{Tar} P^{\pi^{\ast}}(Cancer |
Smoking, Tar) P_{\text{Smoking}}^{{\pi_1}}(Tar)$.
Code and a more detailed description of this case study can be found in the
following
[Jupyter notebook](https://github.com/y0-causal-inference/y0/blob/main/notebooks/Surrogate%20Outcomes.ipynb).

We provide a second case study demonstrating the transport
[@correa2020transport] and counterfactual transport [@correa2022cftransport]
algorithms for epidemiological studies in COVID-19 in this
[Jupyter notebook](https://github.com/y0-causal-inference/y0/blob/main/notebooks/Counterfactual%20Transportability.ipynb).

$Y_0$ has already been used in several scientific studies which also motivated
its further development:

- @mohammadtaheri2022experimentaldesigncausalquery used $Y_0$ to develop an
  automated experimental design workflow.
- @taheri2023adjustment used $Y_0$ for falsification against experimental and
  simulated data for several biological signaling pathways.
- @taheri2024eliater used $Y_0$ and Ananke to implement an automated causal
  workflow for simple causal queries compatible with `ID`.
- @ness_causal_2024 used $Y_0$ as a teaching tool for identification and the
  causal hierarchy.

# Future Directions

There remain several high value identification algorithms to include in $Y_0$ in
the future. First, the cyclic identification algorithm (`ioID`)
[@forré2019causalcalculuspresencecycles] is important to work with more
realistic graphs that contain cycles, such as how biomolecular signaling
pathways often contain feedback loops. Second, Missing data identification
algorithms can account for data that is missing not at random (MNAR) by modeling
the underlying missingness mechanism [@mohan2021]. Third, algorithms that
provide sufficient conditions for identification in hierarchical causal models
[@weinstein2024hierarchicalcausalmodels] would be useful for supporting causal
identification in probabilistic programming languages, such as ChiRho [@chirho].

Several algorithms noted in the review by @JSSv099i05, such as generalized
identification (`gID`) [@lee2019general] and generalized counterfactual
identification (`gID*`) [@correa2021counterfactual], can be formulated as
special cases of counterfactual transportability. Therefore, we plan to improve
the user experience by exposing more powerful algorithms like counterfactual
transport through a simplified APIs corresponding to special cases like `gID`
and `gID*`. Similarly, we plan to implement probabilistic expression
simplification [@tikka2017b] to improve the consistency of the estimands output
from identification algorithms.

It remains an open research question how to estimate the causal effect for an
arbitrary estimand produced by an algorithm more sophisticated than `ID`.
@agrawal2024automated recently demonstrated automatically generating an
efficient and robust estimator for causal queries more sophisticated than `ID`
using ChiRho [@chirho], a causal extension of the Pyro probabilistic programming
language [@bingham2018pyro]. Probabilistic circuits
[@darwiche2022causalinferenceusingtractable; @wang2023tractable] also present a
new paradigm for tractable causal estimation. Such a generalization would enable
the automation of downstream applications in experimental design.

# Availability and Usage

$Y_0$ is available as a package on [PyPI](https://pypi.org/project/y0) with the
source code available at
[https://github.com/y0-causal-inference/y0](https://github.com/y0-causal-inference/y0)
under a BSD 3-clause license, archived to Zenodo at
[doi:10.5281/zenodo.4432901](https://zenodo.org/doi/10.5281/zenodo.4432901), and
documentation available at
[https://y0.readthedocs.io](https://y0.readthedocs.io). The repository also
contains an interactive Jupyter notebook tutorial and notebooks for the case
studies described above.

# Acknowledgements

The authors would like to thank the German NFDI4Chem Consortium for support.
Additionally, the development of $Y_0$ has been partially supported by the
following grants:

- DARPA award HR00111990009
  ([Automating Scientific Knowledge Extraction](https://www.darpa.mil/program/automating-scientific-knowledge-extraction))
- PNNL Data Model Convergence Initiative award 90001
  ([Causal Inference and Machine Learning Methods for Analysis of Security Constrained Unit Commitment](https://web.archive.org/web/20240518030340/https://www.pnnl.gov/projects/dmc/converged-applications))
- DARPA award HR00112220036
  ([Automating Scientific Knowledge Extraction and Modeling](https://www.darpa.mil/program/automating-scientific-knowledge-extraction-and-modeling))
- Award number DE-SC0023091 under the DOE Biosystems Design program
- The PNNL Causality Community of Interest (Causal COIN), which improved $Y_0$. The Causal COIN was supported by the PNNL Center for AI. The Center for AI @pnnl serves as a virtual research hub that enables the use of artificial intelligence (AI) across research domains and provided resources supporting the research reported in this article. The Center is driving U.S. leadership in AI and technological innovation by advancing research and development, fostering an AI-enabled workforce, and engaging in impactful industry partnerships.

The authorship of this manuscript lists the primary contributors as the first
and last authors and all remaining authors in alphabetical order by family name.

# References
