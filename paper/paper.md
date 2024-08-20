---
title: Causal inference with $Y_0$

tags:
  - causal inference
  - network science
  - causal artificial intelligence
  - mathematics

authors:
  - name: Charles Tapley Hoyt
    orcid: 0000-0003-4423-4370
    affiliation: 1
    equal-contrib: true
  - name: Jeremy Zucker
    orcid: 0000-0002-7276-9009
    affiliation: 2
    equal-contrib: true

# TODO contact remaining contributors to ask if they want to be
# co-authors when paper is almost done

affiliations:
  - name: Independent Researcher
    index: 1
  - name: Pacific Northwest National Laboratory
    index: 2

date: 19 August 2024
bibliography: paper.bib
repository: y0-causal-inference/y0
---

# Summary

Causal inference is the process of determining whether and how one variable influences another.
Given a graphical model representing causal dependencies between variables, causal inference
enables asking interventional and counterfactual questions on observational data, even in scenarios where acquiring
interventional data might be either unethical or intractable.
Causal inference workflows can be divided in two parts: 1) identifying an estimand for the causal query and 2) estimating the estimand using interventional and/or observational data.

> For example, <insert description of epidemiology use cases for causal inference>

We implemented $Y_0$, a Python software package that contains data structures and algorithms that can be used to ask
such questions.
While new causal inference algorithms are regularly published, they often lack reusable reference implementations.
Therefore, we implemented $Y_0$ in a modular and extensible way to support the implementation of new algorithms.

# State of the field

New causal inference algorithms are regularly published, but they often lack reusable reference implementations.
[@JSSv099i05:] provide a good overview of several key identification algorithms, but only some of them have corresponding first- or third-party implementations.

[Ananke](https://gitlab.com/causal/ananke) [@lee2023ananke], [pgmpy](https://github.com/pgmpy/pgmpy) [@ankan2015pgmpy], and [DoWhy](https://github.com/py-why/dowhy) [@sharma2020dowhy], and [causaleffect-py](https://github.com/pedemonte96/causaleffect) [@pedemonte2021causalefffectpy] are Python packges that implement the `ID` algorithm [@shpitser2006id] - the most simple identification.
Further, Ananke and DoWhy implement several estimators that work well with estimands from the `ID` algorithm.

[causaleffect](https://github.com/santikka/causaleffect) [@tikka2017causaleffectr] implements `ID`, `IDC` [@shpitser2007idc], surrogate outcomes (`TRSO`) [@tikka2019trso],
transport [@correa2020transport], and several other key algorithms and subroutines.
Similarly, [cfid](https://github.com/santikka/cfid) package [@RJ-2023-053] implements `ID*` [@shpitser2012idstar] and `IDC*` [@shpitser2012idstar].
However, these two packages are difficult to understand.

[CausalFusion](https://www.causalfusion.net) is an interactive web application that provides the implementation of many identification and estimation algorithms, but it requires approved registration, is closed source, and doesn't have public-facing documentation.

Most work in causal inference is focused on _acyclic_ graphs, which is often not an appropriate constraint for real-world causal models such as biological networks.
Algorithms such as cyclic identification (`ioID`) [@forré2019causalcalculuspresencecycles].
Further, estimation of estimands produced by algorithms more sophisticated than `ID` is an open problem.

# Implementation

> A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.

We implemented an internal domain-specific language (DSL) that represents variables and probability expressions

and can fully capture all levels of Pearl's Causal Hierarchy [@bareinboim2022].
Notably, it can express the probabilities of:

- sufficient causation $P(Y_X \mid X^*, Y^*)$
- necessary causation $P(Y^*_{X^*} \mid X, Y)$
- necessary and sufficient causation $P(Y_X, Y^*_{X^*})$

[@meurer2017sympy]

These can include conditionals, counterfactual variables, and other domain-specific glyphs that
are not supported out of the box.

- DSL
    - Sympy isn't sufficient for operations
    - Represent L1, L2, and L3 expressions from
    - Probability of necessary and sufficient causation
    - Simplify algorithm TBD [@tikka2017causaleffectr]
    - Novel complexity measurement
- Graph Data Structures
    - Mixed Directed Graph (acyclic for many algorithms)
    - Latent Variable Graph [@evans2016simplification]
- Graph algorithms and methods
    - Verma constraints [@tian2012verma] (not fully implemented), conditional independencies [@Pearl1989], and
      falsification [@eulig2023falsifyingcausalgraphsusing]
    - Separation (D-Separation [@Pearl_2009], M-Separation [@drton2004mseparation],
      Sigma-Separation [@forre2018sigmaseparation])
    - Graph simplification via LV-dag with Evans' rules [@evans2016simplification]
    - I/O with Ananke, Causal Fusion, NetworkX, Pgmpy [@ankan2015pgmpy], causaleffect
- Identification
    - ID [@shpitser2006id]
    - IDC [@shpitser2007idc]
    - ID* [@shpitser2012idstar]
    - IDC\* [@shpitser2012idstar]
    - Surrogate Outcomes (TRSO) [@tikka2019trso]
    - Tian ID [@tian2010identifying]
    - Transport [@correa2020transport]
    - Counterfactual Transport [@correa2022cftransport]
- Estimation
    - Wrapper around ananke for automation
    - Implementation of linear SCM (skip this, simpler to leave as later work.)

# Case Study

We present a case study regarding the effect of how smoking relates to cancer.
First, we construct a graphical model (\autoref{cancer}A) representing the following prior knowledge:

1. Smoking causes an accumulation of tar in the lungs
2. Accumulation of tar in the lungs increase the risk of cancer
3. Smoking itself also increases the risk of cancer

![**A**) A simplified acyclic directed graph model representing prior knowledge on smoking an cancer and **B
**) a more complex acyclic directed mixed graph that explicitly represents confounding variables.](figures/cancer_tar.pdf){#cancer height="100pt"}

The ID algorithm [@shpitser2006id] estimates the effect of smoking on the risk of cancer in \autoref{cancer}A as
$\sum_{Tar} P(Cancer | Smoking, Tar) P(Tar | Smoking)$.
The model in \autoref{cancer}A is inaccurate because it does not represent confounders between smoking and tar
accumulation, such as the choice to smoke tar-free cigarettes.
Therefore, we add a _bidirected_ edge in \autoref{cancer}B.
Unfortunately, the ID algorithm can not make an estimant for \autoref{cancer}B, which motivates the usage of an
alternative algorithm that incorporates observational and/or interventional data.
For example, if data from an observational trial ($\pi^{\ast}$) and data from an interventional trial on smoking (
$\pi_1$) are available, the TRSO algorithm [@tikka2019trso] estimates the effect of smoking on the risk of cancer in
\autoref{cancer}B as
$\sum_{Tar} P^{\pi^{\ast}}(Cancer | Smoking, Tar) P_{\text{Smoking}}^{{\pi_1}}(Tar)$.
Code and a more detailed description of this case study can be found in the
following [Jupyter notebook](https://github.com/y0-causal-inference/y0/blob/main/notebooks/Surrogate%20Outcomes.ipynb).

We provide a second case study demonstrating the transport [@correa2020transport] and counterfactual
transport [@correa2022cftransport] algorithms for epidemiological studies in COVID-19 in
this [Jupyter notebook](https://github.com/y0-causal-inference/y0/blob/main/notebooks/Counterfactual%20Transportability.ipynb).

# Use Cases

> Mention (if applicable) a representative set of past or ongoing research projects using the software and recent
> scholarly publications enabled by it.

We also refer to previously published uses of $Y_0$:

- [@taheri2022partial]
- [@mohammadtaheri2022experimentaldesigncausalquery]
- [@taheri2023adjustment]
- [@taheri2024eliater]
- Reference other PNNL use cases (even if they're not published)
- Altdeep.ai (https://github.com/altdeep/causalML) course
- what was the other book robert was publishing?

# Future direction

1. What are some algorithmic limitations that need more research? E.g., estimation of multiple outcomes
2. What are some parts of a fully automatable causal workflow we haven't been able to do?
    - We can do identification and generation of an estimand, but only estimate the estimand in a small number of cases.
      What are some forward-looking ways to overcome this?
        - Pyro [@bingham2018pyro] and ChiRho [https://github.com/BasisResearch/chirho]
        - Tractable Circuits [@darwiche2022causalinferenceusingtractable]
3. What are some high-priority algorithms to implement on our to-do list, and why are they important?
    - cyclic ID (ioID) [@forré2019causalcalculuspresencecycles]
    - gID [@lee2019general]
    - gID* [@correa2021counterfactual]
    - Data ID [@nabi2020dataid]
    - Good and bad controls [@cinelli2022crash]

# Availability and usage

`y0` is available as a package on [PyPI](https://pypi.org/project/y0) with the source code available
at [https://github.com/y0-causal-inference/y0](https://github.com/y0-causal-inference/y0) archived to Zenodo
at [doi:10.5281/zenodo.4432901](https://zenodo.org/doi/10.5281/zenodo.4432901) and documentation available
at [https://y0.readthedocs.io](https://y0.readthedocs.io).
The repository also contains an interactive Jupyter notebook tutorial and notebooks for the case studies described
above.

# Support

The development of $Y_0$ has been partially supported by the following grants:

- DARPA award
  HR00111990009 ([Automating Scientific Knowledge Extraction](https://www.darpa.mil/program/automating-scientific-knowledge-extraction))
- PNNL Data Model Convergence Initiative award
  90001 ([Causal Inference and Machine Learning Methods for Analysis of Security Constrained Unit Commitment](https://web.archive.org/web/20240518030340/https://www.pnnl.gov/projects/dmc/converged-applications))
- DARPA award
  HR00112220036 ([Automating Scientific Knowledge Extraction and Modeling](https://www.darpa.mil/program/automating-scientific-knowledge-extraction-and-modeling))

# References
