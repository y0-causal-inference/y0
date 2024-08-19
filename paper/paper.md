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

# Statement of need

> Clearly illustrate the research purpose of the software and places it in the context of related work.

1. What's the point of causal inference? Referencing some Pearl and Barenboim papers.
2. Tasks you can accomplish:
   - estimate causal effect
   - ask "what if" questions related to interventions and counterfactuals
3. What are some of the high-level problems that it solves? Give general feel as well as a few specific examples to
   motivate the case studies/demonstration later.

Issues we need to solve:

1. Reusable implementations of algorithms
2. Components of algorithms that can serve as building blocks for future algorithms
3. Implementation of DSL for probabilistic expressions and mixed graph data structure

# State of the field

> A list of key references, including to other software addressing related needs. Note that the references should
> include full names of venues, e.g., journals and conferences, not abbreviations only understood in the context of a
> specific discipline.

Existing software:

1. Ananke [@lee2023ananke]
2. NetworkX [@hagberg2008networkx]
3. Causal Fusion (https://www.causalfusion.net; closed source, requires approved registration)
4. CausalEffectR [@tikka2017causaleffectr] and causaleffect-py [@pedemonte2021causalefffectpy]
5. DoWhy [@sharma2020dowhy] and [@blobaum2024dowhy]

# Summary

> A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.



- DSL
    - Sympy [@meurer2017sympy] isn't sufficient for operations 
    - Represent L1, L2, and L3 expressions from [@bareinboim2022]
    - Probability of necessary and sufficient causation
    - Simplify algorithm TBD [@tikka2017causaleffectr]
    - Novel complexity measurement
- Graph Data Structures
    - Mixed Directed Graph (acyclic for many algorithms)
    - Latent Variable Graph [@evans2016simplification]
- Graph algorithms and methods
    - Verma constraints [@tian2012verma] (not fully implemented), conditional independencies [@Pearl1989], and falsification [@eulig2023falsifyingcausalgraphsusing]
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

# Case studies / Use Cases

- TODO ID example
- Surrogate outcomes for smoking and cancer [https://github.com/y0-causal-inference/y0/blob/main/notebooks/Surrogate%20Outcomes.ipynb]
- Counterfactual transportability in COVID-19 [https://github.com/y0-causal-inference/y0/blob/main/notebooks/Counterfactual%20Transportability.ipynb]
- Reference case studies from Sara's papers
  - [@zucker2021viralpathogenesis]
  - [@taheri2022partial]
  - [@mohammadtaheri2022experimentaldesigncausalquery]
  - [@taheri2023adjustment]
  - [@taheri2024eliater]
- Reference other PNNL use cases (even if they're not published)

# Ongoing research projects using it

> Mention (if applicable) a representative set of past or ongoing research projects using the software and recent
> scholarly publications enabled by it.

- Altdeep.ai (https://github.com/altdeep/causalML) course

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
