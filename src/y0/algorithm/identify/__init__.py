# -*- coding: utf-8 -*-

"""Identification algorithms.

Identification
--------------
======================  ====================================================================
Algorithm               Implementation
======================  ====================================================================
ID [shpitser2006]_      :mod:`y0.algorithm.identifiy.id_std`
IDC [shpitser2008]_     :mod:`y0.algorithm.identifiy.idc`
ID* [shpitser2012]_     :mod:`y0.algorithm.identifiy.id_star`
IDC* [shpitser2012]_    :mod:`y0.algorithm.identifiy.idc_star`
Cyclic ID [forre2019]_  `Issue #71 <https://github.com/y0-causal-inference/y0/issues/71>`_
gID [correa2019]_       `Issue #72 <https://github.com/y0-causal-inference/y0/issues/72>`_
gID* [correa2021]_      `Issue #121 <https://github.com/y0-causal-inference/y0/issues/121>`_
Data ID [nabi2020]_     `Issue #73 <https://github.com/y0-causal-inference/y0/issues/73>`_
======================  ====================================================================

This table partially draws from [tikka2021]_, which made a nice review of non-counterfactual
identification methods.

Transport
---------
=============================================  ======================================
Algorithm                                      Implementation
=============================================  ======================================
Surrogate Outcomes [tikka2018]_                :mod:`y0.algorithm.transport`
Transportability [correa2020]_                 :mod:`y0.algorithm.transport`
Counterfactual Transportability [correa2022]_  `PR #197 <https://github.com/y0-causal-inference/y0/pull/197>`_
=============================================  ======================================

.. [shpitser2006] `Identification of joint interventional distributions in recursive semi-Markovian
       causal models <https://dl.acm.org/doi/10.5555/1597348.1597382>`_
.. [shpitser2008] `Complete Identification Methods for the Causal
       Hierarchy <https://www.jmlr.org/papers/v9/shpitser08a.html>`_
.. [shpitser2012] https://dl.acm.org/doi/10.5555/1597348.1597382
.. [tikka2018] https://arxiv.org/abs/1806.07172
.. [correa2020] `General Transportability: Synthesis of Experiments from Heterogeneous
    Domains <https://causalai.net/r53.pdf>`_
.. [correa2022] `Counterfactual Transportability: A Formal Approach <https://causalai.net/r82.pdf>`_
.. [forre2019] `Causal Calculus in the Presence of Cycles, Latent Confounders and Selection
    Bias <https://arxiv.org/abs/1901.00433v2>`_
.. [nabi2020] `Full Law Identification In Graphical Models Of Missing Data: Completeness Results
    <https://arxiv.org/abs/2004.04872>`_
.. [correa2019] `General Identifiability with Arbitrary Surrogate Experiments <https://causalai.net/r46.pdf>`_
.. [correa2021] `Nested Counterfactual Identification from Arbitrary Surrogate Experiments
    <https://causalai.net/r79.pdf>`_
.. [tikka2021] `Causal Effect Identification from Multiple Incomplete Data Sources:
    A General Search-based Approach <https://arxiv.org/pdf/1902.01073.pdf>`_

"""

from .api import identify_outcomes
from .id_c import idc
from .id_star import id_star
from .id_std import identify
from .idc_star import idc_star
from .utils import Identification, Query, Unidentifiable

__all__ = [
    # Algorithms
    "identify_outcomes",
    "identify",
    "id_star",
    "idc",
    "idc_star",
    # Data Structures
    "Query",
    # Exceptions
    "Unidentifiable",
    "Identification",
]
