"""Separation algorithms.

========================= ============================================================= ====================================================================
Algorithm                 Description                                                   Implementation
========================= ============================================================= ====================================================================
d-separation [pearl2009]_ Identifies nodes in an acyclic directed mixed graph as        :func:`y0.algorithm.separation.are_d_separated`
                          probabilistically independent
m-separation [drton2003]_ A generalization of d-separation in an acyclic directed mixed `Issue #204 <https://github.com/y0-causal-inference/y0/issues/204>`_
                          graph
σ-separation [forre2018]_ Identifies nodes in any directed mixed graph as               :func:`y0.algorithm.separation.are_sigma_separated`
                          probabilistically dependent (cycles allowed)
σ-single-door [forre2018] Graphical admissibility test and adjustment-set finder for    :func:`y0.algorithm.separation.is_sigma_single_door_admissible`,
                          the σ-separation single-door criterion (cycles allowed)       :func:`y0.algorithm.separation.find_sigma_single_door_sets`,
                                                                                        :func:`y0.algorithm.separation.find_sigma_single_door_set`,
                                                                                        :func:`y0.algorithm.separation.optimal_adjustment_set`,
                                                                                        :func:`y0.algorithm.separation.proper_backdoor_graph`
σ-extension [rantanen2020] σ-extension graph for efficient cyclic adjustment-set        :func:`y0.algorithm.separation.sigma_extension`
                           computation (reduces σ-sep to d-sep)
========================= ============================================================= ====================================================================

.. [forre2018] `Constraint-based Causal Discovery for Non-Linear Structural Causal Models
    with Cycles and Latent Confounders <https://arxiv.org/abs/1807.03024>`_

.. [drton2003] `Iterative Conditional Fitting for Gaussian Ancestral Graph Models
    <https://stat.uw.edu/sites/default/files/files/reports/2003/tr437.pdf>`_

.. [pearl2009] `Causality: Models, Reasoning and Inference: Models, Reasoning and
    Inference <https://doi.org/10.1017/CBO9780511803161>`_
"""

from .sigma_extension import sigma_extension
from .sigma_separation import are_sigma_separated
from .sigma_single_door import (
    find_sigma_single_door_set,
    find_sigma_single_door_sets,
    is_sigma_single_door_admissible,
    optimal_adjustment_set,
    proper_backdoor_graph,
)
from ..conditional_independencies import are_d_separated

__all__ = [
    "are_d_separated",
    "are_sigma_separated",
    "find_sigma_single_door_set",
    "find_sigma_single_door_sets",
    "is_sigma_single_door_admissible",
    "optimal_adjustment_set",
    "proper_backdoor_graph",
    "sigma_extension",
]
