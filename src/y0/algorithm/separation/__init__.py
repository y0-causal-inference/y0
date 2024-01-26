"""Separation algorithms.

=========================  ============================================================================================  ====================================================================
Algorithm                  Description                                                                                   Implementation
=========================  ============================================================================================  ====================================================================
d-separation [pearl2009]_  Identifies nodes in an acyclic directed mixed graph as probabilistically independent          :func:`y0.algorithm.separation.are_d_separated`
m-separation [drton2003]_  A generalization of d-separation in an acyclic directed mixed graph                           `Issue #204 <https://github.com/y0-causal-inference/y0/issues/204>`_
σ-separation [forre2018]_  Identifies nodes in any directed mixed graph as probabilistically dependent (cycles allowed)  :func:`y0.algorithm.separation.are_sigma_separated`
=========================  ============================================================================================  ====================================================================

.. [forre2018] `Constraint-based Causal Discovery for Non-Linear Structural Causal Models with Cycles and
      Latent Confounders <https://arxiv.org/abs/1807.03024>`_
.. [drton2003] `Iterative Conditional Fitting for Gaussian Ancestral Graph Models
      <https://stat.uw.edu/sites/default/files/files/reports/2003/tr437.pdf>`_
.. [pearl2009] `Causality: Models, Reasoning and Inference: Models, Reasoning and Inference
      <https://doi.org/10.1017/CBO9780511803161>`_
"""  # noqa:E501

from .sigma_separation import are_sigma_separated
from ..conditional_independencies import are_d_separated

__all__ = [
    "are_d_separated",
    "are_sigma_separated",
]
