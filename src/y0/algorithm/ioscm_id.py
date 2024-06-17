# -*- coding: utf-8 -*-

"""Implementation of the ID algorithm for input-output structural causal models (ioSCMs).

.. [forré20a] http://proceedings.mlr.press/v115/forre20a/forre20a.pdf.
.. [forré20b] http://proceedings.mlr.press/v115/forre20a/forre20a-supp.pdf
"""

from y0.dsl import Variable  # P,; R,; W,; X,; Y,; Z,
from y0.graph import NxMixedGraph


def get_strongly_connected_component(graph: NxMixedGraph, v: Variable) -> set[Variable]:
    r"""Return the strongly-connected component within which a graph vertex lies.

    The implementation is as simple as creating a dictionary. Adding variables to
    the dictionary removes repeated variables in the input event, and adding values to
    the dictionary using the variables as keys removes repeated values.

    :math: The strongly connected component of $v$ in $G$ is defined to be:
    $\text{Sc}^{G}(v):= \text{Anc}^{G}(v)\cap \text{Desc}^{G}(v)$.

    :param graph:
        The corresponding graph.
    :param v:
        The vertex for which the strongly connected component is to be retrieved.
    :returns:
        The set of variables comprising the strongly connected component $\text{Sc}^{G}(v)$.
    """
    raise NotImplementedError
