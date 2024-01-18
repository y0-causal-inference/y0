"""An implementation of Tian and Pearl's identification algorithm from [tian03a]_."""

import logging

from y0.dsl import Expression, Fraction, One, P, Probability, Product, Sum, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "tian_pearl_identify",
]

logger = logging.getLogger(__name__)


def tian_pearl_identify(
    *,
    input_variables: set[Variable],
    input_district: set[Variable],
    q_expression: Expression,
    graph: NxMixedGraph,
    topo: list[Variable],
) -> Expression | None:
    """Implement the IDENTIFY algorithm as presented in [tian03a]_ with pseudocode in [correa22a]_ (Algorithm 5).

    Santikka has implemented this algorithm in the R package Causal Effect ([santikka20b]_). We draw from that
    implementation. Their version also keeps track of the structure of calls
    :param input_variables: The set of variables, C, for which we're checking if causal identification is possible.
    :param input_district: The C-component, T, containing C.
    :param q_expression: The expression Q[T] as per [tian2003]_, Equation 35.
    :param graph: The relevant graph.
    :param topo: A list of variables in topological order that includes all variables in the graph and may contain more.
    :raises TypeError: at least one input variable is not in the input district.
    :returns: An expression for Q[C] in terms of Q, or Fail.
    """
    # TODO: Verify that the input_variables are all in the input_district
    if not all([v in input_district for v in input_variables]):
        raise TypeError(
            "In tian_pearl_identify: at least one of the input variables C is not in the input district T."
        )
    # TODO: Verify that the input_district is a c-component of the graph.
    # TODO: Santikka's version ([santikka20b]_)
    raise NotImplementedError("Unimplemented function: identify")


def _do_tian_pearl_identify_line_1(
    input_variables: set[Variable],
    input_district: set[Variable],
    graph: NxMixedGraph,
) -> set[Variable] | None:
    """Implement line 1 of the IDENTIFY algorithm in [tian03a]_ and [correa22a]_ (Algorithm 5)."""
    raise NotImplementedError("Unimplemented function: ctfTRu")


def _tian_lemma_1_i(
    *,
    district: set[Variable],
    graph_probability: Probability,
    topo: list[Variable],
) -> Expression | None:
    """Compute the Q value associated with the C-component (district) in a graph as per [tian03a]_, Equation 37.

    This algorithm uses part (i) of Lemma 1 of Tian03a.

    :param district: A list of variables comprising the district for which we're computing a C factor.
    :param graph_probability: the Q value for the full graph.

    :param topo: a topological sort of the vertexes in the graph.
    :raises TypeError: the district or variable set from which it is drawn contained no variables.
    :raises KeyError: a variable in the district is not in the topological sort of the graph vertexes.
    :returns: An expression for Q[district].
    """
    # (Topological sort is O(V+E): https://stackoverflow.com/questions/31010922/)

    variables = set(topo)
    logger.warning("In _tian_lemma_1_i: topo = " + str(topo))
    result = None
    if len(district) == 0 or len(variables) == 0:
        raise TypeError(
            "Error in _tian_lemma_1_i: the district or variable set from which it is drawn contained no variables."
        )
    if any(v not in variables for v in district):
        raise KeyError(
            "Error in _tian_lemma_1_i: a variable in the district is not in the topological sort of the graph vertexes."
        )
    for variable in district:
        preceding_variables = topo[: topo.index(variable)]
        conditioned_variables = list(
            set(graph_probability.parents).union(
                {variable for variable in preceding_variables if variable in variables}
            )
        )  # V^(i-1)
        tmp = P(variable | conditioned_variables)  # v_i
        if result is None:
            result = tmp
        else:
            result = Product.safe([result, tmp])
    return result


def _tian_equation_72(
    *,
    vertex: Variable | None,
    graph_probability: Expression,  # Q[H]
    graph: NxMixedGraph,
) -> Expression:
    r"""Compute the probability of a set of variables according to [tian03a]_, Equation 72.

    This algorithm uses part (ii) of Lemma 4 of [tian03a]_. The context for Equations 71 and 72 follow:

    :math: Let $H \subseteq V$, and assume that $H$ is partitioned into c-components $H_{1}, \dots, V_{h_{l}}$
        in the subgraph $G_{H}$. Then we have...

        (ii) Let $k$ be the number of variables in $H$, and let a topological order of
        the variables in $H$ be $V_{h_{1}} < \cdots < V_{h_{k}}$ be the set of
        variables in $G_{H}$. Let $H^{(i)} = {V_{h_{1}},\dots,V_{h_{i}}$ be the set of variables in $H$ ordered
        before $V_{h_{i}}$ (including $V_{h_{i}}$), $i=1,\dots,k$, and $H^{(0)} = \emptyset$. Then each
        $Q[H_{j}]$, $j = 1,\dots,l$, is computable from $Q[H]$ and is given by

        \begin{equation}
        $Q[H_{j} = \prod_{\{i|V_{h_{i}}\in H_{j}\}}{\frac{Q[H^{(i)}]}{Q[H^{(i-1)}]},$
        \end{equation}

        where each $Q[H^{(i)}], i = 0, 1, \dots\, k$, is given by

        \begin{equation}
        $Q[H^{(i)}] = \sum_{h \backslash h^{(i)}}{Q[H]}$.
        \end{equation}

    (The second equation above is Equation 72.)

    :param vertex: The $i^{th}$ variable in topological order
    :param graph_probability: The probability of $H$ corresponding to $Q[H]$ in Equation 72.
    :param graph: The subgraph under analysis, $G_{H}$.
    :raises KeyError: the input vertex is not in the variable set or not in the topological ordering of graph vertices.
    :returns: An expression for $Q[H^{(i)}]$.
    """
    # $Q[H^{(0)}] = Q[\emptyset] = 1$
    if vertex is None:
        return One()
    # We need to compute a topological order for the subgraph H each time we call this function.
    variables = {node for node in graph.nodes()}
    topo = [variable for variable in graph.topological_sort()]
    logger.warning("In _tian_equation_72: input vertex is " + str(vertex))
    logger.warning("   and variables are " + str(variables))
    logger.warning("   and topo is " + str(topo))
    if vertex not in variables:
        raise KeyError("In _tian_equation_72: input vertex is not in the input graph.")
    return Sum.safe(
        graph_probability, [v for v in topo[topo.index(vertex) + 1 :] if v in variables]
    )


def _tian_lemma_4_ii(
    *,
    district: set[Variable],
    graph_probability: Expression,
    graph: NxMixedGraph,
) -> Expression | None:
    r"""Compute the Q value associated with the C-component (district) in a graph as per [tian03a]_, Equations 71 and 72.

    This algorithm uses part (ii) of Lemma 4 of [tian03a]_. The context for Equations 71 and 72 follow:

    :math: Let $H \subseteq V$, and assume that $H$ is partitioned into c-components $H_{1}, \dots, V_{h_{l}}$
        in the subgraph $G_{H}$. Then we have...

        (ii) Let $k$ be the number of variables in $H$, and let a topological order of
        the variables in $H$ be $V_{h_{1}} < \cdots < V_{h_{k}}$ be the set of
        variables in $G_{H}$. Let $H^{(i)} = {V_{h_{1}},\dots,V_{h_{i}}$ be the set of variables in $H$ ordered
        before $V_{h_{i}}$ (including $V_{h_{i}}$), $i=1,\dots,k$, and $H^{(0)} = \emptyset$. Then each
        $Q[H_{j}]$, $j = 1,\dots,l$, is computable from $Q[H]$ and is given by

        \begin{equation}
        $Q[H_{j} = \prod_{\{i|V_{h_{i}}\in H_{j}\}}{\frac{Q[H^{(i)}]}{Q[H^{(i-1)}]},$
        \end{equation}

        where each $Q[H^{(i)}], i = 0, 1, \dots\, k$, is given by

        \begin{equation}
        $Q[H^{(i)}] = \sum_{h \backslash h^{(i)}}{Q[H]}$.
        \end{equation}

    (The second equation above is Equation 72.)

    :param district: A list of variables comprising the district for which we're computing a C factor.
    :param graph_probability: The expression Q corresponding to the set of variables in v. It is
        Q[A] on the line calling Lemma 4 in [tian2003]_, Figure 7.
    :param graph: The subgraph $G_{H}$ in question.
    :returns: An expression for Q[district].
    """
    # subgraph = graph.subgraph(district)
    topo = [variable for variable in graph.topological_sort()]

    def _one_round(index: int) -> Expression:
        if index == 0:
            logger.warning("In _one_round: index = 0.")
            return_value = _tian_equation_72(
                vertex=topo[index],
                graph_probability=graph_probability,
                graph=graph,
            )
            logger.warning("Returning: " + str(return_value))
            return _tian_equation_72(
                vertex=topo[index],
                graph_probability=graph_probability,
                graph=graph,
            )
        else:
            return_num = _tian_equation_72(
                vertex=topo[index],
                graph_probability=graph_probability,
                graph=graph,
            )
            return_den = _tian_equation_72(
                vertex=topo[index - 1],
                graph_probability=graph_probability,
                graph=graph,
            )
            logger.warning("In _one_round: topo = " + str(topo))
            logger.warning("In one_round with index > 1: return_num = " + str(return_num))
            logger.warning("In one_round with index > 1: return_den = " + str(return_den))
            logger.warning(
                "In one_round with index > 1: returning = " + str(Fraction(return_num, return_den))
            )
            return Fraction(
                _tian_equation_72(
                    vertex=topo[index],
                    graph_probability=graph_probability,
                    graph=graph,
                ),
                _tian_equation_72(
                    vertex=topo[index - 1],
                    graph_probability=graph_probability,
                    graph=graph,
                ),
            )

    product = None
    for vertex in district:
        logger.warning("In Lemma 4(ii): vertex = " + str(vertex))
        index = topo.index(vertex)
        if product is None:
            product = _one_round(index)
            logger.warning("Result of first round is " + str(product))
        else:
            tmp = _one_round(index)
            logger.warning("Result of next round is " + str(tmp))
            product = Product.safe([product, tmp])
        logger.warning("\n")
        logger.warning("Index = " + str(index))
        logger.warning("Product = " + str(product))
    logger.warning("Returning product: " + str(product))
    return product
