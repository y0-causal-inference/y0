"""
An implementation of Tian and Pearl's identification algorithm from [tian03a]_.
"""

from y0.dsl import Expression, One, P, Sum, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "tian_pearl_identify",
]


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
    variables: set[Variable],
    topo: list[Variable],
) -> Expression | None:
    """Compute the Q value associated with the C-component (district) in a graph as per [tian03a]_, Equation 37.

    This algorithm uses part (i) of Lemma 1 of Tian03a.

    :param district: A list of variables comprising the district for which we're computing a C factor.
    :param variables: The variables in the graph under analysis, which may be a subgraph of the variables
                      included with the 'topo' paramter.
    :param topo: A list of variables in topological order that includes at least all variables in v.
    :raises TypeError: the district or variable set from which it is drawn contained no variables.
    :raises KeyError: a variable in the district is not in the variable set passed in as a parameter.
    :returns: An expression for Q[district].
    """
    # TODO: Design question: is it faster to have the nested for loops here, or to only take in the district and
    #       a graph, and run a topological sort on the graph with every call to this lemma? I.e., what is the
    #       running time of topological sort on a graph?
    # (It's O(V+E): https://stackoverflow.com/questions/31010922/
    # how-do-i-make-my-topological-sort-to-be-linear-time-code-is-well-annotated)
    # So when we're doing code integration at the end, we can come back and optimize this code.
    result = None
    if len(district) == 0 or len(variables) == 0:
        raise TypeError(
            "Error in _tian_lemma_1_i: the district or variable set from which it is drawn contained no variables."
        )
    if any(v not in variables for v in district):
        raise KeyError(
            "Error in _tian_lemma_1_i: a variable in the district is not in the variable set passed in as a parameter."
        )
    for variable in district:
        preceding_variables = topo[: topo.index(variable)]
        conditioned_variables = [
            variable for variable in preceding_variables if variable in variables
        ]  # V^(i-1)
        tmp = P(variable | conditioned_variables)  # v_i
        if result is None:
            result = tmp
        else:
            result *= tmp
    return result


def _tian_equation_72(
    *,
    vertex: Variable | None,
    variables: set[Variable],
    graph_probability: Expression,  # Q[H]
    topo: list[Variable],
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
    :param variables: The variables in the graph under analysis, namely the set $H$.
    :param graph_probability: The probability of $H$ corresponding to $Q[H]$ in Equation 72.
    :param topo: A list of variables in topological order that includes at least all variables in v.
    :raises KeyError: the input vertex is not in the variable set or not in the topological ordering of graph vertices.
    :returns: An expression for $Q[H^{(i)}]$.
    """
    # $Q[H^{(0)}] = Q[\emptyset] = 1
    if vertex is None:
        return One()
    if vertex not in variables or vertex not in topo:
        raise KeyError(
            "In _tian_equation_72: input vertex is not in the variable set or topological ordering of graph vertices."
        )
    return Sum.safe(
        graph_probability, [v for v in topo[topo.index(vertex) + 1 :] if v in variables]
    )


def _tian_lemma_4_ii(
    *,
    district: list[Variable],
    variables: list[Variable],
    graph_probability: Expression,
    topo: list[Variable],
) -> Expression:
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
    :param variables: The variables in the graph under analysis.
    :param graph_probability: The expression Q corresponding to the set of variables in v. It is
        Q[A] on the line calling Lemma 4 in [tian2003]_, Figure 7.
    :param topo: A list of variables in topological order that includes at least all variables in v.
    :returns: An expression for Q[district].
    """
    # return_value = None
    # return Product.safe([Fraction(_tian_equation_72(topo[topo.index(v)],))])
    raise NotImplementedError("Unimplemented function: _tian_lemma_4_ii")
