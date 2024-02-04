"""An implementation of Tian and Pearl's identification algorithm from [tian03a]_."""

import logging
from typing import Collection

from y0.dsl import Expression, Fraction, One, P, Probability, Product, Sum, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "tian_pearl_identify",
]

logger = logging.getLogger(__name__)


def tian_pearl_identify(
    *,
    input_variables: frozenset[Variable],
    input_district: frozenset[Variable],
    district_probability: Expression,
    graph: NxMixedGraph,
    topo: list[Variable],
) -> Expression | None:
    """Implement the IDENTIFY algorithm as presented in [tian03a]_ with pseudocode in [correa22a]_ (Algorithm 5).

    Tikka and colleagues implemented this algorithm in the R package Causal Effect ([santikka20b]_). We draw from that
    implementation. Their version also keeps track of the structure of calls
    :param input_variables: The set of variables, C, for which we're checking if causal identification is possible.
    :param input_district: The C-component, T, containing C.
    :param district_probability: The expression Q[T] as per [tian2003]_, Equation 35.
    :param graph: The relevant graph.
    :param topo: A list of variables in topological order that includes all variables in the graph and may contain more.
    :raises KeyError: at least one input variable is not in the input district or at least one input district variable
                      is not in the topologically sorted list of graph variables.
    :raises TypeError: the subgraph of the input graph G comprised of thevertices in the input vertex set T should
                      should have only district and has more.
    :returns: An expression for Q[C] in terms of Q, or Fail.
    """
    if not input_variables.intersection(input_district) == input_variables:
        # if not all(v in input_district for v in input_variables):
        raise KeyError(
            "In tian_pearl_identify: at least one of the input variables C is not in the input district T."
        )
    if not input_district.intersection(set(topo)) == input_district:
        raise KeyError(
            "In tian_pearl_identify: at least one input district variable is not in the "
            + "topologically sorted variable list."
        )
    district_subgraph = graph.subgraph(vertices=input_district)  # $G_{T}$
    if len(district_subgraph.districts()) > 1:
        raise TypeError(
            "In tian_pearl_identify: the subgraph of the input graph G comprised of the"
            + " vertices in the input vertex set T should have only district and has more."
        )
    if (not isinstance(district_probability, Expression)) | isinstance(
        district_probability, Expression
    ) and not (
        isinstance(district_probability, Sum)
        | isinstance(district_probability, Product)
        | isinstance(district_probability, Fraction)
        | isinstance(district_probability, Probability)
    ):
        raise TypeError(
            "In tian_pearl_identify: the district probability must be an expression that is a "
            + "Sum, Product, Fraction, or Probability."
        )
    # ordered_graph_vertices = [v for v in topo if v in graph.nodes()]  # This is V

    ancestral_set = frozenset(
        district_subgraph.ancestors_inclusive(input_variables)
    )  # A = Ancestors of C in $G_{T}$

    # Next, Tikka has an additional line intersecting the ancestral set with the set T in case any C was not in T,
    # but we raise an error in that case as a pre-processing step, so we omit that line.

    ordered_ancestral_set = [a for a in topo if a in ancestral_set]
    if ancestral_set == input_variables:
        logger.warning("In tian_pearl_identify: A = C. Applying Lemma 3.")
        rv = _tian_equation_69(
            ancestral_set=ancestral_set,
            subgraph_variables=input_district,
            subgraph_probability=district_probability,
            graph_topo=topo,
        )
        logger.warning("   Returning Q value: " + str(rv))
    elif ancestral_set == input_district:
        logger.warning("In tian_pearl_identify: A = T. Returning None (i.e., FAIL).")
        rv = None
    elif input_variables.issubset(ancestral_set) and ancestral_set.issubset(input_district):
        ancestral_set_subgraph = graph.subgraph(vertices=ordered_ancestral_set)
        ancestral_set_subgraph_districts = list(ancestral_set_subgraph.districts())
        targeted_ancestral_set_subgraph_district = ancestral_set_subgraph_districts[
            [
                input_variables.issubset(district) for district in ancestral_set_subgraph_districts
            ].index(True)
        ]
        # t_prime = [district for district in ancestral_set_subgraph_districts if
        #             input_variables.intersect(district)==input_variables][0]
        # ordered_t_prime_vertices = [v for v in topo if v in t_prime]
        # t_one = t_prime.intersection(ancestral_set) # RC: This line is in Tikka
        if (
            isinstance(district_probability, Fraction)
            or isinstance(district_probability, Product)
            or isinstance(district_probability, Sum)
        ):  # Compute Q[A] from Lemma 3
            ancestral_set_probability = _tian_equation_69(
                ancestral_set=ancestral_set,
                subgraph_variables=input_district,
                subgraph_probability=district_probability,  # Q[T]
                graph_topo=topo,
            )
        elif isinstance(district_probability, Probability):
            ancestral_set_probability = P(
                ordered_ancestral_set[0].joint(ordered_ancestral_set[1:])
                | district_probability.parents
            )
        else:
            raise TypeError(
                "In tian_pearl_identify: the district probability is an expression of an unknown type."
            )
        # Get Q[T'] by Lemma 4 or Lemma 1
        targeted_ancestral_set_subgraph_district_probability = _compute_c_factor(
            district=targeted_ancestral_set_subgraph_district,
            subgraph_variables=ancestral_set,
            subgraph_probability=ancestral_set_probability,
            graph_topo=topo,
        )
        logger.warning("In tian_pearl_identify: about to recursively call tian_pearl_identify.")
        logger.warning("    C = " + str(input_variables))
        logger.warning("    T' = " + str(targeted_ancestral_set_subgraph_district))
        logger.warning("    Q[T'] =" + str(targeted_ancestral_set_subgraph_district_probability))
        logger.warning("    graph nodes = " + str(list(graph.nodes())))
        logger.warning("    topo = " + str(topo))
        rv = tian_pearl_identify(
            input_variables=input_variables,
            input_district=targeted_ancestral_set_subgraph_district,
            district_probability=targeted_ancestral_set_subgraph_district_probability,
            graph=graph,
            topo=topo,
        )
        logger.warning(
            "In tian_pearl_identify: returned from recursive call to tian_pearl_identify."
        )
        logger.warning("    Return value = " + str(rv))
    return rv


def _do_tian_pearl_identify_line_1(
    input_variables: set[Variable],
    input_district: set[Variable],
    graph: NxMixedGraph,
) -> set[Variable] | None:
    """Implement line 1 of the IDENTIFY algorithm in [tian03a]_ and [correa22a]_ (Algorithm 5)."""
    raise NotImplementedError("Unimplemented function: ctfTRu")


def _tian_lemma_1_i(
    *,
    district: Collection[Variable],
    graph_probability: Probability,
    topo: list[Variable],
) -> Expression:
    """Compute the Q value associated with the C-component (district) in a graph as per [tian03a]_, Equation 37.

    This algorithm uses part (i) of Lemma 1 of Tian03a.

    :param district: A list of variables comprising the district for which we're computing a C factor.
    :param graph_probability: the Q value for the full graph.

    :param topo: a topological sort of the vertices in the graph.
    :raises TypeError: the district or variable set from which it is drawn contained no variables.
    :raises KeyError: a variable in the district is not in the topological sort of the graph vertices.
    :returns: An expression for Q[district].
    """
    # (Topological sort is O(V+E): https://stackoverflow.com/questions/31010922/)
    variables = set(topo)
    logger.warning("In _tian_lemma_1_i: topo = " + str(topo))
    probabilities = []
    if len(district) == 0 or len(variables) == 0:
        raise TypeError(
            "Error in _tian_lemma_1_i: the district or variable set from which it is drawn contained no variables."
        )
    if any(v not in variables for v in district):
        raise KeyError(
            "Error in _tian_lemma_1_i: a variable in the district is not in the topological sort of the graph vertices."
        )
    # A little subtle so it deserves a comment: the Q value passed into Tian's Identify function may
    # already be conditioned on some variables that are in G but not in the subgraph H. In applying Lemma 1
    # (but not Lemma 4), we have to make sure we're also conditioning on those variables.
    graph_probability_parents = set(graph_probability.parents)
    for variable in district:
        preceding_variables = topo[: topo.index(variable)]
        conditioned_variables = graph_probability_parents.union(preceding_variables)  # V^(i-1)
        probability = P(variable | conditioned_variables)  # v_i
        probabilities.append(probability)
    logger.warning("In _tian_lemma_1_i: returning " + str(Product.safe(probabilities)))
    return Product.safe(probabilities)


def _tian_equation_72(
    *,
    vertex: Variable | None,
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
    :param graph_probability: The probability of $H$ corresponding to $Q[H]$ in Equation 72.
    :param topo: a topological sorting of the subgraph under analysis, $G_{H}$.
    :raises KeyError: the input vertex is not in the variable set or not in the topological ordering of graph vertices.
    :returns: An expression for $Q[H^{(i)}]$.
    """
    # $Q[H^{(0)}] = Q[\emptyset] = 1$
    if vertex is None:
        return One()
    variables = set(topo)
    logger.warning("In _tian_equation_72: input vertex is " + str(vertex))
    logger.warning("   and variables are " + str(variables))
    logger.warning("   and topo is " + str(topo))
    if vertex not in variables:
        raise KeyError("In _tian_equation_72: input vertex %s is not in the input graph.", vertex)

    ranges = [v for v in topo[topo.index(vertex) + 1 :]]
    return Sum.safe(graph_probability, ranges)


def _tian_lemma_4_ii(
    *, district: Collection[Variable], graph_probability: Expression, topo: list[Variable]
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
    :param graph_probability: The expression Q corresponding to the set of variables in v. It is
        Q[A] on the line calling Lemma 4 in [tian2003]_, Figure 7.
    :param topo: a topological ordering of the vertices in the subgraph $G_{H}$ in question.
    :returns: An expression for Q[district].
    """

    def _get_expression_from_index(index: int) -> Expression:  # Compute $Q[H^{i}]$ given i
        current_index_expr = _tian_equation_72(
            vertex=topo[index], graph_probability=graph_probability, topo=topo
        )
        if index == 0:
            logger.warning("In _one_round: index = 0\n  returning %s", current_index_expr)
            return current_index_expr
        previous_index_expr = _tian_equation_72(
            vertex=topo[index - 1], graph_probability=graph_probability, topo=topo
        )
        rv = Fraction(current_index_expr, previous_index_expr)
        logger.warning(
            "In one_round with index > 1:\n\ttopo: %s\n\treturn_num: %s\n\treturn_den: %s\n\treturning: %s",
            topo,
            current_index_expr,
            previous_index_expr,
            rv,
        )
        return rv

    expressions = []
    for _, vertex in enumerate(district):
        logger.warning("In Lemma 4(ii): vertex = " + str(vertex))
        index = topo.index(vertex)
        expression = _get_expression_from_index(index)
        expressions.append(expression)
        logger.warning("\nIndex = %d, Q[H^(i)] = %s", index, expression)

    rv = Product.safe(expressions)
    # TODO: We can simplify this product by cancelling terms in the numerator and denominator.
    logger.warning("Returning product: %s", rv)
    return rv


def _compute_c_factor(
    *,
    district: Collection[Variable],
    subgraph_variables: Collection[Variable],
    subgraph_probability: Expression,
    graph_topo: list[Variable],
) -> Expression:
    """Compute the Q value associated with the C-component (district) in a graph as per [tian03a]_ and [santikka20a]_.

    This algorithm uses both Lemma 1, part (i) of Tian03a (Equation 37) and Lemma 4 of Tian 03a (Equations 71 and 72).

    :param district: A list of variables comprising the district C for which we're computing a C factor.
    :param subgraph_variables: The variables in the subgraph T under analysis.
    :param subgraph_probability: The expression Q corresponding to the set of variables in T. As an example, this
              quantity would be Q[A] on the line calling Lemma 4 in [tian2003]_, Figure 7.
    :param graph_topo: A list of variables in topological order that includes all variables in G, where T is contained
              in G.
    :raises TypeError: In _compute_c_factor: expected the subgraph_probability parameter to be a simple probability.
    :returns: An expression for Q[district].
    """
    # The graph_topo is the ordering of vertices in G, but the lemmas use the topological sorting in a subgraph H of G.
    # We take in the topological ordering of G to make testing easier, as there could be multiple ways to
    # sort the vertices in H topologically. It is also faster as topological sort is O(V+E) and getting
    # subgraph_topo below is O(V).
    subgraph_topo = [v for v in graph_topo if v in subgraph_variables]
    logger.warning("In _compute_c_factor: graph_topo = " + str(graph_topo))
    logger.warning("In _compute_c_factor: subgraph_topo = " + str(subgraph_topo))
    if (
        isinstance(subgraph_probability, Fraction)
        or isinstance(subgraph_probability, Product)
        or isinstance(subgraph_probability, Sum)
    ):
        logger.warning("In _compute_c_factor: calling _tian_lemma_4_ii")
        rv = _tian_lemma_4_ii(
            district=district, graph_probability=subgraph_probability, topo=subgraph_topo
        )
        logger.warning("Returning from _compute_c_factor: " + str(rv))
        return rv
        # return _tian_lemma_4_ii(
        #    district=district, graph_probability=subgraph_probability, topo=subgraph_topo
        # )
    else:
        if not isinstance(subgraph_probability, Probability):
            raise TypeError(
                "In _compute_c_factor: expected the subgraph_probability "
                + str(subgraph_probability)
                + " to be a simple probability."
            )
        else:
            return _tian_lemma_1_i(
                district=district, graph_probability=subgraph_probability, topo=subgraph_topo
            )


def _tian_equation_69(
    *,
    ancestral_set: frozenset[Variable],  # A
    subgraph_variables: frozenset[Variable],  # T
    subgraph_probability: Expression,
    graph_topo: list[Variable],  # topological ordering of
) -> Expression:
    """Compute the Q value associated with a subgraph as per Equation 69 of [tian03a]_.

    This algorithm uses both Lemma 3 of Tian 03a (Equation 69).

    :param ancestral_set: A set of variables (W in Equation 69, A in Figure 7 of [tian03a]_) that comprise the
           ancestral set of some other variables (unspecified in Equation 69, and C in Figure 7 of [tian03a]_).
    :param subgraph_variables: The variables in the subgraph under analysis (C in Equation 69, and T in Figure 7).
    :param subgraph_probability: The expression Q corresponding to Q[C] in Equation 69 and Q[T] in Figure 7.
    :param graph_topo: A list of variables in topological order that includes all variables in the graph
           (i.e., V in Equation 69 and G in Figure 7).
    :returns: An expression for Q[ancestral_set].
    """
    # A is W, I want Q[A] = Q[W]
    # T is C, I know Q[C] = the input Q for identify
    # T\A is W', so Sum_{T\A}{Q[T]} = Sum_{W'}{Q[C]} = Q[W] = Q[A]
    # The next two lines are included so the summation shows the marginalization variables in topological order
    marginalization_set = subgraph_variables - ancestral_set
    marginalization_variables = [v for v in graph_topo if v in marginalization_set]
    return Sum.safe(subgraph_probability, marginalization_variables)
