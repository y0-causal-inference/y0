"""Utilities for getting ancestral components."""

import logging
from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations_with_replacement

from y0.dsl import CounterfactualVariable, Intervention, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "get_ancestors_of_counterfactual",
    "minimize_counterfactual",
    "get_ancestral_components",
]

logger = logging.getLogger(__file__)


def get_ancestors_of_counterfactual(event: Variable, graph: NxMixedGraph) -> set[Variable]:
    """Get the ancestors of a counterfactual variable.

    This follows [correa22a]_, Definition 2.1 and Example 2.1.
    If the input variable has no interventions, the problem reduces to getting
    ancestors in a graph.

    :param event: A single counterfactual variable.
    :param graph: The graph containing it.
    :returns:
        A set of counterfactual variables. [correa22a]_ consider
        a "counterfactual variable" to also include variables with no interventions.

        .. note::

            In our case, we allow our returned set to include the "Variable" class for
            $Y_0$ syntax, and should test examples including ordinary variables as
            ancestors.
    :raises TypeError:
        get_ancestors_of_counterfactual only accepts a single Variable or CounterfactualVariable
    """
    # There's a TypeError check here because it is easy for a user to pass a set of variables in, instead of
    # a single variable.
    if not isinstance(event, Variable):
        raise TypeError(
            "This function requires a variable, usually a counterfactual variable, as input."
        )

    # logger.debug("In get_ancestors_of_counterfactual: input = " + str(event))
    if not isinstance(event, CounterfactualVariable):
        return graph.ancestors_inclusive(event)

    # This is the set of variables X in [correa22a]_, Definition 2.1.
    intervention_variables = {intervention.get_base() for intervention in event.interventions}
    intervention_values = set(event.interventions)

    graph_minus_in = graph.remove_in_edges(intervention_variables)
    ancestors = graph.remove_out_edges(intervention_variables).ancestors_inclusive(event.get_base())

    ancestors_of_counterfactual_variable: set[Variable] = set()
    for ancestor in ancestors:
        candidate_interventions_z = {
            value
            for value in intervention_values
            if value.get_base() in graph_minus_in.ancestors_inclusive(ancestor)
        }
        # TODO: graph_minus_in.ancestors_inclusive(candidate_ancestor) returns variables.
        # intervention_variables are Interventions, which are a type of Variable.
        # Will these sets intersect without throwing errors?
        if candidate_interventions_z:
            ancestors_of_counterfactual_variable.add(ancestor.intervene(candidate_interventions_z))
        else:
            ancestors_of_counterfactual_variable.add(ancestor)
    # logger.debug(
    #    "In get_ancestors_of_counterfactual: output = " + str(ancestors_of_counterfactual_variable)
    # )
    return ancestors_of_counterfactual_variable


def _minimize_set(*, graph: NxMixedGraph, variables: Iterable[Variable]) -> set[Variable]:
    r"""Minimize a set of counterfactual variables.

    Source: last paragraph in Section 4 of [correa22a]_, before Section 4.1.
    $||\mathbf Y_*|| = {||Y_{\mathbf x}|| | Y_{\mathbf x}} \in {\mathbf Y_*}$.

    :param variables: A set of counterfactual variables to minimize (some may have no interventions).
    :param graph: The graph containing them.
    :returns:
        A set of minimized counterfactual variables such that each minimized variable
        is an element of the original set.
    """
    return {minimize_counterfactual(variable, graph) for variable in variables}


def minimize_counterfactual(variable: Variable, graph: NxMixedGraph) -> Variable:
    r"""Minimize a single variable which is usually counterfactual and may have multiple interventions.

    Source: last paragraph in Section 4 of [correa22a]_, before Section 4.1.

    $||Y_{\mathbf x}|| = Y_{\mathbf t}, where \mathbf T = \mathbf X \intersect An(Y)_{G_{\overline(\mathbf X)}}$
    and $\mathbf t = \mathbf x \intersect \mathbf T$.

    :param variable: A counterfactual variable to minimize (which may have no interventions).
    :param graph: The graph containing them.
    :returns: a minimized counterfactual variable which may omit some interventions from the original one.
    """
    if not isinstance(variable, CounterfactualVariable):
        return variable

    # :math: $\mathbf X$
    intervention_variables: set[Variable] = {
        intervention.get_base() for intervention in variable.interventions
    }
    # :math: $\mathbf T$
    treatment_variables = (
        graph.remove_in_edges(intervention_variables)
        .ancestors_inclusive(variable.get_base())
        .intersection(intervention_variables)
    )
    # :math: $\mathbf t$
    treatment_interventions: frozenset[Intervention] = frozenset(
        intervention
        for intervention in variable.interventions
        if intervention.get_base() in treatment_variables
    )
    # RJC: [correa22a]_ isn't clear about whether the value of a minimized variable shoudl get preserved.
    #      But they write: "Given a counterfactual variable Y_x some values in $\mathbf x$ may be causally
    #      irrelevant to Y once the rest of $\mathbf x$ is fixed." There's nothing in there to suggest that
    #      minimization of $Y_{\mathbf x}$ wouldn't preserve the counterfactual variable's value.  So
    #      we keep the star value.
    # RJC: Sorting the interventions makes the output more predictable and testing is therefore more robust.
    return CounterfactualVariable(
        name=variable.name,
        star=variable.star,
        interventions=treatment_interventions,
    )


def _get_conditioned_variables_in_ancestral_set(
    *,
    conditioned_variables: set[Variable],
    ancestral_set_root_variable: Variable,
    graph: NxMixedGraph,
) -> frozenset[Variable]:
    r"""Retrieve the intersection of the ancestral set of a root variable and a set of conditioned variables.

    This function computes $\mathbf{V}(\|\mathbf{X_{\ast}}\| \cap An(W_{\mathbf{t}}))$,
    the conditioned variables that are ancestors of the input root variable $W_{\mathbf{t}}$.

    Note that [correa22a]_ contains an apparent contradiction: Example 4.5 indicates that
    this operator may return a set of counterfactual variables such as $\{Z_{x}\}$ in the text,
    but in that case this function should compute
    $\mathbf{V}(\|\mathbf{X_{\ast}}\| \cap An(W_{\mathbf{t}}))_{\mathbf{t}}$ and not
    $\mathbf{V}(\|\mathbf{X_{\ast}}\| \cap An(W_{\mathbf{t}}))$. The ambiguity can best be
    resolved with a correction to the article. Meanwhile, because in practice this function
    is used to consider graphs removing edges out of the set of vertices that the function
    returns, we implement it according to its definition in the text of [correa22a]_:
    $\mathbf{V}(\|\mathbf{X_{\ast}}\| \cap An(W_{\mathbf{t}}))$.

    :param conditioned_variables: Following [correa22a]_ this is $\mathbf{X_{\ast}}$, a set of variables that
           are conditioned on in a query. They may be Variable or CounterfactualVariable objects.
    :param ancestral_set_root_variable: following [correa22a]_ this is $W_{\mathbf{t}}$, a variable
           that the function uses to generate its ancestral set.
    :param graph: the relevant graph (the target domain graph in [correa22a]_).
    :returns: an expression for $\mathbf{V}(\|\mathbf{X_{\ast}}\| \cap An(W_{\mathbf{t}}))$,
           that is, the conditioned variables that are ancestors of the input root variable $W_{\mathbf{t}}$.
    """
    minimized_conditioned_variables = _minimize_set(variables=conditioned_variables, graph=graph)
    ancestral_set = get_ancestors_of_counterfactual(ancestral_set_root_variable, graph)
    conditioned_variables_in_ancestral_set = frozenset(
        variable.get_base()
        for variable in minimized_conditioned_variables.intersection(ancestral_set)
    )
    return conditioned_variables_in_ancestral_set


def _get_ancestral_set_after_intervening_on_conditioned_variables(
    *,
    conditioned_variables: set[Variable],
    ancestral_set_root_variable: Variable,
    graph: NxMixedGraph,
) -> frozenset[Variable]:
    r"""Get a variable's ancestral set after first intervening on any conditioned variables that are its ancestors.

    This function computes $An(W_{\mathbf{t}})_{\mathcal{G}_{\underline{\mathbf{X_{\ast}(W_{\mathbf{t}})}}}$,
    per [correa22a]_.

    :param conditioned_variables: Following [correa22a]_ this is $\mathbf{X_{\ast}}$, a set of variables that
           are conditioned on in a query. They may be Variable or CounterfactualVariable objects. This function
           will not intervene on every one of these conditioned variables, just those that are ancestors of
           the ancestral set root variable.
    :param ancestral_set_root_variable: following [correa22a]_ this is $W_{\mathbf{t}}$, a variable
           that the function uses to generate its ancestral set.
    :param graph: the relevant graph (the target domain graph in [correa22a]_).
    :returns: a set of variables corresponding to
           $An(W_{\mathbf{t}})_{\mathcal{G}_{\underline{\mathbf{X_{\ast}(W_{\mathbf{t}})}}}$,
           that is, the ancestors of $W_{\mathbf{t}}$ in a graph intervening on those
           conditioned variables that would otherwise be in $W_{\mathbf{t}}$'s ancestral set were the
           interventions not applied.
    """
    conditioned_variables_in_ancestral_set = _get_conditioned_variables_in_ancestral_set(
        conditioned_variables=conditioned_variables,
        ancestral_set_root_variable=ancestral_set_root_variable,
        graph=graph,
    )
    graph_with_interventions = graph.remove_out_edges(conditioned_variables_in_ancestral_set)
    new_ancestral_set = get_ancestors_of_counterfactual(
        event=ancestral_set_root_variable, graph=graph_with_interventions
    )
    return frozenset(new_ancestral_set)


def _compute_ancestral_components_from_ancestral_sets(
    *,
    ancestral_sets: set[frozenset[Variable]],
    graph: NxMixedGraph,
) -> frozenset[frozenset[Variable]]:
    r"""Construct a set of ancestral components from ancestral sets following Definition 4.2 of [correa22a]_.

    Note: [correa22a]_ is silent regarding an algorithm for efficiently combining the input
        ancestral sets for this function. This implementation runs in time $O(V^{3})$, where $V$
        is the number of vertices in the graph. The implementation matches Correa and
        Bareinboim's efficiency analysis in Appendix B of [correa22a]_.

    :param ancestral_sets: These are the sets
           $An(W_{\mathbf{t}})_{\mathcal{G}_{\underline{\mathbf{X_{\ast}(W_{\mathbf{t}})}}}$ in
           Definition 4.2 of [correa22a]_. They are induced by $\mathbf{W_{\ast}}$, given $\mathbf{X_{\ast}}$.
    :param graph: the relevant graph $\mathcal{G}$ (without intervening on any conditioned variables).
    :returns: the sets $\mathbf{A}_{1},\mathbf{A}_{2},\ldots$ that form a partition over $An(\mathbf{W_{\ast}})$,
           made of unions of the input ancestral sets. Two sets are combined via a union operation if they are
           not disjoint or there exists a bidirected arrow in $\mathcal{G}$ connecting variables
           in those sets. (Definition 4.2 of [correa22a]_.)
    """
    # Initialization

    # O(V)
    ancestral_components: set[frozenset[Variable]] = set(ancestral_sets)

    # O(V^3)
    merged_ancestral_components_using_vertices: set[frozenset[Variable]] = (
        _merge_frozen_sets_with_common_vertices(ancestral_components)
    )

    # O(V^3)
    merged_ancestral_components: set[frozenset[Variable]] = (
        _merge_frozen_sets_linked_by_bidirectional_edges(
            input_sets=merged_ancestral_components_using_vertices, graph=graph
        )
    )

    # Final check
    # O(V)
    # vertex_to_ancestral_component_mappings: defaultdict[
    #    Variable, set[frozenset[Variable]]
    # ] = defaultdict(set)
    # for s in merged_ancestral_components:
    #    # original_to_merged_ancestral_set_mappings[s] = s
    #    for v in s:
    #        # Two counterfactual variables with the same base variable and different interventions
    #        # may be considered not disjoint, because they correspond to the same
    #        # graph vertex. So we do the indexing by primitive variable, not counterfactual
    #        # variable. See the proof of Lemma A.5 of [correa22a]_.
    #        vertex_to_ancestral_component_mappings[v.get_base()].add(s)

    # 1. Prove this next check never gets triggered. Assume: there's a vertex in more than one component
    # of merged_ancestral_outcomes.
    #
    #    Then, either _merge_frozen_sets_with_common_vertices() returned two frozen sets with a vertex
    # in common or it didn't, and _merge_frozen_sets_linked_by_bidirectional_edges() therefore
    # split a frozen set.  _merge_frozen_sets_linked_by_bidirectional_edges() only ever applies
    # union operations to input sets, so the latter condition could not have happened. That leaves
    # _merge_frozen_sets_with_common_vertices() returning two frozen sets with a vertex in common.
    # Again, it only applies union operations to input sets, so there must have existed two input sets
    # with a vertex in common that didn't get merged.
    # So _merge_frozen_sets_with_common_vertices() returned two frozen sets with a vertex in common.
    #
    #    Consider two such sets, A and B, that both contain vertex V (or equivalently, which contain two
    # counterfactual variables with base variable V). (_convert_counterfactual_variables_to_base_variables()
    # works: a variable is either its own base in which we're done, or it's not in which case this
    # one-line function returns its base.) A and B definitely get compared during the for loop that
    # calls combinations_with_replacement() on input_sets, at which point B goes into the adjaceny
    # list for A and A goes into the adjacency list for B. Next, observe that every vertex V gets
    # visited at least once by dft(), either by a call to dft(V,V) or recursively, by a call to
    # dft(V, W) where W is some other node with V as its neighbor. So, one of three things can happen:
    # 1. A gets visited for the first time with a call dft(A,A), in which case dft(A,A) adds
    #    A to components[A]. The final line will then union sets A and B together as B is a
    #    neighbor of A.
    # 2. B gets visited for the first time with a call dft(B,B), in which case dft(B,B) adds
    #    B to components[B]. The final line will then union sets A and B together as A is a
    #    neighbor of B.
    # 3. A gets visited for the first time with a call dft(A,C), at which point A goes into
    #    components[C]. During the last line of this function, A and its neighbors will
    #    then get unioned together, so A and B will go into one set.
    #    a. If B hasn't been visited, then dft(B, C) gets called, at which point B goes into
    #       components[C]. Then A, B, and C will get merged during the last line of this function
    #       because A and B will be neighbors of C.
    #    b. B cannot have already been visited. Assume it was visited already. It could have been per
    #       a call dft(B,B) which would lead to a call dft(A,B) since A hasn't been visited.
    #       But we visited A for the first time with a call dft(A,C), a contradiction. It could have
    #       been per a call dft(B,C) in which case B is also a neighbor of C, and A, B, and C will get
    #       merged during the last line of this function. Or it could have been per a call dft(B,D)
    #       where D is some other vertex. But since dft is a depth-first traversal, a call
    #       dft(B,D) would have led to a call dft(A,D) and A was visited for the first time
    #       in the call dft(A,C). So all three possibilities lead to contradictions and B cannot
    #       have already been visited.
    #
    #    We have seen that in all possible cases, _merge_frozen_sets_with_common_vertices() leads
    # to A and B merged into a frozen set. So we must reject our assumption that there's a vertex in
    # more than one component of merged_ancestral_outcomes, and this validity check is unnecessary
    # and the line of code will never be run.
    #
    # if any(
    #    len(vertex_to_ancestral_component_mappings[v]) > 1
    #    for v in vertex_to_ancestral_component_mappings.keys()
    # ):
    #    logger.debug(
    #        "In _compute_ancestral_components_from_ancestral_sets: a vertex is still associated "
    #        + "with more than one ancestral component during final checks."
    #    )
    #    for v in vertex_to_ancestral_component_mappings.keys():
    #        logger.debug("Vertex: " + str(v))
    #        logger.debug(
    #            "   Ancestral components associated with this vertex: "
    #            + str(vertex_to_ancestral_component_mappings[v])
    #        )
    #    raise ValueError(
    #        "In _compute_ancestral_components_from_ancestral_sets: a vertex "
    #        + "is still associated with more than one ancestral component during final checks."
    #    )

    # 2. Prove this next check never gets triggered. Assume: there exist two sets A and B such that
    # A is in one component of merged_ancestral_outcomes and B is in a second component, and
    # there exist vertices v1 in A and v2 in B such that a bidirected edge v1 <-> v2 exists in the
    # graph.
    #
    #    Either _merge_frozen_sets_with_common_vertices() placed A and B into the same component or
    # it didn't. It could not have done so because _merge_frozen_sets_linked_by_bidirectional_edges()
    # would then have had to split a component, and the function only calls union operations on
    # input sets. So A and B were in separate input sets passed into
    # _merge_frozen_sets_linked_by_bidirectional_edges(). Therefore we try to show that
    # _merge_frozen_sets_linked_by_bidirectional_edges() returned
    # A in one component and B in a second component of its output, such that
    # there exist vertices v1 in A and v2 in B with a bidirected edge v1 <-> v2 existing in the
    # graph. Clearly A and B can't be in the same input set. Because the edge is in the graph, it
    # gets visited by the for loop visiting graph.undirected.edges, and thus A is a neighbor of
    # B in the adjacency list and B is a neighbor of A. The rest of the code and the rest of the proof
    # is exactly the same as the proof that _merge_frozen_sets_with_common_vertices() always links
    # sets that share a common vertex / counterfactual variables with the same base variable.
    # We have to reject our assumption and thus the next check never gets triggered.
    # if any(
    #    v1 in vertex_to_ancestral_component_mappings
    #    and v2 in vertex_to_ancestral_component_mappings
    #    and vertex_to_ancestral_component_mappings[v1] != vertex_to_ancestral_component_mappings[v2]
    #    for v1, v2 in graph.undirected.edges
    # ):
    #    logger.debug(
    #        "In _compute_ancestral_components_from_ancestral_sets: a bidirected edge still connects "
    #        + "two ancestral components during final checks."
    #    )
    #    raise ValueError(
    #        "In _compute_ancestral_components_from_ancestral_sets: a bidirected edge "
    #        + "still connects two ancestral components during final checks."
    #    )
    #
    #   These two proofs together verify that this algorithm meets the requirements of the
    # last sentence of Definition 4.2 in [correa22a]_.
    return frozenset(merged_ancestral_components)


def get_ancestral_components(
    *, conditioned_variables: set[Variable], root_variables: set[Variable], graph: NxMixedGraph
) -> frozenset[frozenset[Variable]]:
    r"""Compute a set of ancestral components corresponding to Definition 4.2 of [correa22a].

    :param conditioned_variables: The set of variables $\mathbf{X_{\ast}}$ on which
           a counterfactual query has been conditioned.
    :param root_variables: The set of variables $\mathbf{W_{\ast}}$, such that
          $\mathbf{X_{\ast}} \subseteq \mathbf{W_{\ast}}$, that we use to construct ancestral sets
          for each variable in $\mathbf{W_{\ast}}$ and ancestral components from those sets.
    :param graph: the relevant graph $\mathcal{G}$ (without intervening on any conditioned variables).
    :returns: the sets $\mathbf{A}_{1},\mathbf{A}_{2},\ldots$ that form a partition over $An(\mathbf{W_{\ast}})$,
           made of unions of the input ancestral sets. Two sets are combined via a union operation if they are
           not disjoint or there exists a bidirected arrow in $\mathcal{G}$ connecting variables
           in those sets. (Definition 4.2 of [correa22a]_.)
    """
    ancestral_sets: set[frozenset[Variable]] = {
        _get_ancestral_set_after_intervening_on_conditioned_variables(
            conditioned_variables=conditioned_variables, ancestral_set_root_variable=v, graph=graph
        )
        for v in root_variables
    }
    logger.debug("In _get_ancestral_components: ancestral_sets = " + str(ancestral_sets))
    ancestral_components: frozenset[frozenset[Variable]] = (
        _compute_ancestral_components_from_ancestral_sets(
            ancestral_sets=ancestral_sets, graph=graph
        )
    )
    logger.debug(
        "In _get_ancestral_components: computed these ancestral components: "
        + str(ancestral_components)
    )
    return ancestral_components


def _merge_frozen_sets_with_common_vertices(
    input_sets: set[frozenset[Variable]],
) -> set[frozenset[Variable]]:
    r"""Merge a set of frozen sets of counterfactual variables using common vertices.

    Two sets get merged if they share a common graph vertex. That is, there exists a counterfactual
    variable in the first set and a counterfactual variable in the first set such that both
    counterfactual variables have the same base variable name. This algorithm treats input sets as
    nodes in a graph, defines an edge in the graph in all cases for which two input sets share
    a common vertex, computes the union of the vertices in each connected component in
    the resulting graph as a frozen set, and returns the set of those frozen sets.

    Total running time: O(V^3) where V is the number of vertices in the graph.

    :param input_sets: the counterfactual variables in question.
    :returns: A set containing the merged frozen sets.
    """
    # From this StackOverflow post: https://stackoverflow.com/questions/42211947/merging-sets-with-common-elements
    result = set()
    visited: set[frozenset[Variable]] = set()
    components: defaultdict[frozenset[Variable], list[frozenset[Variable]]] = defaultdict(list)
    adj_list: defaultdict[frozenset[Variable], list[frozenset[Variable]]] = defaultdict(list)

    # DFS runs in O(V+E) time where V is the number of vertices and E is the number of edges in a graph
    # (Cormen, Leisserson, Rivest and Stein 2002:541).
    #
    # Here V is the number of input sets and E is the number of edges are the number of sets with common vertices.
    # The number of sets with common vertices is O(V), e.g., there can be V-1 sets each containing two vertices
    # that form a giant cycle, so that's O(V) calls to dft. The number of edges are also O(V) in this case so
    # dft() runs in O(V+V) = O(V) time.
    def _dft(node: frozenset[Variable], key: frozenset[Variable]) -> None:
        visited.add(node)
        components[key].append(node)
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                _dft(neighbor, key)

    # Create a hash table to speed the process of checking if pairs of sets contain common vertices
    # after their members are converted from counterfactual variables to their base variables.
    # O(V) running time
    converted_sets: defaultdict[frozenset[Variable], frozenset[Variable]] = defaultdict(frozenset)
    for s in input_sets:
        converted_sets[s] = get_base_variables(s)

    # Prove this never gets triggered. Assume: there's a vertex in more than one component
    # of merged_ancestral_outcomes.
    # Then, either _merge_frozen_sets_with_common_vertices() returned two frozen sets with a vertex
    # in common or it didn't, and _merge_frozen_sets_linked_by_bidirectional_edges() therefore
    # split a frozen set.  _merge_frozen_sets_linked_by_bidirectional_edges() only ever applies
    # union operations to input sets, so the latter condition could not have happened. That leaves
    # _merge_frozen_sets_with_common_vertices() returning two frozen sets with a vertex in common.
    # Again, it only applies union operations to input sets, so there must have existed two input sets
    # with a vertex in common that didn't get merged.
    # So _merge_frozen_sets_with_common_vertices() returned two frozen sets with a vertex in common.

    # Consider two such sets, A and B, that both contain vertex V (or equivalently, which contain two
    # counterfactual variables with base variable V). (_convert_counterfactual_variables_to_base_variables()
    # works: a variable is either its own base in which we're done, or it's not in which case this
    # one-line function returns its base.) A and B definitely get compared during the for loop that
    # calls combinations_with_replacement() on input_sets, at which point B goes into the adjaceny
    # list for A and A goes into the adjacency list for B. Next, observe that every vertex V gets
    # visited at least once by dft(), either by a call to dft(V,V) or recursively, by a call to
    # dft(V, W) where W is some other node with V as its neighbor. So, one of three things can happen:
    # 1. A gets visited for the first time with a call dft(A,A), in which case dft(A,A) adds
    #    A to components[A]. The final line will then union sets A and B together as B is a
    #    neighbor of A.
    # 2. B gets visited for the first time with a call dft(B,B), in which case dft(B,B) adds
    #    B to components[B]. The final line will then union sets A and B together as A is a
    #    neighbor of B.
    # 3. A gets visited for the first time with a call dft(A,C), at which point A goes into
    #    components[C]. During the last line of this function, A and its neighbors will
    #    then get unioned together, so A and B will go into one set.
    #    a. If B hasn't been visited, then dft(B, C) gets called, at which point B goes into
    #       components[C]. Then A, B, and C will get merged during the last line of this function
    #       because A and B will be neighbors of C.
    #    b. B cannot have already been visited. Assume it was visited already. It could have been per
    #       a call dft(B,B) which would lead to a call dft(A,B) since A hasn't been visited.
    #       But we visited A for the first time with a call dft(A,C), a contradiction. It could have
    #       been per a call dft(B,C) in which case B is also a neighbor of C, and A, B, and C will get
    #       merged during the last line of this function. Or it could have been per a call dft(B,D)
    #       where D is some other vertex. But since dft is a depth-first traversal, a call
    #       dft(B,D) would have led to a call dft(A,D) and A was visited for the first time
    #       in the call dft(A,C). So all three possibilities lead to contradictions and B cannot
    #       have already been visited.
    # We have seen that in all possible cases, _merge_frozen_sets_with_common_vertices() leads
    # to A and B merged into a frozen set. So we must reject our assumption that there's a vertex in
    # more than one component of merged_ancestral_outcomes, and this validity check is unnecessary
    # and the line of code will never be run.

    # The adjacency lists for both A and B get visited during
    # the loop through the adj_list dictionary. If A and B get placed into the same connected component
    # during this step, then they get merged during the union operation during the last line, at which point
    # _merge_frozen_sets_with_common_vertices() does not

    # We wish to show that A and B will get placed
    # into a connected component.
    #
    # One of three things can happen:

    # 3. Note that if A and B are to be ultimately merged, they must be part of a connected
    #
    # B gets visited for the first time with a call dft(B,B) and then calls dft(A,B) within
    # When that happens, either A has been visited before during a
    # call to dft() or it has not. If it has not, then dft(A,A) gets called,
    # Runs in O(V^2) time
    for r1, r2 in combinations_with_replacement(input_sets, 2):
        if converted_sets[r1] & converted_sets[r2]:
            adj_list[r1].append(r2)
            adj_list[r2].append(r1)

    # O(V^2): outer loop is O(V), inner loop is O(V)
    for node in adj_list:
        if node not in visited:
            _dft(node, node)

    # O(V) components, and for each component O(V) neighbors receive a union operation that runs
    # in O(V) time. So, total running time of this loop (and thus the algorithm) is O(V^3).
    for node, neighbors in components.items():
        result.add(node.union(*neighbors))
    return result


def _merge_frozen_sets_linked_by_bidirectional_edges(  # noqa:C901
    input_sets: set[frozenset[Variable]],
    graph: NxMixedGraph,
) -> set[frozenset[Variable]]:
    r"""Merge a set of frozen sets of counterfactual variables using common bidirectional edges.

    Two sets get merged if a bidirectional edge connects them.
    Running time: O(V^3)

    :param input_sets: the counterfactual variables in question.
    :param graph: the graph in question.
    :returns: A set containing the merged frozen sets.
    """
    # Modified from https://stackoverflow.com/questions/42211947/merging-sets-with-common-elements
    result = set()
    visited: set[frozenset[Variable]] = set()
    components: defaultdict[frozenset[Variable], list[frozenset[Variable]]] = defaultdict(list)
    adj_list: defaultdict[frozenset[Variable], list[frozenset[Variable]]] = defaultdict(list)

    # O(V)
    # FIXME this DFT code is duplicate from above
    def _dft(node: frozenset[Variable], key: frozenset[Variable]) -> None:
        visited.add(node)
        components[key].append(node)
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                _dft(neighbor, key)

    # O(V^2)
    converted_sets: defaultdict[frozenset[Variable], frozenset[Variable]] = defaultdict(frozenset)
    for s in input_sets:
        converted_sets[s] = get_base_variables(s)

    # Loop through the sets and create a vertex to set mapping. O(V^2)
    vertices_to_input_sets: defaultdict[Variable, frozenset[Variable]] = defaultdict(frozenset)
    for s in input_sets:
        for v in converted_sets[s]:
            vertices_to_input_sets[v] = s

    # Connect sets to themselves. O(V)
    for s in input_sets:
        adj_list[s].append(s)

    # Connect sets joined by bidirectional edges. O(E)
    for e in graph.undirected.edges:
        v1 = e[0]
        v2 = e[1]
        r1 = vertices_to_input_sets[v1]
        r2 = vertices_to_input_sets[v2]
        if r1 != r2:
            adj_list[r1].append(r2)
            adj_list[r2].append(r1)

    # O(V^2)
    for node in adj_list:
        if node not in visited:
            _dft(node, node)

    # O(V^3).
    for node, neighbors in components.items():
        result.add(node.union(*neighbors))

    # So, total running time is O(V^3+E). But since E can't be greater than V^2,
    # the running time is just O(V^3).
    return result


def get_base_variables(variables: frozenset[Variable]) -> frozenset[Variable]:
    r"""Replace a set of counterfactual variables with a set of corresponding base variables.

    :param variables: the counterfactual variables in question.
    :returns: The base variables.
    """
    return frozenset(variable.get_base() for variable in variables)
