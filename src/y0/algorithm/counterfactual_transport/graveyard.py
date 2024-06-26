"""Unused functions implemented for CFT."""

import unittest
from collections import defaultdict
from typing import Collection

from y0.algorithm.transport import create_transport_diagram, transport_variable
from y0.dsl import Intervention, Variable, W, X, Y, Z
from y0.graph import NxMixedGraph


def is_consistent(
    event_1: list[tuple[Variable, Intervention]], event_2: list[tuple[Variable, Intervention]]
) -> bool:
    r"""Check whether two lists of values are consistent.

    .. todo:: if this function isn't important, please delete

    Note that we could more elegantly represent the values as sets of interventions, but we use
    (variable, intervention) tuples to represent values instead. Because values are often
    associated with counterfactual variables for our purposes, in practice we need these tuples
    to associate a value with its corresponding counterfactual variable without losing track of the
    interventions associated with that counterfactual variable.

    :math: Two values $\mathbf{x} and \mathbf{z}$ are said to be consistent if they share the common values
    for $\mathbf{X} \cap \mathbf{Z}$.
    We assume the domain of every variable is finite.

    :param event_1:
        A tuple associating $\mathbf{X}$, a set of counterfactual variables (or regular variables)
        in $\mathbf{V}$ with $\mathbf{x}$, a set of values for $\mathbf{X}$. We encode the
        counterfactual variables as Variable objects, and the values as Intervention objects.
    :param event_2:
        A tuple associating $\mathbf{Z}$, a set of counterfactual variables (or regular variables)
        in $\mathbf{V}$ with $\mathbf{z}$, a set of values for $\mathbf{Z}$. We encode the
        counterfactual variables as Variable objects, and the values as Intervention objects.
    :returns:
        A boolean indicating whether the values in $\mathbf{x}$ (event_1) and $\mathbf{z}$ (event_2) are consistent.
    """
    # Key the input variables by their domain. Create a dictionary such that the key is variable.get_base() and
    # the value is a set of interventions.

    # We don't use variable.get_base() because even though [correa22a]_ do not include counterfactual
    # variables in their definition for consistent values and merely use the regular notation for variables,
    # that is because the definition for consistency shows up in section 1.1, before they describe a notation
    # for counterfactual variables. They operationalize their definition of consistency in the simplify
    # algorithm, where they speak of consistent values for counterfactual variables such as $Y_\mathbf{X}$.
    # Clearly in that section they do not mean for the reader to compare the values of $Y_\mathbf{X}$ to
    # $Y_\mathbf{X'}$.
    event_1_variables = {variable for variable, _ in event_1}
    event_2_variables = {variable for variable, _ in event_1}
    common_variables = event_1_variables.intersection(event_2_variables)
    common_value_dict = defaultdict(set)
    for variable, value in event_1:
        if variable in common_variables:
            common_value_dict[variable].add(value)
    for variable, value in event_2:
        if variable.get_base() in common_variables:
            common_value_dict[variable].add(value)

    return all(value in common_value_dict[variable] for variable, value in event_1) and all(
        value in common_value_dict[variable] for variable, value in event_2
    )
    # return all([list_1[v].star == list_2[v].star for v in list_1 if v in list_2])


def get_event_subset_for_designated_variables(
    event: list[tuple[Variable, Intervention]], constraint_variables: set[Variable]
) -> list[tuple[Variable, Intervention]]:
    r"""Select a subset of a set of values that correspond to designated variables.

    .. todo:: if this function isn't important, please delete

    Note that we could more elegantly represent the values as sets of interventions, but we use
    (variable, intervention) tuples to represent values instead. Because values are often
    associated with counterfactual variables for our purposes, in practice we need these tuples
    to associate a value with its corresponding counterfactual variable without losing track of the
    interventions associated with that counterfactual variable.

    :math: We denote by $\mathbf{x} \cup \mathbf{Z}$ the subset of $\mathbf{x}$ corresponding
    to variables in $\mathbf{Z}$. We assume the domain of every variable is finite.

    :param event:
        A tuple associating $\mathbf{X}$, a set of counterfactual variables (or regular variables)
        in $\mathbf{V}$ with $\mathbf{x}$, a set of values for $\mathbf{X}$. We encode the
        counterfactual variables as Variable objects, and the values as Intervention objects.
    :param constraint_variables: $\mathbf{Z}$, the set of variables in $\mathbf{V} used to constrain the
        values $\mathbf{x}$.
    :returns:
        An event containing tuples associating variables in $\mathbf{X} \cup \mathbf{Z}$
        with values in $\mathbf{x} \cup \mathbf{Z}$.
    """
    return [(variable, value) for variable, value in event if variable in constraint_variables]


def get_event_subset_excluding_designated_variables(
    event: list[tuple[Variable, Intervention]], constraint_variables: set[Variable]
) -> list[tuple[Variable, Intervention]]:
    r"""Select a subset of a set of values that do not correspond to a set of designated variables.

    .. todo:: if this function isn't important, please delete

    Note that we could more elegantly represent the values as sets of interventions, but we use
    (variable, intervention) tuples to represent values instead. Because values are often
    associated with counterfactual variables for our purposes, in practice we need these tuples
    to associate a value with its corresponding counterfactual variable without losing track of the
    interventions associated with that counterfactual variable.

    :math: We also denote by $\mathbf{x} \backslash \mathbf{Z} the value of \mathbf{X} \backslash \mathbf{Z}
    consistent with \mathbf{x}. We assume the domain of every variable is finite.

    :param event:
        A tuple associating $\mathbf{X}$, a set of counterfactual variables (or regular variables)
        in $\mathbf{V}$ with $\mathbf{x}$, a set of values for $\mathbf{X}$. We encode the
        counterfactual variables as Variable objects, and the values as Intervention objects.
    :param constraint_variables: $\mathbf{Z}$, the set of variables in $\mathbf{V} used to constrain the
        values $\mathbf{x}$.
    :returns:
        An event containing tuples associating variables in $\mathbf{X} \backslash \mathbf{Z}$
        with values in $\mathbf{x} \backslash \mathbf{Z}$.
    """
    return [(variable, value) for variable, value in event if variable not in constraint_variables]


def make_selection_diagrams(
    *, selection_nodes: dict[int, Collection[Variable]], graph: NxMixedGraph
) -> list[tuple[int, NxMixedGraph]]:
    r"""Make a selection diagram.

    .. todo:: if this function isn't important, please delete

    [correa22a]_ refer to transportability diagrams as "selection diagrams" and combine
    multiple domains into a single diagram. The input dict maps an integer corresponding to each domain
    to the set of "selection variables" for that domain. We depart from the notation in [correa22a]_
    They use $\pi$ to denote selection variables in a selection diagram, but because you could in
    theory have multiple $\pi$ variables from different domains pointing to the same node in a graph, we
    prefer to retain the notation of transportability nodes from Tikka and Karvanen 2019 ("Surrogate
    Outcomes and Transportability").

    :param selection_nodes: A mapping of integers (indexes for each domain) to the selection variables for each domain.
    :param graph: The graph containing it.
    :returns: A new graph that is the selection diagram merging the multiple domains.
    """
    # FIXME is this function actually used anywhere? It only appears in a test. Delete.
    selection_diagrams = [(0, graph)]
    selection_diagrams.extend(
        (index, create_transport_diagram(nodes_to_transport=nodes, graph=graph))
        for index, nodes in selection_nodes.items()
    )
    # Note Figure 2(b) in [correa22a]_
    return selection_diagrams


#: From [correa22a]_, Figure 2a.
figure_2a_graph = NxMixedGraph.from_edges(
    directed=[
        (Z, X),
        (Z, Y),
        (X, Y),
        (X, W),
        (W, Y),
    ],
    undirected=[(Z, X), (W, Y)],
)


class TestMakeSelectionDiagrams(unittest.TestCase):
    """Test the results of creating a list of domain selection diagrams."""

    def test_make_selection_diagrams(self):
        """Produce Figure 2(a), Figure 3(a), and Figure 3(b) of [correa22a]_.

        Note that although Correa and Bareinboim describe this set of diagrams
        in the text preceding Example 3.1
        """
        selection_nodes = {1: {Z}, 2: {W}}
        selection_diagrams = make_selection_diagrams(
            selection_nodes=selection_nodes, graph=figure_2a_graph
        )
        expected_domain_1_graph = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (Z, Y),
                (X, Y),
                (X, W),
                (W, Y),
                (
                    transport_variable(Z),
                    Z,
                ),
            ],
            undirected=[(Z, X), (W, Y)],
        )
        expected_domain_2_graph = NxMixedGraph.from_edges(
            directed=[
                (Z, X),
                (Z, Y),
                (X, Y),
                (X, W),
                (W, Y),
                (
                    transport_variable(W),
                    W,
                ),
            ],
            undirected=[(Z, X), (W, Y)],
        )
        expected_selection_diagrams = [
            (0, figure_2a_graph),
            (1, expected_domain_1_graph),
            (2, expected_domain_2_graph),
        ]
        self.assertCountEqual(selection_diagrams, expected_selection_diagrams)
