# -*- coding: utf-8 -*-

"""Implementation of counterfactual transportability from https://proceedings.mlr.press/v162/correa22a/correa22a.pdf."""

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from y0.algorithm.transport import create_transport_diagram
from y0.dsl import CounterfactualVariable, Intervention, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "simplify",
]


def simplify(
    event: List[Tuple[CounterfactualVariable, Intervention]]
) -> Optional[Dict[CounterfactualVariable, Intervention]]:
    """Run algorithm 1, the SIMPLIFY algorithm from Correa, Lee, and Bareinboim 2022.

    :param event: "Y_*, a set of counterfactual variables in V and y_* a set of
                values for Y_*." We encode the counterfactual variables as
                CounterfactualVariable objects, and the values as Intervention objects.
    :returns: "Y_* = y_*". We use a dict with counterfactual variables for keys and
              interventions for values.
    :raises NotImplementedError: not implemented yet.
    """
    # TODO: Ask Jeremy:
    # 1) Is it better to have Union[CounterfactualVariable, Variable] instead of just CounterfactualVariable?
    # 2) Is there a better way to capture the values than with Intervention objects?
    raise NotImplementedError("Unimplemented function: Simplify")
    return None


def get_ancestors_of_counterfactual(
    event: CounterfactualVariable, graph: NxMixedGraph
) -> Set[Union[CounterfactualVariable, Variable]]:
    """Get the ancestors of a counterfactual variable.

    This follows Correa, Lee, and Bareinboim 2022, Definition 2.1 and Example 2.1.

    :param event: A single counterfactual variable.
    :param graph: The graph containing it.
    :returns: a set of counterfactual variables. Correa, Lee, and Bareinboim consider
              a "counterfactual variable" to also include variables with no interventions.
              In our case we allow our returned set to include the "Variable" class for
              Y0 syntax, and should test examples including ordinary variables as
              ancestors.
    :raises NotImplementedError: not implemented yet.
    """
    raise NotImplementedError("Unimplemented function: get_ancestors_of_counterfactual")
    return None


def minimize(
    event: Set[CounterfactualVariable], graph: NxMixedGraph
) -> Set[CounterfactualVariable]:
    r"""Minimize a set of counterfactual variables.

    Source: last paragraph in Section 4 of Correa, Lee, and Barenboim 2022, before Section 4.1.
    Mathematical expression: ||\mathbf Y_*|| = {||Y_{\mathbf x}|| | Y_{\mathbf x}} \elementof \mathbf Y_*}.
    (The math syntax is not necessarily cannonical LaTeX.)

    :param event: A set of counterfactual variables to minimize.
    :param graph: The graph containing them.
    :returns: a set of minimized counterfactual variables such that each minimized variable
              is an element of the original set.
    """
    return_set = set({})
    for cv in event:
        mini_cv = _miniminimize(cv, graph)
        if mini_cv not in return_set:
            return_set.add(mini_cv)
    return return_set


def _miniminimize(event: CounterfactualVariable, graph: NxMixedGraph) -> CounterfactualVariable:
    r"""Minimize a single counterfactual variable which may have multiple interventions.

    Source: last paragraph in Section 4 of Correa, Lee, and Barenboim 2022, before Section 4.1.

    Mathematical expression:
    ||Y_{\mathbf x}|| = Y_{\mathbf t}, where \mathbf T = \mathbf X \intersect An(Y)_{G_{\overline(\mathbf X)}}}.
    (The math syntax is not necessarily cannonical LaTeX.)

    :param event: A counterfactual variable to minimize.
    :param graph: The graph containing them.
    :returns: a minimized counterfactual variable which may omit some interventions from the original one.
    :raises NotImplementedError: not implemented yet.
    """
    raise NotImplementedError("Unimplemented function: _miniminimize")
    return None


def same_district(event: Set[Union[CounterfactualVariable, Variable]], graph: NxMixedGraph) -> bool:
    """Check if a set of counterfactual variables are in the same district (c-component) of a graph.

    :param event: A set of counterfactual variables.
    :param graph: The graph containing them.
    :returns: A boolean.
    :raises NotImplementedError: not implemented yet.
    """
    # TODO: Hint to my future self: use the graph.districts() function and just get the base of
    #       each counterfactual variable. There's a function to get the district for a single variable.
    #       Get the union of the output from that function applied to each variable and see if
    #       its size is greater than one.
    raise NotImplementedError("Unimplemented function: same_district")
    return None


# TODO: Add expected inputs and outputs to the below three algorithms
def sigma_tr() -> None:
    """Implement the sigma-TR algorithm from Correa, Lee, and Bareinboim 2022 (Algorithm 4 in Appendix B)."""
    raise NotImplementedError("Unimplemented function: sigmaTR")


def ctf_tr() -> None:
    """Implement the ctfTR algorithm from Correa, Lee, and Bareinboim 2022 (Algorithm 3)."""
    raise NotImplementedError("Unimplemented function: ctfTR")


def ctf_tru() -> None:
    """Implement the ctfTRu algorithm from Correa, Lee, and Bareinboim 2022 (Algorithm 2)."""
    raise NotImplementedError("Unimplemented function: ctfTRu")


def is_ctf_factor(
    *, event: List[Union[CounterfactualVariable, Variable]], graph: NxMixedGraph
) -> bool:
    """Check if a joint probability distribution of counterfactual variables is a counterfactual factor in a graph.

    See Correa, Lee, and Bareinboim 2022, Definition 3.4. A "ctf-factor" is a counterfactual factor.

    :param event: A list of counterfactual variables, some of which may have no interventions.
    :param graph: The corresponding graph.
    :returns: A single boolean value (True if the input event is a ctf-factor, False otherwise).
    :raises NotImplementedError: not implemented yet.
    """
    raise NotImplementedError("Unimplemented function: is_ctf_factor")
    return None


def make_selection_diagram(
    *, selection_nodes: Dict[int, Iterable[Variable]], graph: NxMixedGraph
) -> NxMixedGraph:
    """Make a selection diagram.

    Correa, Lee, and Barenboim refer to transportability diagrams as "selection diagrams" and combine
    multiple domains into a single diagram. The input dict maps an integer corresponding to each domain
    to the set of "selection variables" for that domain. We depart from Correa, Lee, and Barenboim's
    notation. They use pi to denote selection variables in a selection diagram, but because you could in
    theory have multiple pi variables from different domains pointing to the same node in a graph, we
    prefer to retain the notation of transportability nodes from Tikka and Karvanen 2019 ("Surrogate
    Outcomes and Transportability").

    :param selection_nodes: A mapping of integers (indexes for each domain) to the selection variables for each domain.
    :param graph: The graph containing it.
    :returns: A new graph that is the selection diagram merging the multiple domains.
    """
    selection_diagrams = list()
    for selection_variables in selection_nodes.values():
        selection_diagrams.append(
            create_transport_diagram(nodes_to_transport=selection_variables, graph=graph)
        )
    return _merge_transport_diagrams(graphs=selection_diagrams)


def _merge_transport_diagrams(*, graphs: List[NxMixedGraph]) -> NxMixedGraph:
    """Merge transport diagrams from multiple domains into one diagram.

    This implementation could be incorporated into make_selection_diagram().

    :param graphs: A list of graphs (transport diagrams) corresponding to each domain.
    :returns: a new graph merging the domains.
    :raises NotImplementedError: not implemented yet.
    """
    raise NotImplementedError("Unimplemented function: _merge_transport_diagrams")
    return None
