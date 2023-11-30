"""Implementation of counterfactual transportability."""

from typing import Optional, Union, Iterable, Dict, List

from y0.dsl import CounterfactualVariable, Intervention, Variable
from y0.graph import NxMixedGraph
from y0.algorithm.transport import create_transport_diagram

__all__ = [
    "simplify",
]


def simplify(
    event: list[tuple[CounterfactualVariable, Intervention]]
) -> Optional[dict[CounterfactualVariable, Intervention]]:
    """Run algorithm 1: the SIMPLIFY algorithm from Correa, Lee, and Bareinboim 2022"""
    raise NotImplementedError

def get_ancestors_of_counterfactual(
    event: tuple[CounterfactualVariable, NxMixedGraph]
) -> set(Union[CounterfactualVariable, Variable]):
    """Gets the ancestors of a counterfactual variable following Correa, Lee, and Bareinboim 2022, Definition 2.1 and Example 2.1
       and returns them as a set of variables that may include counterfactual variables."""
    raise NotImplementedError(f"Unimplemented function: get_ancestors_of_counterfactual")    

## TODO: Add expected inputs and outputs to the below three algorithms
def sigmaTR() -> None:
    """Implements the sigma-TR algorithm from Correa, Lee, and Bareinboim 2022 (Algorithm 4 in Appendix B)"""
    raise NotImplementedError(f"Unimplemented function: sigmaTR")

def ctfTR() -> None:
    """Implements the ctfTR algorithm from Correa, Lee, and Bareinboim 2022 (Algorithm 3)"""
    raise NotImplementedError(f"Unimplemented function: ctfTR")

def ctfTRu() -> None:
    """Implements the ctfTRu algorithm from Correa, Lee, and Bareinboim 2022 (Algorithm 2)"""
    raise NotImplementedError(f"Unimplemented function: ctfTRu")

def is_ctf_factor(
    *,
    event: list[tuple[CounterfactualVariable]],
    graph: NxMixedGraph
) -> bool:
    """Checks whether a joint probability distribution of counterfactual variables is a counterfactual factor ("ctf factor") in a graph.
       See Correa, Lee, and Bareinboim 2022, Defenition 3.4
    """
    raise NotImplementedError(f"Unimplemented function: is_ctf_factor")

def make_selection_diagram(
    *,
    graph: NxMixedGraph,
    selection_nodes: Dict[int, Iterable[Variable]]
) -> NxMixedGraph:
    """Correa, Lee, and Barenboim refer to transportability diagrams as "selection diagrams" and combine multiple domains into a single diagram.
       The input dict maps an integer corresponding to each domain to the set of "selection variables" for that domain. We depart from 
       Correa, Lee, and Barenboim's notation. They use pi to denote selection variables in a selection diagram, but because you could in theory
       have multiple pi variables from different domains pointing to the same node in a graph, we prefer to retain the notation of transportability
       nodes from Tikka and Karvanen 2019 ("Surrogate Outcomes and Transportability").
    """
    selection_diagrams = List[NxMixedGraph]
    for selection_variables in selection_nodes.values():
        selection_diagrams.append(create_transport_diagram(
                                    nodes_to_transport = selection_variables,
                                    graph = graph)
        )
    return _merge_transport_diagrams(selection_diagrams)

def _merge_transport_diagrams(
    *,
    graphs: List[NxMixedGraph]
) -> NxMixedGraph:
    """This implementation could be incorporated into make_selection_diagram()"""
    raise NotImplementedError(f"Unimplemented function: _merge_transport_diagrams")
