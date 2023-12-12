# -*- coding: utf-8 -*-

"""Implementation of counterfactual transportability.

.. [correa22a] https://proceedings.mlr.press/v162/correa22a/correa22a.pdf.
"""

import logging
from typing import Iterable, Optional

from y0.algorithm.transport import create_transport_diagram, transport_variable
from y0.dsl import CounterfactualVariable, Expression, Intervention, P, Sum, Variable
from y0.graph import NxMixedGraph

__all__ = [
    "simplify",
    "minimize",
    "get_ancestors_of_counterfactual",
    "same_district",
    "is_counterfactual_factor_form",
    "get_counterfactual_factors",
    "convert_to_counterfactual_factor_form",
    "get_counterfactual_factor_query",
    "do_counterfactual_factor_factorization",
    "make_selection_diagram",
    "counterfactual_factors_are_transportable",
    "sigma_tr",
    "ctf_tr",
    "ctf_tru",
    # TODO add functions/classes/variables you want to appear in the docs and be exposed to the user in this list
    #  Run tox -e docs then `open docs/build/html/index.html` to see docs
]

logger = logging.getLogger(__name__)
# logging.basicConfig(filename="/home/callahanr/Documents/Causality/y0/y0.log")


def simplify(
    event: list[tuple[CounterfactualVariable, Intervention]]
) -> Optional[dict[CounterfactualVariable, Intervention]]:
    """Run algorithm 1, the SIMPLIFY algorithm from [correa22a]_.

    :param event:
        "Y_*, a set of counterfactual variables in V and y_* a set of
        values for Y_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :returns:
        "Y_* = y_*". We use a dict with counterfactual variables for keys and
        interventions for values.
    """
    # TODO: Ask Jeremy:
    # 1) Is it better to have Union[CounterfactualVariable, Variable] instead of just CounterfactualVariable?
    #  answer: no, variable is the superclass so just annotate with Variable
    # 2) Is there a better way to capture the values than with Intervention objects?
    raise NotImplementedError("Unimplemented function: Simplify")


def get_ancestors_of_counterfactual(
    event: CounterfactualVariable, graph: NxMixedGraph
) -> set[Variable]:
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
    # This is the set of variables X in [correa22a]_, Definition 2.1.
    if isinstance(event, CounterfactualVariable):
        interventions = {intervention.get_base() for intervention in event.interventions}

        graph_minus_in = graph.remove_in_edges(interventions)
        ancestors = graph.remove_out_edges(interventions).ancestors_inclusive(event.get_base())

        ancestors_of_counterfactual_variable: set[Variable] = set()
        for ancestor in ancestors:
            candidate_interventions_z = graph_minus_in.ancestors_inclusive(ancestor).intersection(
                interventions
            )
            # TODO: graph_minus_in.ancestors_inclusive(candidate_ancestor) returns variables.
            # intervention_variables are Interventions, which are a type of Variable.
            # Will these sets intersect without throwing errors?
            if candidate_interventions_z:
                ancestors_of_counterfactual_variable.add(
                    ancestor.intervene(candidate_interventions_z)
                )
            else:
                ancestors_of_counterfactual_variable.add(ancestor)
        return ancestors_of_counterfactual_variable
    else:
        # There's a TypeError check here because it is easy for a user to pass a set of variables in, instead of
        # a single variable.
        if not isinstance(event, Variable):
            raise TypeError(
                "This function requires a variable, usually a counterfactual variable, as input."
            )
        return graph.ancestors_inclusive(event)


def minimize(
    event: Iterable[CounterfactualVariable], graph: NxMixedGraph
) -> set[CounterfactualVariable]:
    r"""Minimize a set of counterfactual variables.

    Source: last paragraph in Section 4 of [correa22a]_, before Section 4.1.
    $||\mathbf Y_*|| = {||Y_{\mathbf x}|| | Y_{\mathbf x}} \in {\mathbf Y_*}$.

    :param event: A set of counterfactual variables to minimize.
    :param graph: The graph containing them.
    :returns:
        a set of minimized counterfactual variables such that each minimized variable
        is an element of the original set.
    """
    return_set = set()
    for cv in event:
        mini_cv = _miniminimize(cv, graph)
        if mini_cv not in return_set:
            return_set.add(mini_cv)
    return return_set


def _miniminimize(event: CounterfactualVariable, graph: NxMixedGraph) -> CounterfactualVariable:
    r"""Minimize a single counterfactual variable which may have multiple interventions.

    Source: last paragraph in Section 4 of [correa22a]_, before Section 4.1.

    $||Y_{\mathbf x}|| = Y_{\mathbf t}, where \mathbf T = \mathbf X \intersect An(Y)_{G_{\overline(\mathbf X)}}$.

    :param event: A counterfactual variable to minimize.
    :param graph: The graph containing them.
    :returns: a minimized counterfactual variable which may omit some interventions from the original one.
    """
    raise NotImplementedError("Unimplemented function: _miniminimize")


def same_district(event: set[Variable], graph: NxMixedGraph) -> bool:
    """Check if a set of counterfactual variables are in the same district (c-component) of a graph.

    :param event: A set of counterfactual variables.
    :param graph: The graph containing them.
    :returns: A boolean.
    """
    # TODO: Hint to my future self: use the graph.districts() function and just get the base of
    #       each counterfactual variable. There's a function to get the district for a single variable.
    #       Get the union of the output from that function applied to each variable and see if
    #       its size is greater than one.
    raise NotImplementedError("Unimplemented function: same_district")


def is_counterfactual_factor_form(*, event: set[Variable], graph: NxMixedGraph) -> bool:
    """Check if a set of counterfactual variables is a counterfactual factor in a graph.

    See [correa22a]_, Definition 3.4. A "ctf-factor" is a counterfactual factor.

    For a counterfactual variable to be a counterfactual factor, all of its parents must
    be in the intervention set. If the variable is not a counterfactual variable, then
    it must have no parents to be in counterfactual factor form.

    :param event: A set of counterfactual variables, some of which may have no interventions.
    :param graph: The corresponding graph.
    :returns: A single boolean value (True if the input event is a ctf-factor, False otherwise).
    """
    for variable in event:
        parents = list(graph.directed.predecessors(variable.get_base()))
        if isinstance(variable, CounterfactualVariable):
            for parent in parents:
                if not any(
                    [parent.name == intervention.name for intervention in variable.interventions]
                ):
                    return False
        else:
            if len(parents) > 0:
                return False
    return True


def get_counterfactual_factors(
    *, event: set[Variable], graph: NxMixedGraph
) -> Optional[list[set[Variable]]]:
    """Decompose a joint probability distribution of counterfactual variables.

    Rather than work with probability distributions directly, the function
    takes in a set of counterfactual variables and returns a set of
    sets that correspond to factors associated with individual districts
    (c-components) in the graph. The function returns "None" if any of the
    counterfactual variables are not in counterfactual factor form, or
    if an event variable is not in any district (i.e., not in the graph).

    See [correa22a]_, Definition 3.4. A "ctf-factor" is a counterfactual factor.

    :param event:
        A set of counterfactual variables, some of which may
        have no interventions. All counterfactual variable must be in counterfactual
        factor form.
    :param graph: The corresponding graph.
    :returns:
        A set of sorted lists of counterfactual variables in counterfactual factor
        form, with each list associated with a district of the graph. The lists
        need not contain every variable in the district, but they can't contain
        variables missing from the graph.
    """
    districts = list(graph.districts())
    district_mappings: dict[frozenset, set[Variable]] = {district: set() for district in districts}

    if any(
        [not any([variable.get_base() in district for district in districts]) for variable in event]
    ):
        logger.warn(
            "In get_counterfactual_factors(): a variable in the input event is missing from the graph.\n"
        )
        return None

    if not is_counterfactual_factor_form(event=event, graph=graph):
        logger.warn(
            "In get_counterfactual_factors(): the event (%s) is not in counterfactual factor form.\n",
            str(event),
        )
        return None

    for variable in event:
        for district in districts:
            if variable.get_base() in district:
                district_mappings[district].update({variable})
                break
    return_value = [set(value) for value in district_mappings.values()]
    return return_value


def convert_to_counterfactual_factor_form(
    *, event: set[Variable], graph: NxMixedGraph
) -> set[Variable]:
    r"""Convert a set of variables (which may be counterfactual variables) to counterfactual factor ("ctf-factor") form.

    That requires intervening on all the parents of each counterfactual variable.

    :param event: A list of variables (which may have interventions) and their values (:math:`\mathbf W_\ast`).
    :param graph: The corresponding graph.
    :returns:
        The output above, represented as a set of counterfactual variables (those without interventions
        are just variables).

        :math:`w_{1[\mathbf{pa_{1}}]},w_{2[\mathbf{pa_{2}}]},\cdots,w_{l[\mathbf{pa_{l}}]}`
        for each :math:`W_i \in \mathbf V`.
    """
    # result: set[Variable] = set()
    # for variable in event:
    #    parents = list(graph.directed.predecessors(variable.get_base()))
    #    if len(parents) > 0:
    #        result.update({variable.intervene(parents)})
    #    else:
    #        result.update({variable})
    # return result
    # (Here's the more efficient set comprehension with harder-to-read code)
    return {
        variable.intervene(graph.directed.predecessors(variable.get_base()))
        if len(list(graph.directed.predecessors(variable.get_base()))) > 0
        else variable
        for variable in event
    }


def get_counterfactual_factor_query(
    *, event: list[CounterfactualVariable], graph: NxMixedGraph
) -> Expression:
    r"""Take an arbitrary query and return the counterfactual factor form of the query ("ctf-factor form").

    :param event: A list of values of counterfactual variables. In the paper,  :math:`P^\ast( \mathbf y_\ast)`
    :param graph: The corresponding graph.
    :returns:
        An expression following the right side of Equation 11 in [correa22a]_.

        :math:`\sum_{ \mathbf d_\ast \backslash \mathbf y_\ast} P^\ast( \mathbf d_\ast )`,
            where :math:`\mathbf D_\ast = An( \mathbf Y_\ast )`
    """
    # We can't directly compute the ancestral set via a set comprehension because get_ancestors_of_counterfactual()
    # returns mutable sets, so we'd get an 'unhashable type: set' error
    # note from @cthoyt - use frozenset if you want immutable/hashable sets
    # TODO: Do we need to check for consistency among the elements of the ancestral set before
    #       applying the union operation?
    ancestral_set: set[Variable] = set()
    for counterfactual_variable in event:
        ancestral_set.update(get_ancestors_of_counterfactual(counterfactual_variable, graph))
        # TODO: Check that we get variables only, and not values as well, from calling
        # get_ancestors_of_counterfactual() with these inputs. Because we want the ancestral set.

    # We want the elements of the ancestral set that do not contain elements of the original query, y.
    # This is :math:`\mathbf y_\ast`. For getting the variables to marginalize over, i.e. getting V(W_*) from W_*.
    # variable_names_of_outcome_values: set[Variable] = {counterfactual_variable.get_base() for
    #     counterfactual_variable in event}
    # variable_names_of_ancestral_set: set[Variable] = {counterfactual_variable.get_base() for
    #     counterfactual_variable in ancestral_set}

    #  e.g., Equation 14 in [correa22a]_, without the summation component.
    ancestral_set_values_in_counterfactual_factor_form: set[
        Variable
    ] = convert_to_counterfactual_factor_form(event=ancestral_set, graph=graph)
    # ancestral_set_query_in_counterfactual_factor_form = P(ancestral_set_values_in_counterfactual_factor_form)

    # P*(d_*). It's a counterfactual variable hint, so a distribution can be constructed from it.
    variable_names_of_ancestral_set_in_counterfactual_factor_form: set[Variable] = {
        counterfactual_variable.get_base()
        for counterfactual_variable in ancestral_set_values_in_counterfactual_factor_form
    }

    variable_names_of_outcome_values: set[Variable] = {
        counterfactual_variable.get_base() for counterfactual_variable in event
    }

    # For decomposing this into the factors by c-component, in another function, shortly:
    # ancestral_set_subgraph = graph.subgraph(variable_names_of_ancestral_set_in_counterfactual_factor_form)
    # factorized_ancestral_set_query =
    #     factorize_query(ancestral_set_values_in_counterfactual_factor_form, ancestral_set_subgraph)

    # ancestral_set_subgraph_districts = ancestral_set_subgraph.districts()

    # for ancestor in ancestral_set:
    #    ctf_cv = ancestor.intervene(graph.directed.predecessors(ancestor))
    #    query_in_counterfactual_factor_form.add(ctf_cv)
    #    query_with_just_the_variable_names.add(ctf_cv.get_base())
    sum_range = (
        variable_names_of_ancestral_set_in_counterfactual_factor_form
        - variable_names_of_outcome_values
    )
    result = Sum.safe(P(ancestral_set_values_in_counterfactual_factor_form), sum_range)
    return result


def do_counterfactual_factor_factorization(
    *, event: list[Variable], graph: NxMixedGraph
) -> Expression:
    r"""Take an arbitrary query and return its counterfactual factor form, factorized according to the graph c-components.

    :param event:
        A list of counterfactual variables (the left side of Equation 11 in [correa22a]_).

        :math:P*( \mathbf y_*) (i.e., a joint probability distribution corresponding to a query).
    :param graph: The corresponding graph.
    :returns:
        An expression following the right side of Equation 15 in [correa22a]_ (example: Equation 16).

        :math:Sum_{ \mathbf d_* \\ \mathbf y_*} P*( \mathbf d_* ), where :math:\mathbf D_* = An( \mathbf Y_* ),
        where P*( \mathbf d_* ) has been further decomposed as per
        :math: P*( \mathbf d_* ) = prod_{j}(P*( \mathbf c_{j*}) (Equation 15).
    """
    raise NotImplementedError("Unimplemented function: get_counterfactual_factors")


def make_selection_diagram(
    *, selection_nodes: dict[int, Iterable[Variable]], graph: NxMixedGraph
) -> NxMixedGraph:
    r"""Make a selection diagram.

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
    selection_diagrams = [
        create_transport_diagram(nodes_to_transport=selection_variables, graph=graph)
        for selection_variables in selection_nodes.values()
    ]
    return _merge_transport_diagrams(graphs=selection_diagrams)


def _merge_transport_diagrams(*, graphs: list[NxMixedGraph]) -> NxMixedGraph:
    """Merge transport diagrams from multiple domains into one diagram.

    This implementation could be incorporated into make_selection_diagram().

    :param graphs: A list of graphs (transport diagrams) corresponding to each domain.
    :returns: A new graph merging the domains.
    """
    raise NotImplementedError("Unimplemented function: _merge_transport_diagrams")


def counterfactual_factors_are_transportable(
    *, factors: set[Variable], domain_graph: NxMixedGraph
) -> bool:
    """Determine if a set of counterfactual factors can be transported from a domain to the target.

    :param domain_graph: Corresponds to the domain from which we're doing the transporting.
    :param factors: The counterfactual factors in question.
    :returns: Whether the query is transportable.
    """
    return not any(
        [transport_variable(factor.get_base()) in domain_graph.nodes() for factor in factors]
    )


# TODO: Add expected inputs and outputs to the below three algorithms
def sigma_tr() -> None:
    """Implement the sigma-TR algorithm from [correa22a]_ (Algorithm 4 in Appendix B)."""
    raise NotImplementedError("Unimplemented function: sigmaTR")


def ctf_tr() -> None:
    """Implement the ctfTR algorithm from [correa22a]_ (Algorithm 3)."""
    raise NotImplementedError("Unimplemented function: ctfTR")


def ctf_tru() -> None:
    """Implement the ctfTRu algorithm from [correa22a]_ (Algorithm 2)."""
    raise NotImplementedError("Unimplemented function: ctfTRu")
