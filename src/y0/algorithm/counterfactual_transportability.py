# -*- coding: utf-8 -*-

"""Implementation of counterfactual transportability.

.. [correa22a] https://proceedings.mlr.press/v162/correa22a/correa22a.pdf.
"""

import logging
from collections import defaultdict
from typing import DefaultDict, Iterable, Optional

from y0.algorithm.transport import create_transport_diagram, transport_variable
from y0.dsl import (
    CounterfactualVariable,
    Expression,
    Intervention,
    P,
    Product,
    Sum,
    Variable,
)
from y0.graph import NxMixedGraph

__all__ = [
    "simplify",
    "minimize",
    "get_ancestors_of_counterfactual",
    "same_district",
    "is_counterfactual_factor_form",
    "get_counterfactual_factors",
    "convert_to_counterfactual_factor_form",
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


def _any_variables_with_inconsistent_values(
    variable_to_value_mappings: DefaultDict[Variable, set[Intervention]]
) -> bool:
    r"""Check for variables with inconsistent values following Line 2 of Algorithm 1 in [correa_22a]_."""
    # Part 1 of Line 2:
    # :math: **if** there exists $Y_{\mathbf{x}}\in \mathbf{Y}_\ast$ with
    # two or more different values in  $\mathbf{y_\ast}$ **then return** 0.
    # Note this definition has to do with counterfactual values, and is different than
    # the "inconsistent counterfactual factor" definition in Definition 4.1 of [correa22a]_.
    if any(len(value_set) > 1 for value_set in variable_to_value_mappings.values()):
        return True

    # Part 2 of Line 2:
    # :math: **if** there exists $Y_y\in \mathbf{Y}_\ast$ with $\mathbf{y_*} \cap Y_y \neq y$ **then return** 0.
    for variable in variable_to_value_mappings.keys():
        if isinstance(variable, CounterfactualVariable):
            for intervention in variable.interventions:
                if intervention.get_base() == variable.get_base():  # Y_Y
                    if {intervention} != variable_to_value_mappings[variable]:
                        logger.warn("Part 2 of Line 2 fails: {{intervention}} = " + str({intervention}) + "and variable_to_value_mappings[variable] = " + str(variable_to_value_mappings[variable]))
                        return True
    return False


def _simplify_outcomes_with_consistent_values(
    outcome_variable_to_value_mappings: DefaultDict[Variable, set[Intervention]],
    outcome_variables: set[Variable],
) -> tuple[DefaultDict[Variable, set[Intervention]], set[Variable]]:
    r"""Address Part 2 of Line 3 of SIMPLIFY from [correa22a]_.

    :math: **if** there exists $Y_y\in \mathbf{Y}_\ast$ with $\mathbf{y_*} \cap Y_y = y$ **then**
    remove repeated variables from $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.

    Note that Y_y and Y are repeated variables. So, when the counterfactual variable Y_y and the 
    intervention Y are both in the set of outcome variables, we want to remove one of them and 
    the obvious one to remove is the more complex Y_y. What [correa22a]_ does not specify is,
    in the case where Y_y is in the set of events but Y is not, should Y_y get reduced to Y? 
    The question is analogous to asking, in the case where Y is in the set of events but Y_y is 
    not, should Y be replaced by Y_y? The latter answer is "no" because the notation becomes 
    more complex. So, our answer to the former question is "yes" because the notation 
    becomes simpler without changing the results of Algorithms 2 or 3 in [correa22a]_.

    :param outcome_variable_to_value_mappings:
        A dictionary mapping Variable objects to their values, represented as Intervention objects.
    :param outcome_variables:
        A set of outcome variables (really just the keys for outcome_variable_to_value_mappings, the
        code could be further optimized).
    :returns:
        These same two inputs with any redundant outcome variables removed.
    """
    for variable in list(outcome_variables):
        #logger.warn("Looking at variable " + str(variable))
        #logger.warn(
        #    "Is it a counterfactual variable: " + str(isinstance(variable, CounterfactualVariable))
        #)
        if isinstance(variable, CounterfactualVariable):
            for intervention in variable.interventions:
                #logger.warn(
                #    "In _simplify_outcomes_with_consistent_values: looking at intervention "
                #    + str(intervention)
                #    + " and counterfactual variable "
                #    + str(variable)
                #)
                if intervention.get_base() == variable.get_base():  # Y_Y
                    #logger.warn(
                    #    "In _simplify_outcomes_with_consistent_values:: looking at intervention "
                    #    + str(intervention)
                    #    + " and counterfactual variable "
                    #    + str(variable)
                    #    + ". Bases are equal."
                    #)
                    #logger.warn("Minimized outcome variables: " + str(outcome_variables))
                    #logger.warn("variable.get_base(): " + str(variable.get_base()))
                    #logger.warn(
                    #    "Is it in minimized_outcome_variables: "
                    #    + str(variable.get_base() in outcome_variables)
                    #)
                    if variable.get_base() in outcome_variables:
                        #logger.warn(
                        #    "list(minimized_outcome_variable_to_value_mappings[variable.get_base()])[0]: "
                        #    + str(list(outcome_variable_to_value_mappings[variable.get_base()])[0])
                        #)
                        #logger.warn(
                        #    "   which is of class "
                        #    + str(
                        #        list(outcome_variable_to_value_mappings[variable.get_base()])[
                        #            0
                        #        ].__class__
                        #    )
                        #)
                        # This case must be Y_y with :math:$\mathbf y_\ast \intersect Y_y = y$
                        #if (
                        #    list(outcome_variable_to_value_mappings[variable.get_base()])[0].to_y0()
                        #    == intervention.to_y0()
                        #):
                        #    logger.warn(
                        #        "In _simplify_outcomes_with_consistent_values: found an intervention "
                        #        + str({intervention})
                        #        + " that's the same as the counterfactual variable "
                        #        + str(variable)
                        #        + " with value "
                        #        + str(outcome_variable_to_value_mappings[variable])
                        #    )
                        #else: 
                        #    logger.warn(
                        #        "In _simplify_outcomes_with_consistent_values: found an intervention "
                        #        + str({intervention})
                        #        + " with the same base but value " + str(list(outcome_variable_to_value_mappings[variable.get_base()])[0])
                        #        + " which is different than the counterfactual variable "
                        #        + str(variable)
                        #        + " with value "
                        #        + str(outcome_variable_to_value_mappings[variable])
                        #    )
                        #    # We want to return None, but we'll do so by calling Line 2 again from simplify() after returning from here. 
                        outcome_variable_to_value_mappings[variable.get_base()].update({intervention})
                        del outcome_variable_to_value_mappings[variable]
                        outcome_variables.remove(variable)
                    else: 
                        # Convert the counterfactual variable to an intervention, add the intervention and its value to the 
                        # mappings, and delete the counterfactual variable.
                        #logger.warn(
                        #    "In _simplify_outcomes_with_consistent_values: replacing counterfactual variable "
                        #    + str(variable)
                        #    + " with a pure variable: "
                        #    + str(variable.get_base())
                        #    + " with value "
                        #    + str(outcome_variable_to_value_mappings[variable])
                        #    + " and original outcome_variable_to_value_mappings = " + str(outcome_variable_to_value_mappings)
                        #    + " and outcome_variables = " + str(outcome_variables)
                        #)
                        outcome_variable_to_value_mappings[variable.get_base()].update(outcome_variable_to_value_mappings[variable])
                        outcome_variables.add(variable.get_base())
                        del outcome_variable_to_value_mappings[variable]
                        outcome_variables.remove(variable)
                        #logger.warn("New outcome_variable_to_value_mappings = " + str(outcome_variable_to_value_mappings)
                        #            + " and outcome_variables = " + str(outcome_variables))
    return (outcome_variable_to_value_mappings, outcome_variables)


def simplify(
    *, event: list[tuple[Variable, Intervention]], graph: NxMixedGraph
) -> Optional[list[tuple[Variable, Intervention]]]:
    r"""Run algorithm 1, the SIMPLIFY algorithm from [correa22a]_.

    Correa, Lee, and Bareinboim [correa22a]_ state that this algorithm should return "an interventionally
    minimal event :math:$\mathbf Y_* = \mathbf y_*$ without redundant subscripts or 0 if the counterfactual
    event is guaranteed to have probability 0." In Y0, Zero() is an expression object. Here, we return
    None instead of 0 to denote an event that is impossible, and rely on the calling function to translate either
    the returned event or a None object into an appropriate probabilistic expression.

    This implementation switches the order of Lines 2 and 3 in Algorithm 1 of [correa22a]_. The reason
    is that Line 3 reduces (Y @ -Y, -Y) to (Y, -Y). The substitution could introduce the inconsistency
    [(Y, -Y), (Y, +Y)] in the case where the input to SIMPLIFY is [(Y @ -Y, -Y), (Y, +Y)].

    :param event:
        "Y_*, a set of counterfactual variables in V and y_* a set of
        values for Y_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param graph:
        The associated graph.
    :raises TypeError: Improperly formatted inputs for simplify()
    :returns:
        "Y_* = y_*". We use a dict with counterfactual variables for keys and
        interventions for values.
    """
    # TODO: Ask Jeremy:
    # 1) Is it better to have Union[CounterfactualVariable, Variable] instead of just CounterfactualVariable?
    #  answer: no, variable is the superclass so just annotate with Variable
    # 2) Is there a better way to capture the values than with Intervention objects?
    #
    #  answer(JZ): "Hmm...things are a little broken, because what we would like to express is
    #  `P(Y @ -y == -y)`, but instead, all we have is `P(Y @ -y)`, or `P(-y)`."
    #  If we want to know the value of a counterfactual variable, then we should use Intervention.
    #  Variable(name="Y", star="False") shouldn't exist. Variables should set star to None, so that
    #  it is only interpreted as a Variable.  If you want to talk about the value of a variable, then you
    #  can make a tuple: (Variable(name="Y", star=None), Intervention(name="Y", star=False))
    #
    #  RJC: That's why below we test that a Variable that is not a CounterfactualVariable has
    #       set star to None. Putting the check here means that we don't need separate checks
    #       in functions such as get_counterfactual_ancestors(), because the ctfTRu and ctfTR
    #       algorithms all call SIMPLIFY early in their processing.
    for element in event:
        if (
            len(element) != 2
            or not isinstance(element[0], Variable)
            or not isinstance(element[1], Intervention)
        ):
            raise TypeError(
                f"Improperly formatted inputs for simplify(): check input event element {element}"
            )
        if (
            isinstance(element[0], Variable)
            and not isinstance(element[0], CounterfactualVariable)
            and element[0].star is not None
        ):
            raise TypeError(
                f"Improperly formatted inputs for simplify(): {element[0]} should have "
                "a star value of None because it is a Variable"
            )

    outcome_variables = {element[0] for element in event}

    # Some of the entries in our dict won't be necessary.
    logger.warn("In simplify: outcome_variables = " + str(outcome_variables))
    minimized_outcome_variables: set[Variable] = minimize(variables=outcome_variables, graph=graph)
    logger.warn("In simplify: minimized_outcome_variables = " + str(minimized_outcome_variables))

    # Creating this dict addresses part 1 of Line 3:
    # :math: **if** there exists $Y_{\mathbf{x}}\in \mathbf{Y}_\ast$ with
    # two consistent values in  $\mathbf{y_\ast} \cap Y_x$ **then**
    # remove repeated variables from $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.
    minimized_outcome_variable_to_value_mappings: DefaultDict[
        Variable, set[Intervention]
    ] = defaultdict(set)
    for element in event:
        if element[0] in minimized_outcome_variables:
            minimized_outcome_variable_to_value_mappings[element[0]].add(element[1])

    logger.warn("In simplify after part 1 of line 3: outcome_variables = " + str(minimized_outcome_variables))
    logger.warn("                                    minimize_outcome_variable_to_value_mappings = " + str(minimized_outcome_variable_to_value_mappings))

    # Line 2 of SIMPLIFY.
    if _any_variables_with_inconsistent_values(minimized_outcome_variable_to_value_mappings):
        return None

    logger.warn("In simplify after line 2: outcome_variables = " + str(minimized_outcome_variables))
    logger.warn("                          minimize_outcome_variable_to_value_mappings = " + str(minimized_outcome_variable_to_value_mappings))

    # Part 2 of Line 3:
    # :math: **if** there exists $Y_y\in \mathbf{Y}_\ast$ with $\mathbf{y_*} \cap Y_y = y$ **then**
    # remove repeated variables from $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.
    (
        minimized_outcome_variable_to_value_mappings,
        minimized_outcome_variables,
    ) = _simplify_outcomes_with_consistent_values(
        minimized_outcome_variable_to_value_mappings, minimized_outcome_variables
    )

    # Call line 2 of SIMPLIFY again to handle counterfactual variables such as (Y @ -Y, -Y) that Line 3 reduces to 
    # interventions inconsistent with existing variables such as (Y, +Y)
    if _any_variables_with_inconsistent_values(minimized_outcome_variable_to_value_mappings):
        return None

    logger.warn(
        "In simplify before return: minimized_outcome_variables = "
        + str(minimized_outcome_variables)
    )
    logger.warn(
        "In simplify before return: minimized_outcome_variable_to_value_mappings = "
        + str(minimized_outcome_variable_to_value_mappings)
    )
    result = [
        (key, minimized_outcome_variable_to_value_mappings[key].pop())
        for key in minimized_outcome_variable_to_value_mappings
    ]
    logger.warn("In simplify before return: return value = " + str(result))
    return result


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


def minimize(*, variables: Iterable[Variable], graph: NxMixedGraph) -> set[Variable]:
    r"""Minimize a set of counterfactual variables.

    Source: last paragraph in Section 4 of [correa22a]_, before Section 4.1.
    $||\mathbf Y_*|| = {||Y_{\mathbf x}|| | Y_{\mathbf x}} \in {\mathbf Y_*}$.

    :param variables: A set of counterfactual variables to minimize (some may have no interventions).
    :param graph: The graph containing them.
    :returns:
        A set of minimized counterfactual variables such that each minimized variable
        is an element of the original set.
    """
    return {_do_minimize(variable, graph) for variable in variables}


def _do_minimize(variable: Variable, graph: NxMixedGraph) -> Variable:
    r"""Minimize a single variable which is usually counterfactual and may have multiple interventions.

    Source: last paragraph in Section 4 of [correa22a]_, before Section 4.1.

    $||Y_{\mathbf x}|| = Y_{\mathbf t}, where \mathbf T = \mathbf X \intersect An(Y)_{G_{\overline(\mathbf X)}}$
    and $\mathbf t = \mathbf x \intersect \mathbf T$.

    :param variable: A counterfactual variable to minimize (which may have no interventions).
    :param graph: The graph containing them.
    :returns: a minimized counterfactual variable which may omit some interventions from the original one.
    """
    if isinstance(variable, CounterfactualVariable):
        # :math: $\mathbf x$
        interventions: tuple = variable.interventions
        # :math: $\mathbf X$
        intervention_variables: set[Variable] = {
            intervention.get_base() for intervention in interventions
        }
        # :math: $\mathbf T$
        treatment_variables = (
            graph.remove_in_edges(intervention_variables)
            .ancestors_inclusive(variable.get_base())
            .intersection(intervention_variables)
        )
        # :math: $\mathbf t$
        treatment_interventions: tuple[Intervention] = tuple(
            {
                intervention
                for intervention in sorted(interventions)
                # for intervention in interventions
                if intervention.get_base() in treatment_variables
            }
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
            interventions=tuple(sorted(treatment_interventions)),
        )
    else:
        return variable


def same_district(event: set[Variable], graph: NxMixedGraph) -> bool:
    """Check if a set of counterfactual variables are in the same district (c-component) of a graph.

    Edge cases: return True if the event contains one or no events.

    :param event: A set of counterfactual variables.
    :param graph: The graph containing them.
    :returns: A boolean.
    """
    # TODO: Hint to my future self: use the graph.districts() function and just get the base of
    #       each counterfactual variable. There's a function to get the district for a single variable.
    #       Get the union of the output from that function applied to each variable and see if
    #       its size is greater than one.
    if len(event) < 1:
        return True

    visited_districts: set[frozenset] = {
        graph.get_district(variable.get_base()) for variable in event
    }
    logger.warn("In same_district(): event = " + str(event))
    logger.warn("Visited districts: " + str(visited_districts))
    # raise NotImplementedError("Unimplemented function: same_district")
    return len(visited_districts) == 1


def is_counterfactual_factor_form(*, event: set[Variable], graph: NxMixedGraph) -> bool:
    """Check if a set of counterfactual variables is a counterfactual factor in a graph.

    See [correa22a]_, Definition 3.4. A "ctf-factor" is a counterfactual factor.

    For a counterfactual variable to be in counterfactual factor form, all of its parents
    must be in the intervention set and the variable itself cannot be (because Y_y = y,
    so we want to apply the counterfactual factor form of y by itself). If the variable
    is not a counterfactual variable, then it must have no parents to be in counterfactual
    factor form.

    :param event: A set of counterfactual variables, some of which may have no interventions.
    :param graph: The corresponding graph.
    :returns: A single boolean value (True if the input event is a ctf-factor, False otherwise).
    """
    for variable in event:
        parents = list(graph.directed.predecessors(variable.get_base()))
        if isinstance(variable, CounterfactualVariable):
            if any(
                variable.get_base().name == intervention.name
                for intervention in variable.interventions
            ):
                return False
            for parent in parents:
                if not any(
                    parent.name == intervention.name for intervention in variable.interventions
                ):
                    return False
        else:
            if len(parents) > 0:
                return False
    return True


def get_counterfactual_factors(*, event: set[Variable], graph: NxMixedGraph) -> list[set[Variable]]:
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
    :raises KeyError: An event is not in counterfactual factor form.
    :returns:
        A set of sorted lists of counterfactual variables in counterfactual factor
        form, with each list associated with a district of the graph. The lists
        need not contain every variable in the district, but they can't contain
        variables missing from the graph.
    """
    if not is_counterfactual_factor_form(event=event, graph=graph):
        logger.warn("Supposed to trigger KeyError in get_counterfactual_factors().")
        raise KeyError(
            "In get_counterfactual_factors(): the event %s is not in counterfactual factor form.",
            str(event),
        )

    district_mappings: DefaultDict[frozenset[Variable], set[Variable]] = defaultdict(set)
    for variable in event:
        district_mappings[graph.get_district(variable.get_base())].add(variable)

    # TODO if there aren't duplicates, this can be a set of frozensets
    return_value = [set(value) for value in district_mappings.values()]
    return return_value


def convert_to_counterfactual_factor_form(
    *, event: list[tuple[Variable, Intervention]], graph: NxMixedGraph
) -> list[tuple[Variable, Intervention]]:
    r"""Convert a list of (possibly counterfactual) variables and their values to counterfactual factor ("ctf-factor") form.

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
    return [
        (variable.intervene(graph.directed.predecessors(variable.get_base())), value)
        if len(list(graph.directed.predecessors(variable.get_base()))) > 0
        else (variable, value)
        for (variable, value) in event
    ]


def do_counterfactual_factor_factorization(
    *, variables: list[tuple[Variable, Intervention]], graph: NxMixedGraph
) -> Expression:
    r"""Take an arbitrary query and return its counterfactual factor form, factorized according to the graph c-components.

    :param variables:
        A list of counterfactual variables (the left side of Equation 11 in [correa22a]_).

        :math:P*( \mathbf y_*) (i.e., a joint probability distribution corresponding to a query).
    :param graph: The corresponding graph.
    :raises TypeError: do_counterfactual_factorization() requires at least one variable in the query variable set.
    :returns:
        An expression following the right side of Equation 15 in [correa22a]_ (example: Equation 16).

        :math:Sum_{ \mathbf d_* \backslash \mathbf y_*} P*( \mathbf d_* ), where :math:\mathbf D_* = An( \mathbf Y_* ),
        and where P*( \mathbf d_* ) has been further decomposed as per
        :math: P*( \mathbf d_* ) = prod_{j}(P*( \mathbf c_{j*}) (Equation 15).
    """
    # We can't directly compute the ancestral set via a set comprehension because get_ancestors_of_counterfactual()
    # returns mutable sets, so we'd get an 'unhashable type: set' error
    # note from @cthoyt - use frozenset if you want immutable/hashable sets
    #
    # TODO: Do we need to check for consistency among the elements of the ancestral set before
    #       applying the union operation?
    #
    # TODO: Check that we get variables only, and not values as well, from calling
    # get_ancestors_of_counterfactual() with these inputs. Because we want the ancestral set.
    if not variables:
        raise TypeError(
            "do_counterfactual_factorization() requires at least one variable in the query."
        )

    ancestral_set: set[Variable] = set()
    for counterfactual_variable, value in variables:
        ancestral_set.update(get_ancestors_of_counterfactual(counterfactual_variable, graph))

    #  e.g., Equation 14 in [correa22a]_, without the summation component.
    ancestral_set_in_counterfactual_factor_form: list[
        tuple[Variable, Intervention]
    ] = convert_to_counterfactual_factor_form(event=ancestral_set, graph=graph)

    # P*(d_*). It's a counterfactual variable hint, so a distribution can be constructed from it.
    ancestral_set_variable_names: set[Variable] = {
        variable.get_base() for variable in ancestral_set_in_counterfactual_factor_form
    }

    outcome_variable_names: set[Variable] = {variable.get_base() for variable in variables}

    # Decompose the query by c-component (e.g., Equation 16 in [correa22a]_)
    ancestral_set_subgraph = graph.subgraph(ancestral_set_variable_names)
    factorized_ancestral_set: list[set[Variable]] = get_counterfactual_factors(
        event=ancestral_set_in_counterfactual_factor_form, graph=ancestral_set_subgraph
    )

    # Question for JZ / @cthoyt: The below works, but is ugly. Mypy won't allow me to
    #           initialize result_expression to 'None' because that messes with
    #           result_expression's expected type later in the function. I tried initializing
    #           the expression to 1 with 'result_expression = One()'. That returned
    #           'error: Incompatible types in assignment (expression has type "Expression",
    #           variable has type "One")' from mypy. (One() is a subclass of Expression.)
    #           What's a better way to initialize an 'empty' expression?
    result_expression = Product.safe(P(factor) for factor in factorized_ancestral_set)

    # The summation portion of Equation 11 in [correa22a]_
    sum_range = ancestral_set_variable_names - outcome_variable_names

    result_expression = Sum.safe(result_expression, sum_range)
    return result_expression


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
        transport_variable(factor.get_base()) in domain_graph.nodes() for factor in factors
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
