# -*- coding: utf-8 -*-

"""Implementation of counterfactual transportability.

.. [huang08a] https://link.springer.com/article/10.1007/s10472-008-9101-x.
.. [correa20a] https://proceedings.neurips.cc/paper/2020/file/7b497aa1b2a83ec63d1777a88676b0c2-Paper.pdf.
.. [correa22a] https://proceedings.mlr.press/v162/correa22a/correa22a.pdf.
.. [tikka20a] https://github.com/santikka/causaleffect/blob/master/R/compute.c.factor.R.
.. [tikka20b] https://github.com/santikka/causaleffect/blob/master/R/identify.R.
.. [tian03a] https://ftp.cs.ucla.edu/pub/stat_ser/R290-L.pdf
"""

import logging
from collections import defaultdict
from typing import Collection, DefaultDict, Iterable, Optional

from y0.algorithm.tian_id import _compute_c_factor, identify_variables_in_district
from y0.algorithm.transport import create_transport_diagram, transport_variable
from y0.dsl import (
    CounterfactualVariable,
    Expression,
    Intervention,
    P,
    Product,
    Sum,
    Variable,
    Zero,
)
from y0.graph import NxMixedGraph

__all__ = [
    "simplify",
    "minimize",
    "minimize_event",
    "get_ancestors_of_counterfactual",
    "same_district",
    "is_counterfactual_factor_form",
    "get_counterfactual_factors",
    "convert_to_counterfactual_factor_form",
    "do_counterfactual_factor_factorization",
    "make_selection_diagram",
    "counterfactual_factors_are_transportable",
    "transport_district_intervening_on_parents",
    "transport_unconditional_counterfactual_query",
    "ctf_tr",
    # TODO add functions/classes/variables you want to appear in the docs and be exposed to the user in this list
    #  Run tox -e docs then `open docs/build/html/index.html` to see docs
]

logger = logging.getLogger(__name__)


def _any_variables_with_inconsistent_values(
    # variable_to_value_mappings: DefaultDict[Variable, set[Intervention]]
    *,
    nonreflexive_variable_to_value_mappings: DefaultDict[Variable, set[Intervention]],
    reflexive_variable_to_value_mappings: DefaultDict[Variable, set[Intervention]],
) -> bool:
    r"""Check for variables with inconsistent values following Line 2 of Algorithm 1 in [correa_22a]_."""
    # Part 1 of Line 2:
    # :math: **if** there exists $Y_{\mathbf{x}}\in \mathbf{Y}_\ast$ with
    # two or more different values in  $\mathbf{y_\ast}$ **then return** 0.
    # Note this definition has to do with counterfactual values, and is different than
    # the "inconsistent counterfactual factor" definition in Definition 4.1 of [correa22a]_.
    if any(len(value_set) > 1 for value_set in nonreflexive_variable_to_value_mappings.values()):
        return True

    # Part 2 of Line 2:
    # :math: **if** there exists $Y_y\in \mathbf{Y}_\ast$ with $\mathbf{y_*} \cap Y_y \neq y$ **then return** 0.
    return any(
        (
            not isinstance(variable, CounterfactualVariable)
            and len(reflexive_variable_to_value_mappings[variable]) > 1
        )
        or (
            isinstance(variable, CounterfactualVariable)
            and any(
                {intervention} != reflexive_variable_to_value_mappings[variable]
                for intervention in variable.interventions
            )
        )
        for variable in reflexive_variable_to_value_mappings.keys()
    )
    # Longer version of the above.
    # for variable in reflexive_variable_to_value_mappings.keys():  # Y_y, Y
    #    if not isinstance(variable, CounterfactualVariable):  # Y
    #        # TODO: Check with JZ that it's intended that $Y_y$ and $Y$ are the same.
    #        #       I infer that is so because of Equation 4 in [correa22a]_.
    #        # If Y takes on at least two values as part of the same query, then there exists
    #        # $Y_{y}\in \mathbf{Y}_\ast$ with two or more different values in  $\mathbf{y_\ast}$.
    #        # That implies that there exists $Y_y\in \mathbf{Y}_\ast$ with $\mathbf{y_*} \cap Y_y \neq y$,
    #        # so we return 0.
    #        if len(reflexive_variable_to_value_mappings[variable]) > 1:
    #            logger.warning("Part 2 of Line 2 fails for (non-counterfactual) variables: ")
    #            logger.warning(
    #                "    Variable = "
    #                + str(variable)
    #                + ", values are "
    #                + str(reflexive_variable_to_value_mappings[variable])
    #            )
    #            return True
    #    else:  # Y_y
    #        for intervention in variable.interventions:
    #            if intervention.get_base() != variable.get_base():
    #                raise TypeError(
    #                    f"In _any_variables_with_inconsistent_values: reflexive variable {str(variable)} \
    #                      has an intervention that is not itself: {str(intervention)}."
    #                )
    #            # reflexive_variable_to_value_mappings[variable] = $\mathbf{y_*} \cap Y_y
    #            # {intervention} = y
    #            if {intervention} != reflexive_variable_to_value_mappings[variable]:
    #                logger.warning(
    #                    "Part 2 of Line 2 fails: {{intervention}} = "
    #                    + str({intervention})
    #                    + " and reflexive_variable_to_value_mappings[variable] = "
    #                    + str(reflexive_variable_to_value_mappings[variable])
    #                )
    #                return True
    # return False


# Deprecated.
# def _simplify_self_interventions_with_consistent_values(
#    outcome_variable_to_value_mappings: DefaultDict[Variable, set[Intervention]],
#    outcome_variables: set[Variable],
# ) -> tuple[DefaultDict[Variable, set[Intervention]], set[Variable]]:
#    r"""Address Part 2 of Line 3 of SIMPLIFY from [correa22a]_.
#
#    :math: **if** there exists $Y_y\in \mathbf{Y}_\ast$ with $\mathbf{y_*} \cap Y_y = y$ **then**
#    remove repeated variables from $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.
#
#    Note that Y_y and Y are repeated variables. So, when the counterfactual variable Y_y and the
#    intervention Y are both in the set of outcome variables, we want to remove one of them and
#    the obvious one to remove is the more complex Y_y. What [correa22a]_ does not specify is,
#    in the case where Y_y is in the set of events but Y is not, should Y_y get reduced to Y?
#    The question is analogous to asking, in the case where Y is in the set of events but Y_y is
#    not, should Y be replaced by Y_y? The latter answer is "no" because the notation becomes
#    more complex. So, our answer to the former question is "yes" because the notation
#    becomes simpler without changing the results of Algorithms 2 or 3 in [correa22a]_.
#
#    :param outcome_variable_to_value_mappings:
#        A dictionary mapping Variable objects to their values, represented as Intervention objects.
#    :param outcome_variables:
#        A set of outcome variables (really just the keys for outcome_variable_to_value_mappings, the
#        code could be further optimized).
#    :returns:
#        These same two inputs with any redundant outcome variables removed.
#    """
#    for variable in list(outcome_variables):  # if isinstance(variable, CounterfactualVariable):
#        if not isinstance(variable, CounterfactualVariable):
#            continue
#        for intervention in variable.interventions:
#            if intervention.get_base() != variable.get_base():
#                continue
#            # Y_Y
#            if variable.get_base() in outcome_variables:
#                outcome_variable_to_value_mappings[variable.get_base()].update({intervention})
#                del outcome_variable_to_value_mappings[variable]
#                outcome_variables.remove(variable)
#            else:
#                outcome_variable_to_value_mappings[variable.get_base()].update(
#                    outcome_variable_to_value_mappings[variable]
#                )
#                outcome_variables.add(variable.get_base())
#                del outcome_variable_to_value_mappings[variable]
#                outcome_variables.remove(variable)
#    return outcome_variable_to_value_mappings, outcome_variables


def get_event_subset_for_designated_variables(
    event: list[tuple[Variable, Intervention]], constraint_variables: set[Variable]
) -> list[tuple[Variable, Intervention]]:
    r"""Select a subset of a set of values that correspond to designated variables.

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


# def is_consistent(list_1: dict[Variable,Intervention], list_2: dict[Variable,Intervention]):
def is_consistent(
    event_1: list[tuple[Variable, Intervention]], event_2: list[tuple[Variable, Intervention]]
) -> bool:
    r"""Check whether two lists of values are consistent.

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


def _remove_repeated_variables_and_values(
    *, event: list[tuple[Variable, Intervention]]
) -> defaultdict[Variable, set[Intervention]]:
    r"""Implement the first half of Line 3 of the SIMPLIFY algorithm from [correa22a]_.

    The implementation is as simple as creating a dictionary. Adding variables to
    the dictionary removes repeated variables in the input event, and adding values to
    the dictionary using the variables as keys removes repeated values.

    :math: **if** there exists $Y_{\mathbf{x}}\in \mathbf{Y}_\ast$ with
    two consistent values in  $\mathbf{y_\ast} \cap Y_x$ **then**
    remove repeated variables from $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.

    :param event:
        A tuple associating $\mathbf{Y_\ast}$, a set of counterfactual variables (or regular variables)
        in $\mathbf{V}$ with $\mathbf{y_\ast}$, a set of values for $\mathbf{Y_\ast}$. We encode the
        counterfactual variables as Variable objects, and the values as Intervention objects.
    :returns:
        A dictionary mapping the event variables to all values associated with each variable in the event.
    """
    variable_to_value_mappings: DefaultDict[Variable, set[Intervention]] = defaultdict(set)
    for variable, intervention in event:
        variable_to_value_mappings[variable].add(intervention)
    return variable_to_value_mappings


def _split_event_by_reflexivity(
    event: list[tuple[Variable, Intervention]]
) -> tuple[list[tuple[Variable, Intervention]], list[tuple[Variable, Intervention]]]:
    r"""Categorize variables in an event by reflexivity (i.e., whether they intervene on themselves).

    :param event:
        "Y_*, a set of counterfactual variables in V and y_* a set of
        values for Y_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :returns:
        Two events, one containing variables in :math: $Y_{\mathbf{x}} \in \mathbf{Y}_\ast$}
        and one containing variables in :math: $Y_{y} \in \mathbf{Y}_\ast$.
        Note that only if minimization has already taken place (which is the case here),
        variables that are not counterfactual variables are considered the equivalent
        of :math: $Y_{\mathbf{y} \in \mathbf{Y}_\ast$ and fall into the latter category.
    """
    # Y_y
    reflexive_interventions_event: list[tuple[Variable, Intervention]] = [
        (variable, value)
        for variable, value in event
        if (
            isinstance(variable, CounterfactualVariable)
            and any(
                intervention.get_base() == variable.get_base()
                for intervention in variable.interventions
            )
        )
        or not (isinstance(variable, CounterfactualVariable))
    ]
    # Y_x
    nonreflexive_interventions_event: list[tuple[Variable, Intervention]] = [
        (variable, value)
        for variable, value in event
        if isinstance(variable, CounterfactualVariable)
        and not any(
            intervention.get_base() == variable.get_base()
            for intervention in variable.interventions
        )
    ]
    # for variable, value in event:
    #    if isinstance(variable, CounterfactualVariable):
    #        if any(
    #            [
    #                intervention.get_base() == variable.get_base()
    #                for intervention in variable.interventions
    #            ]
    #        ):
    #            reflexive_interventions_event.append((variable, value))
    #        else:
    #            nonreflexive_interventions_event.append((variable, value))
    #    else:
    #        # A variable with no intervention, Y, is the same thing as Y_y in the event that
    #        # minimization has already taken place (which is the case here)
    #        reflexive_interventions_event.append((variable, value))
    return reflexive_interventions_event, nonreflexive_interventions_event


def _reduce_reflexive_counterfactual_variables_to_interventions(
    variables: defaultdict[Variable, set[Intervention]]
) -> defaultdict[Variable, set[Intervention]]:
    r"""Simplify counterfactual variables intervening on themselves to Intervention objects with the same base.

    :param variables: A defaultdict mapping :math: $\mathbf{Y_\ast}$, a set of counterfactual variables in
        $\mathbf{V}$, to $\mathbf{y_\ast}$, a set of values for $\mathbf{Y_\ast}$. Each variable in
        $\mathbf{Y_\ast}$ is assumed to be either $Y_{y}$ \in $\mathbf{Y_\ast}$ or just $Y$ in $\mathbf{Y_\ast}$,
        where $Y$ is considered a special case of $Y_{y}$ because minimization has already taken place.
        The $\mathbf{Y_\ast}$ variables are CounterfactualVariable objects, and the values as Intervention objects.
    :raises TypeError: a variable in the input dictionary has more than one intervention or its intervention is
        not itself.
    :returns:
        A defaultdict mapping simple variables :math: $\mathbf{Y}$ to $\mathbf{y}$, a set of corresponding values.
    """
    result_dict: DefaultDict[Variable, set[Intervention]] = defaultdict(set)
    for variable in variables:
        if not isinstance(variable, CounterfactualVariable):
            result_dict[variable].update(variables[variable])
        else:
            if len(variable.interventions) != 1:
                raise TypeError(
                    "In _reduce_reflexive_counterfactual_variables_to_interventions: all variables in the \
                            input dictionary should have exactly one intervention, but this one has more than one: "
                    + str(variable)
                )
            if any(
                intervention.get_base() != variable.get_base()
                for intervention in variable.interventions
            ):
                raise TypeError(
                    "In _reduce_reflexive_counterfactual_variables_to_interventions: variable "
                    + str(variable)
                    + " has an intervention that is not itself: "
                    + str(variable.interventions)
                )
            result_dict[variable.get_base()].update(variables[variable])
    return result_dict


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
    if any([len(tup) != 2 for tup in event]):
        raise TypeError(
            "Improperly formatted inputs for simplify(): an event element is a tuple with length not equal to 2."
        )
    for variable, intervention in event:
        if not isinstance(variable, Variable) or not isinstance(intervention, Intervention):
            raise TypeError(
                "Improperly formatted inputs for simplify(): check input event element ("
                + str(variable)
                + ", "
                + str(intervention)
                + ")"
            )
        if (
            isinstance(variable, Variable)
            and not isinstance(variable, CounterfactualVariable)
            and variable.star is not None
        ):
            raise TypeError(
                f"Improperly formatted inputs for simplify(): {variable} should have "
                "a star value of None because it is a Variable"
            )

    # It's not enough to minimize the variables, we need to keep track of what values are associated with
    # the minimized variables. So we minimize the event.
    # minimized_variables: set[Variable] = minimize(variables={variable for variable, _ in event}, graph=graph)
    minimized_event: list[tuple[Variable, Intervention]] = minimize_event(event=event, graph=graph)
    # logger.warning("In simplify: minimized_event = " + str(minimized_event))

    # Split the query into Y_x variables and Y_y ("reflexive") variables
    (
        reflexive_interventions_event,
        nonreflexive_interventions_event,
    ) = _split_event_by_reflexivity(minimized_event)

    # Creating this dict addresses part 1 of Line 3:
    # :math: If there exists $Y_{\mathbf{X}} \in \mathbf{Y_\ast}$ with two consistent values in
    # $\mathbf{y_\ast} \cap Y_{\mathbf{X}}$ then remove repeated variables from
    # $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.
    minimized_nonreflexive_variable_to_value_mappings = _remove_repeated_variables_and_values(
        event=nonreflexive_interventions_event
    )
    # Creating this dict partly addresses part 2 of Line 3:
    # :math: If there exists $Y_{y} \in \mathbf{Y_\ast}$ with
    # $\mathbf{y_\ast} \cap Y_{y} = y$ then remove repeated variables from
    # $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.
    #
    # There is an exception: we don't yet handle the edge case that the CounterfactualVariable Y_y
    # and the Intervention Y, when observed as part of the same event, are considered repeated
    # variables after minimization has taken place.
    minimized_reflexive_variable_to_value_mappings = _remove_repeated_variables_and_values(
        event=reflexive_interventions_event
    )

    # logger.warning(
    #    "In simplify after part 1 of line 3: minimized_nonreflexive_variable_to_value_mappings = "
    #    + str(minimized_nonreflexive_variable_to_value_mappings)
    # )
    # logger.warning(
    #    "                                    minimized_reflexive_variable_to_value_mappings = "
    #    + str(minimized_reflexive_variable_to_value_mappings)
    # )

    # Line 2 of SIMPLIFY.
    if _any_variables_with_inconsistent_values(
        nonreflexive_variable_to_value_mappings=minimized_nonreflexive_variable_to_value_mappings,
        reflexive_variable_to_value_mappings=minimized_reflexive_variable_to_value_mappings,
    ):
        return None

    # logger.warning(
    #    "In simplify after line 2: minimized_nonreflexive_variable_to_value_mappings = "
    #    + str(minimized_nonreflexive_variable_to_value_mappings)
    # )
    # logger.warning(
    #    "                                    minimized_reflexive_variable_to_value_mappings = "
    #    + str(minimized_reflexive_variable_to_value_mappings)
    # )

    # Now we're able to reduce the reflexive counterfactual variables to interventions.
    # This simultaneously addresses Part 2 of Line 3:
    # :math: **if** there exists $Y_y\in \mathbf{Y}_\ast$ with $\mathbf{y_*} \cap Y_y = y$ **then**
    # remove repeated variables from $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.
    minimized_reflexive_variable_to_value_mappings = (
        _reduce_reflexive_counterfactual_variables_to_interventions(
            minimized_reflexive_variable_to_value_mappings
        )
    )

    # logger.warning(
    #    "In simplify after part 2 of line 3: minimized_reflexive_variable_to_value_mappings = "
    #    + str(minimized_reflexive_variable_to_value_mappings)
    # )

    # (Original part 2 of Line 3):
    # :math: **if** there exists $Y_y\in \mathbf{Y}_\ast$ with $\mathbf{y_*} \cap Y_y = y$ **then**
    # remove repeated variables from $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.
    # (
    #    minimized_variable_to_value_mappings,
    #    minimized_variables,
    # ) = _simplify_self_interventions_with_consistent_values(
    #    minimized_variable_to_value_mappings, minimized_variables
    # )

    # Call line 2 of SIMPLIFY again to handle counterfactual variables such as (Y @ -Y, -Y) that Line 3 reduces to
    # interventions inconsistent with existing variables such as (Y, +Y). This handles an
    # edge case for part 2 of Line 3:
    # :math: **if** there exists $Y_y\in \mathbf{Y}_\ast$ with $\mathbf{y_*} \cap Y_y = y$ **then**
    # remove repeated variables from $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.
    if _any_variables_with_inconsistent_values(
        nonreflexive_variable_to_value_mappings=minimized_nonreflexive_variable_to_value_mappings,
        reflexive_variable_to_value_mappings=minimized_reflexive_variable_to_value_mappings,
    ):
        return None

    # logger.warning(
    #    "In simplify after line 2: minimized_nonreflexive_variable_to_value_mappings = "
    #    + str(minimized_nonreflexive_variable_to_value_mappings)
    # )
    # logger.warning(
    #    "                                    minimized_reflexive_variable_to_value_mappings = "
    #    + str(minimized_reflexive_variable_to_value_mappings)
    # )

    simplified_event = [
        (key, minimized_nonreflexive_variable_to_value_mappings[key].pop())
        for key in minimized_nonreflexive_variable_to_value_mappings
    ] + [
        (key, minimized_reflexive_variable_to_value_mappings[key].pop())
        for key in minimized_reflexive_variable_to_value_mappings
    ]
    logger.warning("In simplify before return: return value = " + str(simplified_event))
    return simplified_event


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

    # logger.warning("In get_ancestors_of_counterfactual: input = " + str(event))
    if not isinstance(event, CounterfactualVariable):
        return graph.ancestors_inclusive(event)

    # This is the set of variables X in [correa22a]_, Definition 2.1.
    intervention_variables = {intervention.get_base() for intervention in event.interventions}
    intervention_values = {intervention for intervention in event.interventions}

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
    # logger.warning(
    #    "In get_ancestors_of_counterfactual: output = " + str(ancestors_of_counterfactual_variable)
    # )
    return ancestors_of_counterfactual_variable


# Deprecated.
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


def minimize_event(
    *, event: list[tuple[Variable, Intervention]], graph: NxMixedGraph
) -> list[tuple[Variable, Intervention]]:
    r"""Minimize a set of counterfactual variables wrapped into an event.

    Source: last paragraph in Section 4 of [correa22a]_, before Section 4.1.
    $||\mathbf Y_*|| = {||Y_{\mathbf x}|| | Y_{\mathbf x}} \in {\mathbf Y_*}$.

    :param event: A set of counterfactual variables to minimize (some may have no interventions),
                  along with their associated values.
    :param graph: The graph containing them.
    :returns:
        An event comprised of a set of minimized counterfactual variables such that each minimized variable
        is an element of the original set, and the values associated with those variables.
    """
    return [(_do_minimize(variable, graph), value) for variable, value in event]


def _do_minimize(variable: Variable, graph: NxMixedGraph) -> Variable:
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

    # :math: $\mathbf x$
    interventions = variable.interventions
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
    treatment_interventions: frozenset[Intervention] = frozenset(
        {
            intervention
            for intervention in interventions
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
        interventions=treatment_interventions,
    )


def same_district(event: set[Variable], graph: NxMixedGraph) -> bool:
    """Check if a set of counterfactual variables are in the same district (c-component) of a graph.

    Edge cases: return True if the event contains one or no events.

    :param event: A set of counterfactual variables.
    :param graph: The graph containing them.
    :returns: A boolean.
    """
    if len(event) < 1:
        return True

    visited_districts: set[frozenset] = {
        graph.get_district(variable.get_base()) for variable in event
    }
    logger.warning("In same_district(): event = " + str(event))
    logger.warning("Visited districts: " + str(visited_districts))
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
        have no interventions. All counterfactual variables must be in counterfactual
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
        logger.warning("Supposed to trigger KeyError in get_counterfactual_factors().")
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


def get_counterfactual_factors_retaining_variable_values(
    *, event: set[tuple[Variable, Intervention | None]], graph: NxMixedGraph
) -> list[set[tuple[Variable, Intervention | None]]]:
    """Decompose a joint probability distribution of counterfactual variables.

    Rather than work with probability distributions directly, the function
    takes in a set of counterfactual variables and returns a set of
    sets that correspond to factors associated with individual districts
    (c-components) in the graph. The function returns "None" if any of the
    counterfactual variables are not in counterfactual factor form, or
    if an event variable is not in any district (i.e., not in the graph).
    This variation of the function keeps the values of the counterfactual
    variables bound to the variables in the form of tuples (necessary for
    Algorithm 2 of [correa22a]_).

    See [correa22a]_, Definition 3.4. A "ctf-factor" is a counterfactual factor.

    :param event:
        A set of counterfactual variables, some of which may
        have no interventions. All counterfactual variables must be in counterfactual
        factor form.
    :param graph: The corresponding graph.
    :raises KeyError: An event is not in counterfactual factor form.
    :returns:
        A set of sorted lists of counterfactual variables in counterfactual factor
        form along with the values for those variables, with each list associated
        with a district of the graph. The lists need not contain every variable in
        the district, but they can't contain variables missing from the graph.
    """
    event_without_values = {variable for variable, _ in event}

    if not is_counterfactual_factor_form(event=event_without_values, graph=graph):
        logger.warning(
            "Supposed to trigger KeyError in get_counterfactual_factors_retaining_variable_values()."
        )
        raise KeyError(
            "In get_counterfactual_factors_retaining_variable_values(): the event %s is not"
            + " in counterfactual factor form.",
            str(event),
        )

    district_mappings: DefaultDict[
        frozenset[Variable], set[tuple[Variable, Intervention | None]]
    ] = defaultdict(set)
    for variable, value in event:
        district_mappings[graph.get_district(variable.get_base())].add((variable, value))
    logger.warning(
        "In get_counterfactual_factors_retaining_variable_values(): district_mappings = "
        + str(district_mappings)
    )

    # TODO if there aren't duplicates, this can be a set of frozensets
    return_value = [set(value) for value in district_mappings.values()]
    logger.warning(
        "In get_counterfactual_factors_retaining_variable_values(): returning " + str(return_value)
    )
    return return_value


def convert_to_counterfactual_factor_form(
    *, event: list[tuple[Variable, Intervention | None]], graph: NxMixedGraph
) -> list[tuple[Variable, Intervention | None]]:
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
    result: list[tuple[Variable, Intervention | None]] = []
    for variable, value in event:
        candidate_parents = set(graph.directed.predecessors(variable.get_base()))
        if isinstance(variable, CounterfactualVariable):
            parents = {
                intervention
                for intervention in variable.interventions
                if intervention.get_base() in candidate_parents
            }
        else:
            parents = set()
        parent_names = {parent.get_base() for parent in parents}
        parents.update(
            {
                candidate_parent
                for candidate_parent in candidate_parents
                if candidate_parent.get_base() not in parent_names
            }
        )
        if len(parents) > 0:
            result += [(variable.get_base().intervene(parents), value)]
        else:
            result += [(variable.get_base(), value)]
    return result


def do_counterfactual_factor_factorization(
    *, variables: list[tuple[Variable, Intervention]], graph: NxMixedGraph
) -> tuple[Expression, list[tuple[Variable, Intervention]]]:
    r"""Take an arbitrary query and return its counterfactual factor form, factorized according to the graph c-components.

    :param variables:
        A list of counterfactual variables (the left side of Equation 11 in [correa22a]_).

        :math:P*( \mathbf y_*) (i.e., a joint probability distribution corresponding to a query).
    :param graph: The corresponding graph.
    :raises TypeError: do_counterfactual_factorization() requires at least one variable in the query variable set.
    :returns:
        An expression following the right side of Equation 15 in [correa22a]_ (example: Equation 16).
        Additionally, the function returns a tuple matching variables to interventions, in order to specify
        the values of those variables in the returned expression that are not indexes of the summation term
        (i.e. the expression doesn't marginalize over those variables).

        :math:Sum_{ \mathbf d_* \backslash \mathbf y_*} P*( \mathbf d_* ), where :math:\mathbf D_* = An( \mathbf Y_* ),
        and where P*( \mathbf d_* ) has been further decomposed as per
        :math: P*( \mathbf d_* ) = prod_{j}(P*( \mathbf c_{j*}) (Equation 15).
    """
    # We can't directly compute the ancestral set via a set comprehension because get_ancestors_of_counterfactual()
    # returns mutable sets, so we'd get an 'unhashable type: set' error
    # note from @cthoyt - use frozenset if you want immutable/hashable sets
    #                     (Thanks!)
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

    result_event: list[tuple[Variable, Intervention]] = [
        (convert_to_counterfactual_factor_form(event=[(variable, value)], graph=graph)[0][0], value)
        for variable, value in variables
    ]

    ancestral_set: set[Variable] = set()
    for counterfactual_variable, _ in variables:
        ancestral_set.update(get_ancestors_of_counterfactual(counterfactual_variable, graph))

    #  e.g., Equation 14 in [correa22a]_, without the summation component.
    ancestral_set_in_counterfactual_factor_form: set[Variable] = {
        convert_to_counterfactual_factor_form(event=[(variable, None)], graph=graph)[0][0]
        for variable in ancestral_set
    }

    # P*(d_*). It's a counterfactual variable hint, so a distribution can be constructed from it.
    ancestral_set_variable_names: set[Variable] = {
        variable.get_base() for variable in ancestral_set_in_counterfactual_factor_form
    }

    outcome_variable_bases: set[Variable] = {variable.get_base() for (variable, _) in variables}

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
    sum_range = ancestral_set_variable_names - outcome_variable_bases

    result_expression = Sum.safe(result_expression, sum_range)
    return result_expression, result_event


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


def _remove_transportability_vertices(*, vertices: Collection[Variable]) -> set[Variable]:
    """Remove the transportability nodes from a set of vertices.

    :param vertices: The input vertices.
    :returns: The input vertices, without the transportability nodes.
    """
    return {v for v in vertices if not v.name.startswith("T_")}


def _validate_sigma_tr_inputs(
    *,
    district: Collection[Variable],
    domain_graphs: list[tuple[NxMixedGraph, list[Variable]]],
    domain_data: list[tuple[Collection[Variable], Expression]],
) -> None:
    r"""Conduct pre-processing checks for the sigma-TR algorithm from [correa22a]_ (Algorithm 4 in Appendix B).

    :param district: the C-component $\mathbf{C}\_{i}$ under analysis.
    :param domain_graphs: A set of $K$ tuples, one for each of the $K$ domains. Each tuple
           contains a selection diagram for that domain. In particular the graph contains
           transportability nodes for every vertex that is distributed differently in the
           domain in question than in the target domain (e.g., Vertex Z in Figure 3(a)
           in [correa22a]_), and it is a causal diagram such that its edges represent
           the state of the graph after a regime corresponding to domain $k$ has been
           applied (e.g., policy $\sigma_{X}$ in Figure 4 of [correa22a]_). The second
           element of the tuple is a topologically sorted list of all the vertices in
           the corresponding graph that are not transportability nodes. (Nodes that
           have no parents come first in such lists.)
    :param domain_data: Corresponding to $\mathcal{Z}$ in [correa22a]_, this is a set of
           $K$ tuples, one for each of the $K$ domains. Each tuple contains a set of
           variables corresponding to $\sigma_{\mathbf{Z}_{k}}$ and an expression
           denoting the probability distribution
           $P^{k}(\mathbf{V};\sigma_{\mathbf{Z}\_{j}})|{\mathbf{Z}_{j}} \in \mathcal{Z}^{i}$.
    :raises TypeError: the input arguments are in an improper format or not internally consistent.
    :raises KeyError: a variable in an input argument is missing from another input argument
           and should be there.
    """
    # Preliminary checks, starting with type checking
    if not (isinstance(district, Collection) and all(isinstance(v, Variable) for v in district)):
        raise TypeError("In sigma_tr: the input district must be a Collection of Variable objects.")
    if not (isinstance(domain_graphs, list) and all(isinstance(t, tuple) for t in domain_graphs)):
        raise TypeError("In sigma_tr: the input domain graphs must be a list of tuples.")
    if not all(
        isinstance(g, NxMixedGraph)
        and isinstance(l, list)
        and all(isinstance(v, Variable) for v in l)
        for g, l in domain_graphs
    ):
        raise TypeError(
            "In sigma_tr: the input domain graph tuples must all contain NxMixedGraph objects and lists of variables."
        )
    if not (isinstance(domain_data, list) and all(isinstance(t, tuple) for t in domain_data)):
        raise TypeError("In sigma_tr: the input domain data must be a list of tuples.")
    # TODO: Consider how to handle cases where a probability distribution is One() or Zero()
    if not all(
        isinstance(sigma_z, Collection)
        and all(isinstance(v, Variable) for v in sigma_z)
        and isinstance(e, Expression)
        for sigma_z, e in domain_data
    ):
        raise TypeError(
            "In sigma_tr: the input domain data tuples must all contain Collections of Variable objects "
            + "(first element) and Expressions (second element)."
        )
    # Check we have no empty lists
    if len(domain_graphs) == 0 or len(domain_data) == 0:
        raise TypeError(
            "In sigma_tr: empty list for either domain_graphs or domain_data. "
            + "Check your inputs."
        )
    if len(district) == 0:
        raise TypeError("In sigma_tr: the input district cannot be an empty set.")
    if any(len(g.nodes()) == 0 for g, _ in domain_graphs):
        raise TypeError(
            "In sigma_tr: at least one input domain graph contained no nodes. Check your inputs."
        )
    if any(len(topo) == 0 for _, topo in domain_graphs):
        raise TypeError(
            "In sigma_tr: an input set of topologically sorted vertices was empty. Check your inputs."
        )
    if len(domain_graphs) != len(domain_data):
        raise TypeError(
            "In sigma_tr: the length of the domain_graphs and domain_data " + "must be the same."
        )
    # Technically the topologically sorted vertices could be for the graph $G$ containing $G_{\mathbf{C}_{i}}$,
    # but we currently have a stricter requirement that they are for $G_{\mathbf{C}_{i}}$. That requirement
    # could be relaxed if it becomes a computational burden in the ctf_TRu algorithm.
    for k in range(len(domain_graphs)):
        logger.warning("k = " + str(k))
        topo_vertices = frozenset(domain_graphs[k][1])
        expression_vertices = frozenset(domain_data[k][1].get_variables())
        graph_vertices = frozenset(domain_graphs[k][0].nodes())
        graph_vertices_without_transportability_nodes = frozenset(
            _remove_transportability_vertices(vertices=graph_vertices)
        )
        policy_vertices = frozenset(domain_data[k][0])
        if topo_vertices != graph_vertices:
            raise KeyError(
                "In sigma_tr: the vertices in each domain graph must match those in the "
                + "corresponding topologically sorted list of vertices. Check your inputs. "
                + "Graph vertices: "
                + str(graph_vertices)
                + ". List: "
                + str(topo_vertices)
                + "."
            )
        # It's possible for the probability distribution to contain vertices not in the graph
        # due to conditioning on vertices outside the c-component associated with this graph.
        # The other way around is not possible, though.
        if not all(v in expression_vertices for v in graph_vertices_without_transportability_nodes):
            raise KeyError(
                "In sigma_tr: some of the vertices in a domain graph do not appear in the expression"
                + " for the probability of the graph. Check your inputs. Graph vertices: "
                + str(graph_vertices_without_transportability_nodes)
                + ". Expression vertices: "
                + str(expression_vertices)
                + "."
            )
        if not all(v in graph_vertices_without_transportability_nodes for v in policy_vertices):
            raise KeyError(
                "In sigma_tr: the set of vertices for which a policy has been applied for one "
                + "of the domains contains at least one vertex not in the domain graph. Check your inputs. "
                + "Policy vertices: "
                + str(policy_vertices)
                + ". Graph vertices: "
                + str(graph_vertices_without_transportability_nodes)
            )
        if not all(v in graph_vertices_without_transportability_nodes for v in district):
            for v in district:
                if v not in graph_vertices_without_transportability_nodes:
                    raise KeyError(
                        "In sigma_tr: one of the variables in the input district "
                        + "is not in a domain graph. District: "
                        + str(district)
                        + ". Node missing from the graph: "
                        + str(v)
                        + ". Nodes in the graph: "
                        + str(graph_vertices_without_transportability_nodes)
                        + "."
                    )
    return


def _no_intervention_variables_in_domain(
    *, district: Collection[Variable], interventions: Collection[Variable]
):
    r"""Check that a district in a graph contains no intervention veriables.

    Helper function for the sigma-TR algorithm from [correa22a]_ (Algorithm 4 in Appendix B).
    :param district: the C-component $\mathbf{C}\_{i}$ under analysis.
    :param interventions: Corresponding to $\mathcal{Z}$ in [correa22a]_, this is a set of
           variables corresponding to $\sigma_{\mathbf{Z}_{k}}$.
    :returns: true or false.
    """
    return len(set(district).intersection(interventions)) == 0


def _no_transportability_nodes_in_domain(
    *, district: Collection[Variable], domain_graph: NxMixedGraph
):
    r"""Check that a district in a graph contains no transportability nodes.

    Helper function for the transport_district_intervening_on_parents algorithm from
    [correa22a]_ (Algorithm 4 in Appendix B).
    :param district: the C-component $\mathbf{C}\_{i}$ under analysis.
    :param domain_graph: a selection diagram for the domain in question.
           The graph contains a transportability node for every vertex distributed differently
           in the domain in question than in the target domain (e.g., Vertex Z in Figure 3(a)
           in [correa22a]_), and it is a causal diagram such that its edges represent
           the state of the graph after a regime corresponding to domain $k$ has been
           applied (e.g., policy $\sigma_{X}$ in Figure 4 of [correa22a]_).
    :returns: true or false.
    """
    return not any(transport_variable(v) in domain_graph.nodes() for v in district)


def transport_district_intervening_on_parents(
    *,
    district: Collection[Variable],
    domain_graphs: list[tuple[NxMixedGraph, list[Variable]]],
    domain_data: list[tuple[Collection[Variable], Expression]],
) -> Expression | None:
    r"""Implement the sigma-TR algorithm from [correa22a]_ (Algorithm 4 in Appendix B).

    :param district: the C-component $\mathbf{C}\_{i}$ under analysis.
    :param domain_graphs: A set of $K$ tuples, one for each of the $K$ domains. Each tuple
           contains a selection diagram for that domain. In particular the graph contains
           transportability nodes for every vertex that is distributed differently in the
           domain in question than in the target domain (e.g., Vertex Z in Figure 3(a)
           in [correa22a]_), and it is a causal diagram such that its edges represent
           the state of the graph after a regime corresponding to domain $k$ has been
           applied (e.g., policy $\sigma_{X}$ in Figure 4 of [correa22a]_). The second
           element of the tuple is a topologically sorted list of all the vertices in
           the corresponding graph that are not transportability nodes. (Nodes that
           have no parents come first in such lists.)
    :param domain_data: Corresponding to $\mathcal{Z}$ in [correa22a]_, this is a set of
           $K$ tuples, one for each of the $K$ domains. Each tuple contains a set of
           variables corresponding to $\sigma_{\mathbf{Z}_{k}}$ and an expression
           denoting the probability distribution
           $P^{k}(\mathbf{V};\sigma_{\mathbf{Z}\_{j}})|{\mathbf{Z}_{j}} \in \mathcal{Z}^{i}$.
    :raises TypeError: the vertices in the input district are part of more than one district
           in a domain graph.
    :returns: A probabilistic expression for $P^{\ast}_{Pa(\mathbf{C})_{i}}(\mathbf{C}\_i)$ if
           it is transportable, or None if it is not transportable.
    """
    # Note that we currently don't require the user to input $\mathcal{G}^{\ast}$, the target
    # graph, and therefore can't verify that the input district is in fact a district of the
    # target graph as part of validating the user input.
    _validate_sigma_tr_inputs(
        district=district, domain_graphs=domain_graphs, domain_data=domain_data
    )
    # Line 1
    for k in range(len(domain_graphs)):
        # Also Line 1 (the published pseudocode could break the for loop and this test into two lines)
        if _no_intervention_variables_in_domain(
            district=district, interventions=domain_data[k][0]
        ) and _no_transportability_nodes_in_domain(
            district=district, domain_graph=domain_graphs[k][0]
        ):
            logger.warning("In sigma_tr: domain = " + str(k))
            domain_graph = domain_graphs[k][0]
            domain_graph_variables = _remove_transportability_vertices(
                vertices=domain_graph.nodes()
            )
            domain_topo = domain_graphs[k][1]

            # Line 2
            super_district = frozenset().union(
                *[domain_graph.get_district(v) for v in district]
            )  # $B_{i}$
            # Sanity check: confirm that $C_{i} \subseteq B_{i}$
            if any(super_district != domain_graph.get_district(v) for v in district):
                raise TypeError(
                    "Error in transport_district_intervening_on_parents: the vertices in an input district "
                    + "are part of more than one district in a domain graph. Input district: "
                    + str(district)
                    + ". "
                    + "Domain index: "
                    + str(k)
                    + "."
                )

            # Line 3
            logger.warning("Subgraph_probability: " + domain_data[k][1].to_latex())
            super_district_q_probability = _compute_c_factor(
                district=district,
                subgraph_variables=domain_graph_variables,
                subgraph_probability=domain_data[k][1],
                graph_topo=domain_topo,
            )
            logger.warning(
                "super_district_q_probability: " + super_district_q_probability.to_latex()
            )
            # Line 4
            district_q_probability = identify_variables_in_district(
                input_variables=frozenset(district),
                input_district=super_district,
                district_probability=super_district_q_probability,
                graph=domain_graph,
                topo=domain_topo,
            )
            # Incorporate the domain index in the representation for Q

            # Lines 5-7
            if district_q_probability is not None:
                logger.warning("Returning from sigma_tr: " + district_q_probability.to_latex())
                return district_q_probability
    # Line 9
    return None


def _transport_unconditional_counterfactual_query_line_2(
    event: list[tuple[Variable, Intervention]], graph: NxMixedGraph
) -> tuple[
    set[tuple[Variable, Intervention | None]], list[set[tuple[Variable, Intervention | None]]]
]:
    r"""Implement the ctfTRu algorithm from [correa22a]_ (Algorithm 2).

    :param event:
        "Y_*, a set of counterfactual variables in V and y_* a set of
        values for Y_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param graph: The graph corresponding to the target domain for the query.
    :returns: a tuple containing:
        1. An expression for $W_{\ast} = An(\mathbf{Y_{\ast}})$, which is an event
              because some variables in $W_{\ast}$ may have values and others may not.
              We need to keep the known values coupled to their associated variables.
        2. A list of ctf-factors corresponding to $W_{\ast}$, and values for those
              variables as well when available.
    """
    ancestral_set: set[Variable] = set()
    # $W_{\ast}$
    for variable, _ in event:
        logger.warning(
            "Updating ancestral set for variable "
            + str(variable)
            + " with "
            + str(get_ancestors_of_counterfactual(variable, graph))
        )
        ancestral_set.update(get_ancestors_of_counterfactual(variable, graph))
    # logger.warning(
    #    "In transport_unconditional_counterfactual_query_line_2: "
    #    + "ancestral_set = "
    #    + str(ancestral_set)
    # )
    outcome_value_dict = {variable: value for variable, value in event}
    ancestral_set_with_values: set[tuple[Variable, Intervention | None]] = {
        (variable, outcome_value_dict[variable])
        if variable in outcome_value_dict
        else (variable, None)
        for variable in ancestral_set
    }
    # logger.warning(
    #    "In transport_unconditional_counterfactual_query_line_2: "
    #    + "ancestral_set_with_values = "
    #    + str(ancestral_set_with_values)
    # )

    #  e.g., Equation 13 in [correa22a]_, without the summation component.
    # ancestral_set_in_counterfactual_factor_form_with_values: set[
    #    tuple[Variable, Intervention | None]
    # ] = {
    #    (convert_to_counterfactual_factor_form(event=[(variable, value)], graph=graph)[0][0], value)
    #    for variable, value in ancestral_set_with_values
    # }
    ancestral_set_in_counterfactual_factor_form_with_values_as_list: list[
        tuple[Variable, Intervention | None]
    ] = [
        (convert_to_counterfactual_factor_form(event=[(variable, value)], graph=graph)[0][0], value)
        for variable, value in ancestral_set_with_values
    ]
    # logger.warning(
    #    "In transport_unconditional_counterfactual_query_line_2: "
    #    + "ancestral_set_in_counterfactual_factor_form_with_values_as_list = "
    #    + str(ancestral_set_in_counterfactual_factor_form_with_values_as_list)
    # )
    ancestral_set_in_counterfactual_factor_form_with_values = set(
        ancestral_set_in_counterfactual_factor_form_with_values_as_list
    )
    # logger.warning(
    #    "   As a set that is: " + str(ancestral_set_in_counterfactual_factor_form_with_values)
    # )
    ancestor_bases = {v.get_base() for v in ancestral_set}
    outcome_ancestor_graph = graph.subgraph(ancestor_bases)
    factorized_ancestral_set_with_values: list[
        set[tuple[Variable, Intervention | None]]
    ] = get_counterfactual_factors_retaining_variable_values(
        event=ancestral_set_in_counterfactual_factor_form_with_values, graph=outcome_ancestor_graph
    )

    return ancestral_set_with_values, factorized_ancestral_set_with_values


def _inconsistent_counterfactual_factor_variable_and_intervention_values(
    *, event: Collection[tuple[Variable, Intervention | None]]
) -> bool:
    r"""Determine whether a counterfactual factor has a variable value inconsistent with any intervention value.

    Source: Definition 4.1 in [correa22a]_ (Algorithm 2), part (i).
    :param event:
        $\mathbf{W_{\ast}}$, a set of counterfactual variables in V and $\mathbf{w_{\ast}}$, a set of
        values for $\mathbf{W_{\ast}}$. We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :returns: True if the counterfactual factor has a variable value inconsistent with any intervention value, and
        false otherwise.
    """
    intervention_variable_dictionary: dict[Variable, set[Intervention]] = defaultdict(set)
    # $W_{\ast}$
    counterfactual_factor_variable_names = {variable.get_base() for variable, _ in event}
    counterfactual_factor_variables_with_interventions = {
        variable for variable, _ in event if isinstance(variable, CounterfactualVariable)
    }
    # $\mathbf{T} \cup \mathbf{V(W_{\ast})} = \mathbf{Z}$ per Definition 4.1 (i)
    intervention_variable_names = {
        intervention.get_base()
        for variable in counterfactual_factor_variables_with_interventions
        for intervention in variable.interventions
    }.intersection(counterfactual_factor_variable_names)
    for counterfactual_factor_variable, counterfactual_factor_variable_value in event:
        if (
            counterfactual_factor_variable.get_base() in intervention_variable_names
            and counterfactual_factor_variable_value is not None
        ):
            intervention_variable_dictionary[counterfactual_factor_variable.get_base()].update(
                {counterfactual_factor_variable_value}
            )
        if isinstance(counterfactual_factor_variable, CounterfactualVariable):
            for intervention in counterfactual_factor_variable.interventions:
                if intervention.get_base() in intervention_variable_names:
                    intervention_variable_dictionary[intervention.get_base()].update({intervention})
    logger.warning(
        "In _inconsistent_counterfactual_factor_variable_and_intervention_values: dictionary = "
        + str(intervention_variable_dictionary)
    )
    return any(len(values) > 1 for values in intervention_variable_dictionary.values())


def _inconsistent_counterfactual_factor_variable_intervention_values(
    *, event: Collection[tuple[Variable, Intervention | None]]
) -> bool:
    r"""Determine whether a counterfactual factor has two inconsistent intervention values.

    Source: Definition 4.1 in [correa22a]_ (Algorithm 2), part (ii).
    :param event:
        $\mathbf{W_{\ast}}$, a set of counterfactual variables in V and $\mathbf{w_{\ast}}$, a set of
        values for $\mathbf{W_{\ast}}$. We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :returns: True if the counterfactual factor has at least two inconsistent intervention values, and false otherwise.
    """
    intervention_variable_dictionary = defaultdict(set)
    counterfactual_factor_variables_with_interventions = {
        variable for variable, _ in event if isinstance(variable, CounterfactualVariable)
    }
    # $\mathbf{T}$ per Definition 4.1 (ii)
    intervention_variable_names = {
        intervention.get_base()
        for variable in counterfactual_factor_variables_with_interventions
        for intervention in variable.interventions
    }
    for counterfactual_factor_variable, _ in event:
        if isinstance(counterfactual_factor_variable, CounterfactualVariable):
            for intervention in counterfactual_factor_variable.interventions:
                if intervention.get_base() in intervention_variable_names:
                    intervention_variable_dictionary[intervention.get_base()].add(intervention)
    logger.warning(
        "In _inconsistent_counterfactual_factor_variable_intervention_values: dictionary = "
        + str(intervention_variable_dictionary)
    )
    return any(len(values) > 1 for values in intervention_variable_dictionary.values())


def _counterfactual_factor_is_inconsistent(
    *, event: Collection[tuple[Variable, Intervention | None]]
) -> bool:
    r"""Determine whether a counterfactual factor is inconsistent.

      Source: Definition 4.1 in [correa22a]_ (Algorithm 2).
    :param event:
        $\mathbf{W_{\ast}}$, a set of counterfactual variables in V and $\mathbf{w_{\ast}}$, a set of
        values for $\mathbf{W_{\ast}}$. We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :returns: True if the counterfactual factor is consistent, and false if it is inconsistent.
    """
    # Are counterfactual factor and intervention values inconsistent?
    return _inconsistent_counterfactual_factor_variable_and_intervention_values(
        event=event
    ) or _inconsistent_counterfactual_factor_variable_intervention_values(event=event)
    # Are different counterfactual factor intervention values inconsistent?


def transport_unconditional_counterfactual_query(
    *,
    event: list[tuple[Variable, Intervention]],
    target_domain_graph: NxMixedGraph,
    domain_graphs: list[tuple[NxMixedGraph, list[Variable]]],
    domain_data: list[tuple[Collection[Variable], Expression]],
) -> tuple[Expression, list[tuple[Variable, Intervention]] | None] | None:
    r"""Implement the ctfTRu algorithm from [correa22a]_ (Algorithm 2).

    :param event:
        "Y_*, a set of counterfactual variables in V and y_* a set of
        values for Y_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param domain_graphs: A set of $K$ tuples, one for each of the $K$ domains. Each tuple
           contains a selection diagram for that domain and a topologically sorted list
           of all the vertices in the corresponding graph that are not transportability
           nodes. (Nodes that have no parents come first in such lists.) The selection
           diagram contains a transportability node for every vertex that is distributed
           differently in the domain in question than in the target domain (e.g., Vertex
           Z in Figure 3(a) in [correa22a]_), and it is a causal diagram such that its edges
           represent the state of the graph after a regime corresponding to domain $k$ has
           been applied (e.g., policy $\sigma_{X}$ in Figure 4 of [correa22a]_).
    :param target_domain_graph: a graph for the target domain.
    :param domain_data: Corresponding to $\mathcal{Z}$ in [correa22a]_, this is a set of
           $K$ tuples, one for each of the $K$ domains except for the target domain.
           Each tuple contains a set of variables corresponding to
           $\sigma_{\mathbf{Z}_{k}}$ and an expression denoting the probability distribution
           $P^{k}(\mathbf{V};\sigma_{\mathbf{Z}\_{j}})|{\mathbf{Z}_{j}} \in \mathcal{Z}^{i}$.
    :returns: an expression for $P^{\ast}(\mathbf{Y_{\ast}}=\mathbf{y_{\ast}})$
    """
    logger.warning("In transport_unconditional_counterfactual_query: input event = " + str(event))
    # Line 1
    simplified_event: list[tuple[Variable, Intervention]] | None = simplify(
        event=event, graph=target_domain_graph
    )
    logger.warning(
        "In transport_unconditional_counterfactual_query: simplifed event = "
        + str(simplified_event)
    )

    if simplified_event is not None:
        # Line 2
        # tuple[set[tuple[Variable, Intervention | None]], list[set[tuple[Variable, Intervention | None]]]
        (
            outcome_ancestors_with_values,
            counterfactual_factors_with_values,
        ) = _transport_unconditional_counterfactual_query_line_2(
            simplified_event, target_domain_graph
        )

        # Line 3
        if any(
            _counterfactual_factor_is_inconsistent(event=factor)
            for factor in counterfactual_factors_with_values
        ):
            logger.warning(
                "In transport_unconditional_counterfactual_query: inconsistent counterfactual "
                + "factor. Returning FAIL (None)"
            )
            return None  # This means FAIL

        # Line 4
        district_probabilities_intervening_on_parents = []
        for factor in counterfactual_factors_with_values:
            # Sigma-TR just takes in the district and then intervenes on the parents of the district. So
            # we need to strip the district variables of their interventions before running Sigma-TR.
            district_without_interventions = {variable.get_base() for variable, _ in factor}
            # Line 5
            district_probability_intervening_on_parents = transport_district_intervening_on_parents(
                district=district_without_interventions,
                domain_graphs=domain_graphs,
                domain_data=domain_data,
            )  # The q_value
            if district_probability_intervening_on_parents is None:
                logger.warning(
                    "In transport_unconditional_counterfactual_query: unable to transport "
                    + "counterfactual factor: "
                    + str(factor)
                )
                return None  # Return FAIL (Line 13)
            else:
                # Note about lines 7-9 of the algorithm:
                # Line 7 does not require a specific implementation. It is merely a comment to the
                #    reader regarding how the district probability intervening on the district's
                #    parents (i.e., its Q expression) is transported from one of the available domains.
                # Line 8 is a statement that the values over which the Q expression is evaluated come
                #    from the union of the parents of each of the parents of the district variables
                #    and the values of the district variables themselves. Many of these values are
                #    not associated with actual input variables $\mathbf{y_{\ast}}$. Instead, we
                #    will end up marginalizing over them in Line 14. Correa and Bareinboim point this
                #    out ([correa22a]_) in their appendix after Equation 27, stating that
                #    "$\mathbf{d_{\ast}}$ is the union of $\mathbf{z_{\ast}} and the indexing values
                #    of the sum," where the "sum" refers (in Algorithm 2) to
                #    $\sum_{\mathbf{w_{\ast}\backslash y_{\ast}}}\prod_{i}{P^{\ast}(\mathbf{C}_{i\ast}
                #       = \mathbf{c}_{i\ast})}$.
                #    So, especially because no actual values of the counterfactual factor variables
                #    or their parents will need to be used until after Line 14, we defer binding of
                #    these variable values to a Q expression until Line 14. Line 8 is therefore
                #    unnecessary for us as written. We do, however, check that the variables
                #    in the Q expression are contained in the set of variables referenced in
                #    Line 8.
                # Line 9 involves formally evaluating Q over the set of values $\mathbf{c}$. We defer
                #    this action until Line 14, when we do so by simply returning the simplified event
                #    with the expression for $P^{\ast}(\mathbf{Y_{\ast} = y_{\ast}})$.
                line_8_variables: set[Variable] = set()
                for variable in district_without_interventions:
                    line_8_variables = line_8_variables.union(
                        set(target_domain_graph.directed.predecessors(variable.get_base()))
                    )
                line_8_variables = line_8_variables.union(district_without_interventions)
                # logger.warning(
                #    "In transport_unconditional_counterfactual_query: line_8_variables = "
                #    + str(line_8_variables)
                # )
                if not all(
                    variable in line_8_variables
                    for variable in district_probability_intervening_on_parents.get_variables()
                ):
                    logger.warning(
                        "Found a variable in the Q expression that is not a district variable or one of its parents."
                    )
                    logger.warning("District variables and their parents: " + str(line_8_variables))
                    logger.warning(
                        "Q expression variables: "
                        + str(district_probability_intervening_on_parents.get_variables())
                    )
                logger.warning(
                    "In transport_unconditional_counterfactual_query: got a Q value of "
                    + district_probability_intervening_on_parents.to_latex()
                    + " for district "
                    + str(district_without_interventions)
                    + " corresponding to counterfactual factor "
                    + str(factor)
                    + "."
                )
                district_probabilities_intervening_on_parents.append(
                    district_probability_intervening_on_parents
                )
        ancestors_excluding_outcomes = {
            variable.get_base()
            for (variable, value) in outcome_ancestors_with_values
            if (variable, value) not in simplified_event
        }
        transported_unconditional_query = Sum.safe(
            Product.safe(district_probabilities_intervening_on_parents),
            ancestors_excluding_outcomes,
        )
        logger.warning(
            "Returning: "
            + transported_unconditional_query.to_latex()
            + " for simplified event: "
            + str(simplified_event)
        )
        return (transported_unconditional_query, simplified_event)
    else:
        return (Zero(), None)  # as specified by the output for Algorithm 1 in [correa22a]_


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
    raise NotImplementedError("Unimplemented function: conditioned_variables_in_ancestral_set")


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
    raise NotImplementedError(
        "Unimplemented function: "
        + "get_ancestral_set_after_intervening_on_conditioned_variables_in_the_set"
    )


def _compute_ancestral_components_from_ancestral_sets(
    *,
    ancestral_sets: set[frozenset[Variable]],
    graph: NxMixedGraph,
) -> frozenset[Variable]:
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
    raise NotImplementedError(
        "Unimplemented function: compute_ancestral_components_from_ancestral_sets"
    )


def _get_ancestral_components(
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
    raise NotImplementedError("Unimplemented function: _get_ancestral_components")


def transport_conditional_counterfactual_query(
    *,
    outcomes: list[tuple[Variable, Intervention]],
    conditions: list[tuple[Variable, Intervention]],
    target_domain_graph: NxMixedGraph,
    domain_graphs: list[tuple[NxMixedGraph, list[Variable]]],
    domain_data: list[tuple[Collection[Variable], Expression]],
) -> tuple[Expression, list[tuple[Variable, Intervention]] | None] | None:
    r"""Implement the ctfTR algorithm from [correa22a]_ (Algorithm 3).

    :param outcomes:
        "Y_*, a set of counterfactual variables in V and y_* a set of
        values for Y_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param conditions:
        "X_*, a set of counterfactual variables in V and x_* a set of
        values for X_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param domain_graphs: A set of $K$ tuples, one for each of the $K$ domains. Each tuple
           contains a selection diagram for that domain and a topologically sorted list
           of all the vertices in the corresponding graph that are not transportability
           nodes. (Nodes that have no parents come first in such lists.) The selection
           diagram contains a transportability node for every vertex that is distributed
           differently in the domain in question than in the target domain (e.g., Vertex
           Z in Figure 3(a) in [correa22a]_), and it is a causal diagram such that its edges
           represent the state of the graph after a regime corresponding to domain $k$ has
           been applied (e.g., policy $\sigma_{X}$ in Figure 4 of [correa22a]_).
    :param target_domain_graph: a graph for the target domain.
    :param domain_data: Corresponding to $\mathcal{Z}$ in [correa22a]_, this is a set of
           $K$ tuples, one for each of the $K$ domains except for the target domain.
           Each tuple contains a set of variables corresponding to
           $\sigma_{\mathbf{Z}_{k}}$ and an expression denoting the probability distribution
           $P^{k}(\mathbf{V};\sigma_{\mathbf{Z}\_{j}})|{\mathbf{Z}_{j}} \in \mathcal{Z}^{i}$.
    :returns: an expression for $P^{\ast}(\mathbf{Y_{\ast}}=\mathbf{y_{\ast}})$.
    """
    raise NotImplementedError("Unimplemented function: transport_conditional_counterfactual_query")
