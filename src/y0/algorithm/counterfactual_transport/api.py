"""Implementation of counterfactual transportability.

.. [correa22a] https://proceedings.mlr.press/v162/correa22a/correa22a.pdf.
.. [correa20a] https://proceedings.neurips.cc/paper/2020/file/7b497aa1b2a83ec63d1777a88676b0c2-Paper.pdf.
"""

import itertools as itt
import logging
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import NamedTuple

from networkx import is_directed_acyclic_graph

from y0.algorithm.tian_id import compute_c_factor, identify_district_variables
from y0.algorithm.transport import is_transport_node, transport_variable
from y0.dsl import (
    PP,
    TARGET_DOMAIN,
    CounterfactualVariable,
    Expression,
    Fraction,
    Intervention,
    One,
    P,
    Population,
    PopulationProbability,
    Probability,
    Product,
    Sum,
    Variable,
    Zero,
)
from y0.graph import NxMixedGraph

from .ancestor_utils import (
    get_ancestors_of_counterfactual,
    get_ancestral_components,
    minimize_counterfactual,
)

__all__ = [
    # TODO do a proper audit of which of these a user should ever have to import
    "unconditional_cft",
    "conditional_cft",
    "transport_unconditional_counterfactual_query",
    "transport_conditional_counterfactual_query",
    #
    "Event",
    "CFTDomain",
    "ConditionalCFTResult",
    "UnconditionalCFTResult",
    # Utilities
    "simplify",
    "minimize_event",
    "same_district",
    "is_counterfactual_factor_form",
    "get_counterfactual_factors",
    "convert_to_counterfactual_factor_form",
    "do_counterfactual_factor_factorization",
    "counterfactual_factors_are_transportable",
    "transport_district_intervening_on_parents",
    # TODO add functions/classes/variables you want to appear in the docs and be exposed to the user in this list
    #  Run tox -e docs then `open docs/build/html/index.html` to see docs
]

logger = logging.getLogger(__name__)

# FIXME potentially rename, since there are more strict events that are tuple[Variable, Intervention]
EventItem = tuple[Variable, Intervention | None]
Event = list[EventItem]


def event_to_probability(event: Event) -> Probability:
    """Turn an event list into a probability object."""
    values: list[Variable] = []
    for variable, value in event:
        if variable.star is not None:
            raise ValueError("events can not have star values for the variable")
        star = None if value is None else value.star
        if isinstance(variable, CounterfactualVariable):
            values.append(
                CounterfactualVariable(
                    name=variable.name,
                    star=star,
                    interventions=variable.interventions,
                )
            )
        else:
            values.append(Variable(name=variable.name, star=star))
    return Probability.safe(values)


def _any_variables_with_inconsistent_values(
    *,
    nonreflexive_variable_to_value_mappings: dict[Variable, set[Intervention | None]],
    reflexive_variable_to_value_mappings: dict[Variable, set[Intervention | None]],
) -> bool:
    r"""Check for variables with inconsistent values following Line 2 of Algorithm 1 in [correa_22a]_."""
    # Part 1 of Line 2:
    # :math: **if** there exists $Y_{\mathbf{x}}\in \mathbf{Y}_\ast$ with
    # two or more different values in  $\mathbf{y_\ast}$ **then return** 0.
    # Note this definition has to do with counterfactual values, and is different than
    # the "inconsistent counterfactual factor" definition in Definition 4.1 of [correa22a]_.
    if any(
        len(value_set) > 1 and None in value_set
        for value_set in nonreflexive_variable_to_value_mappings.values()
    ) or any(
        None in reflexive_variable_to_value_mappings[key]
        and not isinstance(key, CounterfactualVariable)
        and len(reflexive_variable_to_value_mappings[key]) > 1
        # FIXME use .items() when iterating over dicts to directly give a name to the value!
        for key in reflexive_variable_to_value_mappings
    ):
        # most of the type errors in this module should actually be value errors
        raise TypeError(
            "In _any_variables_with_inconsistent values: a variable lacking interventions on itself "
            + "has an assigned value and also a value of None. That should not occur. Check your inputs."
        )

    if any(len(value_set) > 1 for value_set in nonreflexive_variable_to_value_mappings.values()):
        return True

    # Part 2 of Line 2:
    # :math: **if** there exists $Y_y\in \mathbf{Y}_\ast$ with $\mathbf{y_*} \cap Y_y \neq y$ **then return** 0.
    if any(
        None in reflexive_variable_to_value_mappings[key]
        and (isinstance(key, CounterfactualVariable))
        for key in reflexive_variable_to_value_mappings
    ):
        raise TypeError(
            "In _any_variables_with_inconsistent values: a variable containing interventions on itself "
            + "has an assigned value of None. That should not occur. Check your inputs."
        )
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
    # Longer version of the above. (Old)
    # for variable in reflexive_variable_to_value_mappings.keys():  # Y_y, Y
    #    if not isinstance(variable, CounterfactualVariable):  # Y
    #        # TODO: Check with JZ that it's intended that $Y_y$ and $Y$ are the same.
    #        #       I infer that is so because of Equation 4 in [correa22a]_.
    #        # If Y takes on at least two values as part of the same query, then there exists
    #        # $Y_{y}\in \mathbf{Y}_\ast$ with two or more different values in  $\mathbf{y_\ast}$.
    #        # That implies that there exists $Y_y\in \mathbf{Y}_\ast$ with $\mathbf{y_*} \cap Y_y \neq y$,
    #        # so we return 0.
    #        if len(reflexive_variable_to_value_mappings[variable]) > 1:
    #            logger.debug("Part 2 of Line 2 fails for (non-counterfactual) variables: ")
    #            logger.debug(
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
    #                logger.debug(
    #                    "Part 2 of Line 2 fails: {{intervention}} = "
    #                    + str({intervention})
    #                    + " and reflexive_variable_to_value_mappings[variable] = "
    #                    + str(reflexive_variable_to_value_mappings[variable])
    #                )
    #                return True
    # return False


# Deprecated.
# def _simplify_self_interventions_with_consistent_values(
#    outcome_variable_to_value_mappings: defaultdict[Variable, set[Intervention]],
#    outcome_variables: set[Variable],
# ) -> tuple[defaultdict[Variable, set[Intervention]], set[Variable]]:
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


def _remove_repeated_variables_and_values(event: Event) -> dict[Variable, set[Intervention | None]]:
    r"""Implement the first half of Line 3 of the SIMPLIFY algorithm from [correa22a]_.

    The implementation is as simple as creating a dictionary. Adding variables to
    the dictionary removes repeated variables in the input event, and adding values to
    the dictionary using the variables as keys removes repeated values.

    :math: **if** there exists $Y_{\mathbf{x}}\in \mathbf{Y}_\ast$ with
    two consistent values in  $\mathbf{y_\ast} \cap Y_x$ **then**
    remove repeated variables from $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.

    In this function we also handle a case that [correa22a]_ does not address. Some variables
    passed in to Algorithm 2 of [correa22a]_ (transport_unconditional_counterfactual_query())
    from Algorithm 3 of [correa22a]_ (transport_conditional_counterfactual_query())
    may have values of None, because they were ancestors of a counterfactual when processed
    in Algorithm 3 and not variables passed in to Algorithm 3 along with actual values.
    Such variables may be redundant with variables passed in to Algorithm 2 for which values
    are observed. Because a value of None in this context implies that the variable could
    take any value, it is consistent with a specific value. Its intervention set is also
    guaranteed to match the intervention set of its counterpart that has a specific value, because
    all variables passed from Algorithm 3 to Algorithm 2 are in counterfactual factor form.
    Therefore, in this function, we want to treat the None value as consistent with a specific value.

    :param event:
        A tuple associating $\mathbf{Y_\ast}$, a set of counterfactual variables (or regular variables)
        in $\mathbf{V}$ with $\mathbf{y_\ast}$, a set of values for $\mathbf{Y_\ast}$. We encode the
        counterfactual variables as Variable objects, and the values as Intervention objects.
    :returns:
        A dictionary mapping the event variables to all values associated with each variable in the event.
    """
    variable_to_value_mappings: defaultdict[Variable, set[Intervention | None]] = defaultdict(set)
    for variable, value in event:
        variable_to_value_mappings[variable].add(value)
    for variable in variable_to_value_mappings.keys():
        if (
            len(variable_to_value_mappings[variable]) > 1
            and None in variable_to_value_mappings[variable]
        ):
            variable_to_value_mappings[variable].remove(None)
    return dict(variable_to_value_mappings)


def _split_event_by_reflexivity(event: Event) -> tuple[Event, Event]:
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
    reflexive_interventions_event: Event = [
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
    nonreflexive_interventions_event: Event = [
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
    variables: dict[Variable, set[Intervention | None]],
) -> dict[Variable, set[Intervention | None]]:
    r"""Simplify counterfactual variables intervening on themselves to Intervention objects with the same base.

    :param variables: A mapping :math: $\mathbf{Y_\ast}$, a set of counterfactual variables in
        $\mathbf{V}$, to $\mathbf{y_\ast}$, a set of values for $\mathbf{Y_\ast}$. Each variable in
        $\mathbf{Y_\ast}$ is assumed to be either $Y_{y}$ \in $\mathbf{Y_\ast}$ or just $Y$ in $\mathbf{Y_\ast}$,
        where $Y$ is considered a special case of $Y_{y}$ because minimization has already taken place.
        The $\mathbf{Y_\ast}$ variables are CounterfactualVariable objects, and the values are Intervention objects.
    :raises ValueError:
        a variable in the input dictionary has more than one intervention or its intervention is not itself.
    :returns:
        A mapping from simple variables :math: $\mathbf{Y}$ to $\mathbf{y}$ to a set of corresponding values.
    """
    result_dict: defaultdict[Variable, set[Intervention | None]] = defaultdict(set)
    for variable, interventions in variables.items():
        if not isinstance(variable, CounterfactualVariable):
            result_dict[variable].update(interventions)
        else:
            if len(variable.interventions) != 1:
                raise ValueError(f"Variable had more than one intervention: {variable}")
            if _check_nonreflexive(variable):
                raise ValueError(f"Variable had non-reflexive intervention: {variable}")
            result_dict[variable.get_base()].update(interventions)
    return dict(result_dict)


def _check_nonreflexive(variable: CounterfactualVariable) -> bool:
    base = variable.get_base()
    return any(intervention.get_base() != base for intervention in variable.interventions)


def simplify(*, event: Event, graph: NxMixedGraph) -> Event | None:
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
    if any(len(tup) != 2 for tup in event):
        raise TypeError(
            "Improperly formatted inputs for simplify(): an event element is a tuple with length not equal to 2."
        )
    for variable, intervention in event:
        if not isinstance(variable, Variable) or not (
            isinstance(intervention, Intervention) or intervention is None
        ):
            # FIXME please replace all instances of concatenating str() with usage of f strings
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
    minimized_event: Event = minimize_event(event=event, graph=graph)
    # logger.debug("In simplify: minimized_event = " + str(minimized_event))

    # Split the query into Y_x variables and Y_y ("reflexive") variables
    (
        reflexive_interventions_event,
        nonreflexive_interventions_event,
    ) = _split_event_by_reflexivity(minimized_event)

    # Creating this dict addresses part 1 of Line 3:
    # :math: If there exists $Y_{\mathbf{X}} \in \mathbf{Y_\ast}$ with two consistent values in
    # $\mathbf{y_\ast} \cap Y_{\mathbf{X}}$ then remove repeated variables from
    # $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.
    minimized_nonreflexive_variable_to_value_mappings: dict[Variable, set[Intervention | None]] = (
        _remove_repeated_variables_and_values(nonreflexive_interventions_event)
    )
    # Creating this dict partly addresses part 2 of Line 3:
    # :math: If there exists $Y_{y} \in \mathbf{Y_\ast}$ with
    # $\mathbf{y_\ast} \cap Y_{y} = y$ then remove repeated variables from
    # $\mathbf{Y_\ast}$ and values $\mathbf{y_\ast}$.
    #
    # There is an exception: we don't yet handle the edge case that the CounterfactualVariable Y_y
    # and the Intervention Y, when observed as part of the same event, are considered repeated
    # variables after minimization has taken place.
    minimized_reflexive_variable_to_value_mappings: dict[Variable, set[Intervention | None]] = (
        _remove_repeated_variables_and_values(reflexive_interventions_event)
    )

    # logger.debug(
    #    "In simplify after part 1 of line 3: minimized_nonreflexive_variable_to_value_mappings = "
    #    + str(minimized_nonreflexive_variable_to_value_mappings)
    # )
    # logger.debug(
    #    "                                    minimized_reflexive_variable_to_value_mappings = "
    #    + str(minimized_reflexive_variable_to_value_mappings)
    # )

    # Line 2 of SIMPLIFY.
    if _any_variables_with_inconsistent_values(
        nonreflexive_variable_to_value_mappings=minimized_nonreflexive_variable_to_value_mappings,
        reflexive_variable_to_value_mappings=minimized_reflexive_variable_to_value_mappings,
    ):
        return None

    # logger.debug(
    #    "In simplify after line 2: minimized_nonreflexive_variable_to_value_mappings = "
    #    + str(minimized_nonreflexive_variable_to_value_mappings)
    # )
    # logger.debug(
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

    # logger.debug(
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

    # logger.debug(
    #    "In simplify after line 2: minimized_nonreflexive_variable_to_value_mappings = "
    #    + str(minimized_nonreflexive_variable_to_value_mappings)
    # )
    # logger.debug(
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
    # FIXME please replace all instances of concatenating str() with usage of f strings
    logger.debug("In simplify before return: return value = " + str(simplified_event))
    return simplified_event


def minimize_event(*, event: Event, graph: NxMixedGraph) -> Event:
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
    return [(minimize_counterfactual(variable, graph), value) for variable, value in event]


def same_district(event: set[Variable], graph: NxMixedGraph) -> bool:
    """Check if a set of counterfactual variables are in the same district (c-component) of a graph.

    Edge cases: return True if the event contains one or no events.

    :param event: A set of counterfactual variables.
    :param graph: The graph containing them.
    :returns: A boolean.
    """
    if len(event) < 1:
        return True
    visited_districts: set[frozenset[Variable]] = {
        graph.get_district(variable.get_base()) for variable in event
    }
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
    :raises ValueError: An event is not in counterfactual factor form.
    :returns:
        A set of sorted lists of counterfactual variables in counterfactual factor
        form, with each list associated with a district of the graph. The lists
        need not contain every variable in the district, but they can't contain
        variables missing from the graph.
    """
    if not is_counterfactual_factor_form(event=event, graph=graph):
        logger.debug("Supposed to trigger ValueError in get_counterfactual_factors().")
        # FIXME please replace all instances of concatenating str() with usage of f strings
        raise ValueError(
            "In get_counterfactual_factors(): the event %s is not in counterfactual factor form.",
            str(event),
        )

    district_mappings: defaultdict[frozenset[Variable], set[Variable]] = defaultdict(set)
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
    :raises ValueError: An event is not in counterfactual factor form.
    :returns:
        A set of sorted lists of counterfactual variables in counterfactual factor
        form along with the values for those variables, with each list associated
        with a district of the graph. The lists need not contain every variable in
        the district, but they can't contain variables missing from the graph.
    """
    event_without_values = {variable for variable, _ in event}

    if not is_counterfactual_factor_form(event=event_without_values, graph=graph):
        logger.debug(
            "Supposed to trigger ValueError in get_counterfactual_factors_retaining_variable_values()."
        )
        # FIXME please replace all instances of concatenating str() with usage of f strings
        logger.debug(f"    Event = {event!s}")
        raise ValueError(
            "In get_counterfactual_factors_retaining_variable_values(): the event %s is not"
            + " in counterfactual factor form.",
            str(event),
        )

    district_mappings: defaultdict[
        frozenset[Variable], set[tuple[Variable, Intervention | None]]
    ] = defaultdict(set)
    for variable, value in event:
        district_mappings[graph.get_district(variable.get_base())].add((variable, value))

    logger.debug(
        "In get_counterfactual_factors_retaining_variable_values(): district_mappings = "
        + str(district_mappings)
    )

    # TODO if there aren't duplicates, this can be a set of frozensets
    return_value = [set(value) for value in district_mappings.values()]
    logger.debug(
        "In get_counterfactual_factors_retaining_variable_values(): returning " + str(return_value)
    )
    return return_value


def convert_to_counterfactual_factor_form(*, event: Event, graph: NxMixedGraph) -> Event:
    r"""Convert a list of variables and their values to counterfactual factor ("ctf-factor") form.

    That requires intervening on all the parents of each counterfactual variable.

    :param event: A list of variables (which may have interventions) and their values (:math:`\mathbf W_\ast`).
    :param graph: The corresponding graph.
    :returns:
        The output above, represented as a set of counterfactual variables (those without interventions
        are just variables).

        :math:`w_{1[\mathbf{pa_{1}}]},w_{2[\mathbf{pa_{2}}]},\cdots,w_{l[\mathbf{pa_{l}}]}`
        for each :math:`W_i \in \mathbf V`.
    """
    result: Event = []
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
            # FIXME why not just append the list? much better for readability
            result += [(variable.get_base().intervene(parents), value)]
        else:
            result += [(variable.get_base(), value)]
    return result


def do_counterfactual_factor_factorization(
    *, variables: list[tuple[Variable, Intervention]], graph: NxMixedGraph
) -> tuple[Expression, list[tuple[Variable, Intervention]]]:
    r"""Get the counterfactual factor form for a query, factorized according to the graph c-components.

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
            # FIXME the stack trace always says which function was called when an exception
            #  is raised, so it's not necessary to write the name of the function inside.
            #  in fact, it's better not to since this might get out of sync with the function name
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

    result_expression = Product.safe(P(factor) for factor in factorized_ancestral_set)

    # The summation portion of Equation 11 in [correa22a]_
    sum_range = ancestral_set_variable_names - outcome_variable_bases

    result_expression = Sum.safe(result_expression, sum_range)
    return result_expression, result_event


def counterfactual_factors_are_transportable(
    *, factors: set[Variable], domain_graph: NxMixedGraph
) -> bool:
    """Determine if a set of counterfactual factors can be transported from a domain to the target.

    :param domain_graph: Corresponds to the domain from which we're doing the transporting.
    :param factors: The counterfactual factors in question.
    :returns: Whether the query is transportable.
    """
    nodes = set(domain_graph.nodes())
    return not any(transport_variable(factor.get_base()) in nodes for factor in factors)


def _remove_transportability_vertices(*, vertices: Collection[Variable]) -> set[Variable]:
    """Remove the transportability nodes from a set of vertices.

    :param vertices: The input vertices.
    :returns: The input vertices, without the transportability nodes.
    """
    return {v for v in vertices if not is_transport_node(v)}


def validate_inputs_for_transport_district_intervening_on_parents(  # noqa:C901
    *,
    district: Collection[Variable],
    domain_graphs: list[tuple[NxMixedGraph, list[Variable]]],
    domain_data: list[tuple[Collection[Variable], PopulationProbability]],
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
    :raises NotImplementedError: a domain_data probability expression of type One() or Zero() was
           passed in.
    :raises TypeError: the input arguments are in an improper format or not internally consistent.
    :raises KeyError: a variable in an input argument is missing from another input argument
           and should be there.
    """
    # Preliminary checks, starting with type checking
    if not (isinstance(district, Collection) and all(isinstance(v, Variable) for v in district)):
        raise TypeError(
            "In validate_inputs_for_transport_district_intervening_on_parents: "
            + "the input district must be a Collection of Variable objects."
        )
    if not (isinstance(domain_graphs, list) and all(isinstance(t, tuple) for t in domain_graphs)):
        raise TypeError(
            "In validate_inputs_for_transport_district_intervening_on_parents: the "
            + "input domain graphs must be a list of tuples."
        )
    if not all(
        isinstance(g, NxMixedGraph)
        and isinstance(l_variables, list)
        and all(isinstance(v, Variable) for v in l_variables)
        for g, l_variables in domain_graphs
    ):
        raise TypeError(
            "In validate_inputs_for_transport_district_intervening_on_parents: the input domain "
            + "graph tuples must all contain NxMixedGraph objects and lists of variables."
        )
    if not (isinstance(domain_data, list) and all(isinstance(t, tuple) for t in domain_data)):
        raise TypeError(
            "In validate_inputs_for_transport_district_intervening_on_parents: the "
            + "input domain data must be a list of tuples."
        )
    if not all(
        isinstance(sigma_z, Collection)
        and all(isinstance(v, Variable) for v in sigma_z)
        and isinstance(e, Expression)
        for sigma_z, e in domain_data
    ):
        raise TypeError(
            "In validate_inputs_for_transport_district_intervening_on_parents: the input "
            + "domain data tuples must all contain Collections of Variable objects "
            + "(first element) and Expressions (second element)."
        )
    if any(e == Zero() or e == One() for _, e in domain_data):
        raise NotImplementedError(
            "In validate_inputs_for_transport_district_intervening_on_parents: this algorithm "
            + "does not currently handle domain_data probability expressions that are of type "
            + "One() or Zero()."
        )
    # Check we have no empty lists
    if len(domain_graphs) == 0 or len(domain_data) == 0:
        raise TypeError(
            "In validate_inputs_for_transport_district_intervening_on_parents: empty list for "
            + "either domain_graphs or domain_data. Check your inputs."
        )
    if len(district) == 0:
        raise TypeError(
            "In validate_inputs_for_transport_district_intervening_on_parents: the "
            + "input district cannot be an empty set."
        )
    if any(len(g.nodes()) == 0 for g, _ in domain_graphs):
        raise TypeError(
            "In validate_inputs_for_transport_district_intervening_on_parents: at least one input "
            + "domain graph contained no nodes. Check your inputs."
        )
    if any(len(topo) == 0 for _, topo in domain_graphs):
        raise TypeError(
            "In validate_inputs_for_transport_district_intervening_on_parents: an input set of "
            + "topologically sorted vertices was empty. Check your inputs."
        )
    if len(domain_graphs) != len(domain_data):
        raise TypeError(
            "In validate_inputs_for_transport_district_intervening_on_parents: the length of the "
            + "domain_graphs and domain_data must be the same."
        )
    # Technically the topologically sorted vertices could be for the graph $G$ containing $G_{\mathbf{C}_{i}}$,
    # but we currently have a stricter requirement that they are for $G_{\mathbf{C}_{i}}$. That requirement
    # could be relaxed if it becomes a computational burden in the ctf_TRu algorithm.
    for k in range(len(domain_graphs)):
        logger.debug("k = " + str(k))
        topo_vertices = frozenset(domain_graphs[k][1])
        expression_vertices = frozenset(domain_data[k][1].get_variables())
        graph_vertices = frozenset(domain_graphs[k][0].nodes())
        graph_vertices_without_transportability_nodes = frozenset(
            _remove_transportability_vertices(vertices=graph_vertices)
        )
        policy_vertices = frozenset(domain_data[k][0])
        if topo_vertices != graph_vertices:
            raise KeyError(
                "In validate_inputs_for_transport_district_intervening_on_parents: the vertices "
                + "in each domain graph must match those in the "
                + "corresponding topologically sorted list of vertices. Check your inputs and "
                + "note that the topologically sorted vertex lists "
                + "must contain any transportability nodes. An easy way to "
                + "generate a usable list is to call "
                + "[graph_name].topological_sort() on the graph. "
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
                "In validate_inputs_for_transport_district_intervening_on_parents: some of the "
                + "vertices in a domain graph do not appear in the expression"
                + " for the probability of the graph. Check your inputs. Graph vertices: "
                + str(graph_vertices_without_transportability_nodes)
                + ". Expression vertices: "
                + str(expression_vertices)
                + "."
            )
        if not all(v in graph_vertices_without_transportability_nodes for v in policy_vertices):
            raise KeyError(
                "In validate_inputs_for_transport_district_intervening_on_parents: the set of "
                + "vertices for which a policy has been applied for one "
                + "of the domains contains at least one vertex not in the domain graph. Check your inputs. "
                + "Policy vertices: "
                + str(policy_vertices)
                + ". Graph vertices: "
                + str(graph_vertices_without_transportability_nodes)
            )
        for v in district:
            if v not in graph_vertices_without_transportability_nodes:
                raise KeyError(
                    "In validate_inputs_for_transport_district_intervening_on_parents: one of "
                    + "the variables in the input district "
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
) -> bool:
    r"""Check that a district in a graph contains no intervention veriables.

    Helper function for the transport_district_intervening_on_parents algorithm
    from [correa22a]_ (Algorithm 4 in Appendix B).
    :param district: the C-component $\mathbf{C}\_{i}$ under analysis.
    :param interventions: Corresponding to $\mathcal{Z}$ in [correa22a]_, this is a set of
           variables corresponding to $\sigma_{\mathbf{Z}_{k}}$.
    :returns: true or false.
    """
    return len(set(district).intersection(interventions)) == 0


def _no_transportability_nodes_in_domain(
    *, district: Collection[Variable], domain_graph: NxMixedGraph
) -> bool:
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
    domain_data: list[tuple[Collection[Variable], PopulationProbability]],
) -> Expression | None:
    r"""Implement the sigma-TR algorithm from [correa22a]_ (Algorithm 4 in Appendix B).

    See also: [correa20a]_, which contains the original sigma-TR algorithm. Algorithm 4 in
    [correa22a]_ is the algorithm in [correa20a]_ modified to run on a single district in a
    graph.

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
    :param domain_data:
            .. todo::

                Start with a human readable version of explaining what the data structure is.
                Then, after it is possible to understand without thinking about math at all,
                you can then say "this corresponds to the impenetrable math notation used in some paper"

            Corresponding to $\mathcal{Z}$ in [correa22a]_, this is a set of
            $K$ tuples, one for each of the $K$ domains. Each tuple contains a set of
            variables corresponding to $\sigma_{\mathbf{Z}_{k}}$ and an expression
            denoting the probability distribution
            $P^{k}(\mathbf{V};\sigma_{\mathbf{Z}\_{j}})|{\mathbf{Z}_{j}} \in \mathcal{Z}^{i}$.
    :raises ValueError: the vertices in the input district are part of more than one district
           in a domain graph.
    :returns: A probabilistic expression for $P^{\ast}_{Pa(\mathbf{C})_{i}}(\mathbf{C}\_i)$ if
           it is transportable, or None if it is not transportable.
    """
    # We don't require the user to input $\mathcal{G}^{\ast}$, the target graph, and
    # therefore can't verify that the input district is in fact a district of the
    # target graph as part of validating the user input.
    validate_inputs_for_transport_district_intervening_on_parents(
        district=district, domain_graphs=domain_graphs, domain_data=domain_data
    )
    logger.debug("In transport_district_intervening_on_parents: input validated successfully.")
    # Line 1
    # FIXME use list comprehension + enumerate
    for k in range(len(domain_graphs)):
        # Also Line 1 (the published pseudocode could break the for loop and this test into two lines)
        logger.debug(" k = " + str(k))
        logger.debug(
            " _no_intervention_variables_in_domain: "
            + str(
                _no_intervention_variables_in_domain(
                    district=district, interventions=domain_data[k][0]
                )
            )
        )
        logger.debug(
            " _no_transportability_nodes_in_domain: "
            + str(
                _no_transportability_nodes_in_domain(
                    district=district, domain_graph=domain_graphs[k][0]
                )
            )
        )
        if _no_intervention_variables_in_domain(
            district=district, interventions=domain_data[k][0]
        ) and _no_transportability_nodes_in_domain(
            district=district, domain_graph=domain_graphs[k][0]
        ):
            logger.debug("In transport_district_intervening_on_parents: domain = " + str(k))
            domain_graph = domain_graphs[k][0]
            domain_graph_variables = _remove_transportability_vertices(
                vertices=domain_graph.nodes()
            )
            domain_topo = domain_graphs[k][1]

            # Line 2
            domain_graph_district = frozenset().union(
                *[domain_graph.get_district(v) for v in district]
            )  # $B_{i}$
            logger.debug(
                "In transport_district_intervening_on_parents: domain_graph_district = "
                + str(domain_graph_district)
            )
            # Sanity check: confirm that $C_{i} \subseteq B_{i}$
            if any(
                domain_graph_district != frozenset(domain_graph.get_district(v)) for v in district
            ):
                # FIXME use f-string and triple quote. This is unreadable.
                raise ValueError(
                    "Error in transport_district_intervening_on_parents: the vertices in an input district "
                    + "are part of more than one district in a domain graph. Input district: "
                    + str(district)
                    + " and domain_graph_district derived from it: "
                    + str(domain_graph_district)
                    + ". "
                    + "Also, here is each domain graph district we are comparing it to:"
                    + str({frozenset(domain_graph.get_district(v)) for v in district})
                    + "Domain index: "
                    + str(k)
                    + "."
                )

            # Line 3
            logger.debug("Subgraph_probability: " + domain_data[k][1].to_latex())
            domain_graph_district_q_probability = compute_c_factor(
                district=domain_graph_district,  # district=district,
                subgraph_variables=domain_graph_variables,
                subgraph_probability=domain_data[k][1],
                graph_topo=domain_topo,
            )
            logger.debug(
                "domain_graph_district_q_probability: "
                + domain_graph_district_q_probability.to_latex()
            )
            # Line 4
            district_q_probability = identify_district_variables(
                input_variables=frozenset(district),
                input_district=domain_graph_district,
                district_probability=domain_graph_district_q_probability,
                graph=domain_graph,
                topo=domain_topo,
            )
            # Incorporate the domain index in the representation for Q

            # Lines 5-7
            if district_q_probability is not None:
                return district_q_probability
    # Line 9
    return None


def _transport_unconditional_counterfactual_query_line_2(
    event: Event, graph: NxMixedGraph
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
        ancestral_set.update(get_ancestors_of_counterfactual(variable, graph))
    outcome_value_dict = dict(event)
    ancestral_set_with_values: set[tuple[Variable, Intervention | None]] = {
        (
            (variable, outcome_value_dict[variable])
            if variable in outcome_value_dict
            else (variable, None)
        )
        for variable in ancestral_set
    }
    logger.debug(
        "In _transport_unconditional_counterfactual_query_line_2: ancestral_set_with_values = "
        + str(ancestral_set_with_values)
    )

    #  e.g., Equation 13 in [correa22a]_, without the summation component.
    ancestral_set_in_counterfactual_factor_form_with_values: set[
        tuple[Variable, Intervention | None]
    ] = {
        (convert_to_counterfactual_factor_form(event=[(variable, value)], graph=graph)[0][0], value)
        for variable, value in ancestral_set_with_values
    }
    ancestor_bases = {v.get_base() for v in ancestral_set}
    outcome_ancestor_graph = graph.subgraph(ancestor_bases)
    factorized_ancestral_set_with_values: list[set[tuple[Variable, Intervention | None]]] = (
        get_counterfactual_factors_retaining_variable_values(
            event=ancestral_set_in_counterfactual_factor_form_with_values,
            graph=outcome_ancestor_graph,
        )
    )
    logger.debug(
        "In _transport_unconditional_counterfactual_query_line_2: factorized_ancestral_set_with_values = "
        + str(factorized_ancestral_set_with_values)
    )
    return ancestral_set_with_values, factorized_ancestral_set_with_values


def _any_variable_values_inconsistent_with_interventions(
    event: Collection[tuple[Variable, Intervention | None]],
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
    logger.debug(
        "In _any_variable_values_inconsistent_with_interventions: dictionary = "
        + str(intervention_variable_dictionary)
    )
    return any(len(values) > 1 for values in intervention_variable_dictionary.values())


def _any_inconsistent_intervention_values(
    event: Collection[tuple[Variable, Intervention | None]],
) -> bool:
    r"""Determine whether a counterfactual factor has two inconsistent intervention values.

    :param event:
        $\mathbf{W_{\ast}}$, a set of counterfactual variables in V and $\mathbf{w_{\ast}}$, a set of
        values for $\mathbf{W_{\ast}}$. We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :returns: True if the counterfactual factor has at least two inconsistent intervention values, and false otherwise.

    .. seealso:: Definition 4.1 in [correa22a]_ (Algorithm 2), part (ii).
    """
    base_to_interventions = defaultdict(set)
    # $\mathbf{T}$ per Definition 4.1 (ii)
    for counterfactual_factor_variable, _ in event:
        if isinstance(counterfactual_factor_variable, CounterfactualVariable):
            for intervention in counterfactual_factor_variable.interventions:
                base_to_interventions[intervention.get_base()].add(intervention)
    logger.debug(
        "In _any_inconsistent_intervention_values: dictionary = " + str(base_to_interventions)
    )
    return any(len(values) > 1 for values in base_to_interventions.values())


def _counterfactual_factor_is_inconsistent(
    event: Collection[tuple[Variable, Intervention | None]],
) -> bool:
    r"""Determine whether a counterfactual factor is inconsistent.

    :param event:
        $\mathbf{W_{\ast}}$, a set of counterfactual variables in V and $\mathbf{w_{\ast}}$, a set of
        values for $\mathbf{W_{\ast}}$. We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :returns: If the counterfactual factor is consistent

    .. seealso:: Source: Definition 4.1 in [correa22a]_ (Algorithm 2).
    """
    # Are counterfactual factor and intervention values inconsistent?
    return _any_variable_values_inconsistent_with_interventions(
        event
    ) or _any_inconsistent_intervention_values(event)
    # Are different counterfactual factor intervention values inconsistent?


def _validate_transport_unconditional_counterfactual_query_input(  # noqa:C901
    event: Event,
    target_domain_graph: NxMixedGraph,
    domain_graphs: list[tuple[NxMixedGraph, list[Variable]]],
    domain_data: list[tuple[Collection[Variable], PopulationProbability]],
) -> None:
    r"""Conduct pre-processing checks to transport unconditional counterfacutal queries (Algorithm 2 from [correa22a]_).

    :param event:
        "Y_*, a set of counterfactual variables in V and y_* a set of
        values for Y_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param target_domain_graph: a graph for the target domain.
    :param domain_graphs: A set of $K$ tuples, one for each of the $K$ domains. Each tuple
           contains a selection diagram for that domain and a topologically sorted list
           of all the vertices in the corresponding graph that are not transportability
           nodes. (Nodes that have no parents come first in such lists.) The selection
           diagram contains a transportability node for every vertex that is distributed
           differently in the domain in question than in the target domain (e.g., Vertex
           Z in Figure 3(a) in [correa22a]_), and it is a causal diagram such that its edges
           represent the state of the graph after a regime corresponding to domain $k$ has
           been applied (e.g., policy $\sigma_{X}$ in Figure 4 of [correa22a]_).
    :param domain_data: Corresponding to $\mathcal{Z}$ in [correa22a]_, this is a set of
           $K$ tuples, one for each of the $K$ domains except for the target domain.
           Each tuple contains a set of variables corresponding to
           $\sigma_{\mathbf{Z}_{k}}$ and an expression denoting the probability distribution
           $P^{k}(\mathbf{V};\sigma_{\mathbf{Z}\_{j}})|{\mathbf{Z}_{j}} \in \mathcal{Z}^{i}$.
    :raises TypeError: a validity check associated with either the input variables
           or the output event failed. See the error message for specifics.
    :raises ValueError: an input variable is of valid type but has an invalid value. See
           the error message for specifics.
    :raises NotImplementedError: this algorithm does not currently handle input graph probability
           Expression objects that are One() or Zero(), or cases where the conditioned and
           outcome variable sets share one or more variables in common.
    """
    # Here are all the checks (numbering is just based on convenience during implementation, and
    #    the numbered order is not necessarily the order of implementation):
    # 1. Type checking for the event
    # 2. Type checking for target_domain_graph
    # 3. Type checking for domain_graphs
    # 4. Type checking for domain_data
    #    4.5. Make sure probabilistic expressions in domain_data aren't Zero() or One()
    # 5. Make sure the event isn't empty
    # 6. (Skipped for the conditional transportability algorithm, included for unconditional
    #    transportability) Make sure at least one event element has a non-None value
    # 7. Check domain_graphs and domain_data aren't empty lists
    # 8. Check all graphs in domain_graphs have nodes
    # 9. Check all topologically sorted lists have entries
    # 9.2. Check that the target domain graph contains no transportability nodes and is a directed acyclic graph
    # 9.5. Check that the domain_graphs and domain_data list lengths are equal
    # 9.7. Check that every domain graph is a directed acyclic graph
    # 10. Check that every topological order list in domain_graphs is a valid topological order,
    #     given the corresponding graph
    # 11. Check the domain graph vertices are all the same as the target domain graph vertices,
    #     net of transportability nodes
    # 12. Check the event vertices are in the target domain graph (given check #11, that
    #     means they're in every graph)
    # 13. Check the event variables have the same base variable as the base variable of their
    #     corresponding values, or are none
    # 14. Domain graphs: make sure the vertex set of the topologically sorted vertex order matches
    #     the set of vertices in each corresponding domain graph
    # 15. It's possible for a graph probability expression to contain vertices not in the graph
    #     due to conditioning on vertices outside the graph. But the graph vertices must all be
    #     represented in that graph probability expression.
    # 15.5. All the policy vertices must be in the graph, for each domain.
    # 16. If the target domain graph is also in the domain_graphs list (i.e., data were collected for
    #     the target domain), then the target domain graph in the domain_graphs list must be
    #     identical to the target_domain_graph parameter.
    if not (isinstance(event, list) and all(isinstance(t, tuple) and len(t) == 2 for t in event)):
        raise TypeError(
            "In _validate_transport_unconditional_counterfactual_query_input: the input event "
            + "must be a list of tuples of length 2. Check your inputs."
        )
    if not all(
        isinstance(variable, Variable) and (value is None or isinstance(value, Intervention))
        for variable, value in event
    ):
        raise TypeError(
            "In _validate_transport_unconditional_counterfactual_query_input: each tuple in the input event "
            + "must contain a Variable object and its corresponding value (an Intervention or None). "
            + "Check your inputs."
        )

    # Type checking for inputs consistent with both Algorithms 2 and 3
    # 2.
    if not isinstance(target_domain_graph, NxMixedGraph):
        raise TypeError(
            "In _validate_transport_unconditional_counterfactual_query_input: the target_domain_graph "
            + "must be an NxMixedGraph object."
        )

    # Check we have no empty inputs
    # 5.
    if len(event) == 0:
        raise ValueError(
            "In _validate_transport_unconditional_counterfactual_query_input: empty list for "
            + "the event. Check your inputs."
        )
    # 8. (Target domain graph)
    if len(target_domain_graph.nodes()) == 0:
        raise ValueError(
            "In _validate_transport_unconditional_counterfactual_query_input: the target "
            + "domain graph contained no nodes. Check your inputs."
        )

    # Type checking for inputs consistent with Algorithms 2,3, and 4
    # 3.
    if not (isinstance(domain_graphs, list) and all(isinstance(t, tuple) for t in domain_graphs)):
        raise TypeError(
            "In _validate_transport_unconditional_counterfactual_query_input: the "
            + "domain_graphs input parameter must be a list of tuples."
        )
    if not all(
        isinstance(g, NxMixedGraph)
        and isinstance(l_variables, list)
        and all(isinstance(v, Variable) for v in l_variables)
        for g, l_variables in domain_graphs
    ):
        raise TypeError(
            "In _validate_transport_unconditional_counterfactual_query_input: the input domain "
            + "graph tuples must all contain NxMixedGraph objects and lists of variables."
        )
    # 4 and 4.5.
    if not (isinstance(domain_data, list) and all(isinstance(t, tuple) for t in domain_data)):
        raise TypeError(
            "In _validate_transport_unconditional_counterfactual_query_input: the "
            + "input domain data must be a list of tuples."
        )
    if any(e == Zero() or e == One() for _, e in domain_data):
        raise NotImplementedError(
            "In _validate_transport_unconditional_counterfactual_query_input: this algorithm "
            + "does not currently handle domain_data probability expressions that are of type "
            + "One() or Zero()."
        )
    if not all(
        isinstance(sigma_z, Collection)
        and all(isinstance(v, Variable) for v in sigma_z)
        and isinstance(pp, PopulationProbability)
        for sigma_z, pp in domain_data
    ):
        raise TypeError(
            "In _validate_transport_unconditional_counterfactual_query_input: the input "
            + "domain data tuples must all contain Collections of Variable objects "
            + "(first element) and PopulationProbability expressions (second element)."
        )

    # 6. (Skipped for the conditional transportability algorithm, included for unconditional
    #    transportability) Make sure at least one event element has a non-None value
    if all(value is None for _, value in event):
        raise ValueError(
            "In _validate_transport_unconditional_counterfactual_query_input: the event list "
            + "must contain at least one variable with a value that is not None. Check your inputs."
        )

    # Check we have no empty inputs (Algorithms 2, 3, and 4)
    # 7.
    if len(domain_graphs) == 0 or len(domain_data) == 0:
        raise ValueError(
            "In _validate_transport_unconditional_counterfactual_query_input: empty list for "
            + "either domain_graphs or domain_data. Check your inputs."
        )
    # 8.
    if any(len(g.nodes()) == 0 for g, _ in domain_graphs):
        raise ValueError(
            "In _validate_transport_unconditional_counterfactual_query_input: at least one input "
            + "domain graph contained no nodes. Check your inputs."
        )
    # 9.
    if any(len(topo) == 0 for _, topo in domain_graphs):
        raise ValueError(
            "In _validate_transport_unconditional_counterfactual_query_input: an input set of "
            + "topologically sorted vertices was empty. Check your inputs."
        )
    # 9.5.
    if len(domain_graphs) != len(domain_data):
        raise ValueError(
            "In _validate_transport_unconditional_counterfactual_query_input: the length of the "
            + "domain_graphs and domain_data must be the same."
        )

    # Check the target domain graph contains no transportability nodes and is a directed acyclic graph
    # 9.2.
    if frozenset(target_domain_graph.nodes()) != frozenset(
        _remove_transportability_vertices(vertices=target_domain_graph.nodes())
    ):
        raise ValueError(
            "In _validate_transport_unconditional_counterfactual_query_input: the target domain graph "
            + "cannot contain a transportability node. Check your inputs."
        )
    if not is_directed_acyclic_graph(target_domain_graph.directed):
        raise ValueError(
            "In _validate_transport_unconditional_counterfactual_query_input: the directed edges in "
            + "the target domain graph cannot form a cycle. Check your inputs."
        )
    # Check the domain graph vertices are all the same as the target domain graph vertices,
    #    net of transportability nodes
    # 11.
    if any(
        frozenset(target_domain_graph.nodes())
        != frozenset(_remove_transportability_vertices(vertices=domain_graph.nodes()))
        for domain_graph, _ in domain_graphs
    ):
        raise ValueError(
            "In _validate_transport_unconditional_counterfactual_query_input: a domain graph contained"
            + " different vertices than the target domain graph after excluding transportability "
            + "nodes. Check your inputs."
        )
    # Check the event vertices are in the target domain graph (given the above check, that means they're in every graph)
    # 12.
    if any(variable.get_base() not in target_domain_graph.nodes() for variable, _ in event):
        raise ValueError(
            "In _validate_transport_unconditional_counterfactual_query_input: one of the input "
            + "event variables is not in the target domain graph. Check your inputs. "
        )
    # 13.
    if any(
        value is not None and variable.get_base() != value.get_base() for variable, value in event
    ):
        raise ValueError(
            "In _validate_transport_unconditional_counterfactual_query_input: all input "
            + "event variables must either have values of None or the same base variable "
            + "as their corresponding values (e.g., your variable is (W @ -X) and its value "
            + "must be +W or -W, but it's -X). Check your inputs."
        )

    # Technically the topologically sorted vertices could be for a superset of the vertices
    # in the input graphs, but we currently require them to be for the vertices in the input graphs.
    for k in range(len(domain_graphs)):
        topo_vertices = frozenset(domain_graphs[k][1])
        expression_vertices = frozenset(domain_data[k][1].get_variables())
        graph_vertices = frozenset(domain_graphs[k][0].nodes())
        graph_vertices_without_transportability_nodes = frozenset(
            _remove_transportability_vertices(vertices=graph_vertices)
        )
        policy_vertices = frozenset(domain_data[k][0])
        # 14.
        if topo_vertices != graph_vertices:
            raise ValueError(
                "In _validate_transport_unconditional_counterfactual_query_input: the vertices "
                + "in each domain graph must match those in the "
                + "corresponding topologically sorted list of vertices. Check your inputs and "
                + "note that the topologically sorted vertex lists "
                + "must contain any transportability nodes. An easy way to "
                + "generate a usable list is to call "
                + "[graph_name].topological_sort() on the graph. "
                + "Graph vertices: "
                + str(graph_vertices)
                + ". Topologically sorted list of vertices: "
                + str(topo_vertices)
                + ". Domain index (zero-indexed): "
                + str(k)
            )
        # It's possible for a graph probability expression to contain vertices not in the graph
        # due to conditioning on vertices outside the graph. But the graph vertices must all be
        # represented in that graph probability expression.
        # 15.
        if not all(v in expression_vertices for v in graph_vertices_without_transportability_nodes):
            raise ValueError(
                "In _validate_transport_unconditional_counterfactual_query_input: some of the "
                + "vertices in a domain graph do not appear in the expression"
                + " for the probability of the graph. Check your inputs. Graph vertices: "
                + str(graph_vertices_without_transportability_nodes)
                + ". Expression vertices: "
                + str(expression_vertices)
                + ". Domain index (zero-indexed): "
                + str(k)
            )
        # 15.5.
        if not all(v in graph_vertices_without_transportability_nodes for v in policy_vertices):
            raise ValueError(
                "In _validate_transport_unconditional_counterfactual_query_input: the set of "
                + "vertices for which a policy has been applied for one "
                + "of the domains contains at least one vertex not in the domain graph. Check your inputs. "
                + "Policy vertices: "
                + str(policy_vertices)
                + ". Graph vertices: "
                + str(graph_vertices_without_transportability_nodes)
                + ". Domain index (zero-indexed): "
                + str(k)
            )
        # 9.7. (The directed acyclic graph check must come before the topological order check)
        if not is_directed_acyclic_graph(domain_graphs[k][0].directed):
            raise ValueError(
                "In _validate_transport_unconditional_counterfactual_query_input: the directed edges in "
                + "domain graph entry "
                + str(k)
                + " (zero-indexed) form a cycle and the graph must be a "
                + "directed acyclic graph. Check your inputs."
            )
        # 10.
        if not _valid_topo_list(topo=domain_graphs[k][1], graph=domain_graphs[k][0]):
            raise ValueError(
                "In _validate_transport_unconditional_counterfactual_query_input: the provided topologically "
                + "sorted order of the vertices ("
                + str(domain_graphs[k][1])
                + ") for domain graph entry "
                + str(k)
                + " (zero-indexed) "
                + "is not valid, given the input domain_graph. Check your inputs."
            )
        # 16. If the target domain graph is also in the domain_graphs list (i.e., data were collected for
        #     the target domain), then the target domain graph in the domain_graphs list must be
        #     identical to the target_domain_graph parameter.
        # TODO: relax the code base to allow Expressions in the domain_data instead of
        #     PopulationProbability types as follows:
        # if isinstance(domain_data[k][1], PopulationProbability) and
        #    str(domain_data[k][1].population)==str(TARGET_DOMAIN):
        #    That covers a corner case where a user wishes to run this algorithm with a single domain and
        #    without specifying a population for the graph probability. In that case the problem reduces
        #    to running ID* anyway, so it's not likely to get much usage.
        if str(domain_data[k][1].population) == str(TARGET_DOMAIN):
            if domain_graphs[k][0] != target_domain_graph:
                raise ValueError(
                    "In _validate_transport_unconditional_counterfactual_query_input: the domain_data input contains "
                    + 'a graph probability expression from the target domain (i.e., "pi*"), but the corresponding '
                    + "domain_graph is not the same graph as the target_domain_graph. Check your inputs. Domain index "
                    + "(zero-indexed): "
                    + str(k)
                )
    return


@dataclass
class CFTDomain:
    r"""Represents a counterfactual transport domain.

    Each CFTDomain class contains a selection diagram for that domain,
    an expression denoting the probability distribution
    $P^{k}(\mathbf{V};\sigma_{\mathbf{Z}\_{j}})|{\mathbf{Z}_{j}} \in \mathcal{Z}^{i}$,
    a set of policy variables corresponding to $\sigma_{\mathbf{Z}_{k}}$,
    and a topologically sorted list of all the vertices in the corresponding graph
    that are not transportability nodes. (Nodes that have no parents come first in
    such lists.) The selection diagram contains a transportability node for every
    vertex that is distributed differently in the domain in question than in the
    target domain (e.g., Vertex Z in Figure 3(a) in [correa22a]_), and it is a causal
    diagram such that its edges represent the state of the graph after a regime
    corresponding to domain $k$ has been applied (e.g., policy $\sigma_{X}$ in Figure
    4 of [correa22a]_).
    """

    #: The domain graph (a selection diagram, containing transportability nodes)
    graph: NxMixedGraph
    #: An expression for the joint probability of the vertices in the domain graph,
    #: which may be a conditional probability with vertices not in the graph to the
    #: right of the conditioning bar (i.e., we may be getting a subgraph of something larger).
    #: Can also directly give a Variable, in which case, the population probability is inferred
    #: to be the joint population probability over all non-transport nodes in the graph
    population: PopulationProbability
    #: The variables in the domain receiving an intervention, which may be stochastic
    #: (i.e., intervening on the variable does not necessarily break all incoming
    #: edges and may even add some edges)
    policy_variables: Collection[Variable] = field(default_factory=set)
    #: Topological ordering of the vertices in the domain graph (optional)
    ordering: list[Variable] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.population, Population):
            nodes = [node for node in self.graph.nodes() if not is_transport_node(node)]
            self.population = PP[self.population](nodes)


class UnconditionalCFTResult(NamedTuple):
    """Represents the result from an unconditional counterfactual transportability query."""

    expression: Expression
    event: Event | None

    def display(self) -> None:
        """Display this result."""
        from IPython.display import display

        display(event_to_probability(self.event))  # type:ignore
        display(self.expression)


def _event_from_counterfactuals(
    variables: Variable | list[Variable],
) -> list[tuple[Variable, Intervention | None]]:
    if isinstance(variables, Variable):
        variables = [variables]
    rv = []
    for variable in variables:
        base = _event_base(variable)
        if variable.star is not None:
            value = Intervention(name=variable.name, star=variable.star)
        else:
            value = None
        rv.append((base, value))
    return rv


def _event_from_counterfactuals_strict(
    variables: Variable | list[Variable],
) -> list[tuple[Variable, Intervention]]:
    if isinstance(variables, Variable):
        variables = [variables]
    rv = []
    for variable in variables:
        base = _event_base(variable)
        if variable.star is not None:
            value = Intervention(name=variable.name, star=variable.star)
        else:
            raise TypeError
        rv.append((base, value))
    return rv


def _event_base(variable: Variable) -> Variable:
    if isinstance(variable, CounterfactualVariable):
        return CounterfactualVariable(
            name=variable.name,
            star=None,
            interventions=variable.interventions,
        )
    else:
        return variable.get_base()


def unconditional_cft(
    *,
    event: Variable | list[Variable],
    target_domain_graph: NxMixedGraph,
    domains: list[CFTDomain],
) -> UnconditionalCFTResult | None:
    r"""Run an unconditional counterfactual transportability query (Algorithm 2 from [correa22a]_).

    :param event:
        "Y_*, a set of counterfactual variables in V and y_* a set of
        values for Y_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param target_domain_graph: a graph for the target domain.
    :param domains: A set of $K$ CFTDomain classes, one for each of the $K$ domains. See the
        documentation for CFTDomain for more details.

    :returns: The result of the query as an UnconditionalCFTResult object.
    """
    domain_graphs = [
        (domain.graph, domain.ordering or domain.graph.topological_sort()) for domain in domains
    ]
    domain_data = [(domain.policy_variables, domain.population) for domain in domains]
    return transport_unconditional_counterfactual_query(
        event=_event_from_counterfactuals(event),
        target_domain_graph=target_domain_graph,
        domain_graphs=domain_graphs,
        domain_data=domain_data,
    )


def transport_unconditional_counterfactual_query(
    *,
    event: Event,
    target_domain_graph: NxMixedGraph,
    domain_graphs: list[tuple[NxMixedGraph, list[Variable]]],
    domain_data: list[tuple[Collection[Variable], PopulationProbability]],
) -> UnconditionalCFTResult | None:
    r"""Implement the ctfTRu algorithm from [correa22a]_ (Algorithm 2).

    :param event:
        "Y_*, a set of counterfactual variables in V and y_* a set of
        values for Y_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param target_domain_graph: a graph for the target domain.
    :param domain_graphs: A set of $K$ tuples, one for each of the $K$ domains. Each tuple
           contains a selection diagram for that domain and a topologically sorted list
           of all the vertices in the corresponding graph that are not transportability
           nodes. (Nodes that have no parents come first in such lists.) The selection
           diagram contains a transportability node for every vertex that is distributed
           differently in the domain in question than in the target domain (e.g., Vertex
           Z in Figure 3(a) in [correa22a]_), and it is a causal diagram such that its edges
           represent the state of the graph after a regime corresponding to domain $k$ has
           been applied (e.g., policy $\sigma_{X}$ in Figure 4 of [correa22a]_).
    :param domain_data: Corresponding to $\mathcal{Z}$ in [correa22a]_, this is a set of
           $K$ tuples, one for each of the $K$ domains except for the target domain.
           Each tuple contains a set of variables corresponding to
           $\sigma_{\mathbf{Z}_{k}}$ and an expression denoting the probability distribution
           $P^{k}(\mathbf{V};\sigma_{\mathbf{Z}\_{j}})|{\mathbf{Z}_{j}} \in \mathcal{Z}^{i}$.
    :returns: an expression for $P^{\ast}(\mathbf{Y_{\ast}}=\mathbf{y_{\ast}})$
    """
    _validate_transport_unconditional_counterfactual_query_input(
        event=event,
        target_domain_graph=target_domain_graph,  #: NxMixedGraph,
        domain_graphs=domain_graphs,  #: list[tuple[NxMixedGraph, list[Variable]]],
        domain_data=domain_data,  #: list[tuple[Collection[Variable], PopulationProbability]],
    )
    # Line 1
    simplified_event: Event | None = simplify(event=event, graph=target_domain_graph)
    if simplified_event is None:
        # as specified by the output for Algorithm 1 in [correa22a]_
        return UnconditionalCFTResult(expression=Zero(), event=simplified_event)

    # Line 2
    # FIXME if you want to make type annotations, then create a named tuple
    # tuple[set[tuple[Variable, Intervention | None]], list[set[tuple[Variable, Intervention | None]]]
    (
        outcome_ancestors_with_values,
        counterfactual_factors_with_values,
    ) = _transport_unconditional_counterfactual_query_line_2(simplified_event, target_domain_graph)

    # Line 3
    if any(
        _counterfactual_factor_is_inconsistent(factor)
        for factor in counterfactual_factors_with_values
    ):
        logger.debug(
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
        logger.debug(
            "In transport_unconditional_counterfactual_query: attempting to transport district "
            + str(district_without_interventions)
        )
        district_probability_intervening_on_parents = transport_district_intervening_on_parents(
            district=district_without_interventions,
            domain_graphs=domain_graphs,
            domain_data=domain_data,
        )  # The q_value
        if district_probability_intervening_on_parents is None:
            # logger.debug(
            #    "In transport_unconditional_counterfactual_query: unable to transport "
            #    + "counterfactual factor: "
            #    + str(factor)
            # )
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
            district_variables_and_their_parents: set[Variable] = (
                set()
            )  # district_without_interventions
            for variable in district_without_interventions:
                district_variables_and_their_parents.update(
                    set(target_domain_graph.directed.predecessors(variable.get_base()))
                )
            district_variables_and_their_parents.update(district_without_interventions)
            logger.debug(
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
        for variable, value in outcome_ancestors_with_values
        if (variable, value) not in simplified_event
    }
    transported_unconditional_query = Sum.safe(
        Product.safe(district_probabilities_intervening_on_parents),
        ancestors_excluding_outcomes,
    )
    return UnconditionalCFTResult(
        expression=transported_unconditional_query, event=simplified_event
    )


# Internal subroutine for transport_conditional_counterfactual_query()
def _initialize_conditional_transportability_data_structures(
    *,
    outcomes: list[tuple[Variable, Intervention]],
    conditions: list[tuple[Variable, Intervention]],
) -> tuple[
    set[Variable],
    set[Variable],
    set[Variable],
    dict[Variable, set[Intervention]],
    dict[Variable, set[Intervention]],
    set[Variable],
    set[Variable],
]:
    r"""Set up data structures to process a conditional counterfactual query per Algorithm 3 of [correa22a]_.

    :param outcomes:
        "Y_*, a set of counterfactual variables in V and y_* a set of
        values for Y_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and their values as Intervention objects.
    :param conditions:
        "X_*, a set of counterfactual variables in V and x_* a set of
        values for X_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and their values as Intervention objects.
    :returns: a tuple of hash tables and dictionaries that speed processing of a conditional counterfactual query.
    """
    conditioned_variables: set[Variable] = {variable for variable, _ in conditions}
    outcome_variables: set[Variable] = {outcome for outcome, _ in outcomes}
    outcome_and_conditioned_variables: set[Variable] = conditioned_variables.union(
        outcome_variables
    )
    outcome_variable_to_value_mappings: defaultdict[Variable, set[Intervention]] = defaultdict(set)
    # outcome_and_conditioned_variable_to_value_mappings: defaultdict[
    #    Variable, set[Intervention]
    # ] = defaultdict(set)
    outcome_and_conditioned_variable_names_to_values: defaultdict[Variable, set[Intervention]] = (
        defaultdict(set)
    )
    for key, value in outcomes:
        outcome_variable_to_value_mappings[key].update({value})
        # outcome_and_conditioned_variable_to_value_mappings[key].update({value})
        outcome_and_conditioned_variable_names_to_values[key.get_base()].add(value)
    for key, value in conditions:
        # outcome_and_conditioned_variable_to_value_mappings[key].update({value})
        outcome_and_conditioned_variable_names_to_values[key.get_base()].add(value)

    outcome_and_conditioned_variable_names: set[Variable] = {
        v.get_base() for v in outcome_and_conditioned_variables
    }
    conditioned_variable_names: set[Variable] = {v.get_base() for v in conditioned_variables}

    # TODO use named tuple here
    return (
        conditioned_variables,
        outcome_variables,
        outcome_and_conditioned_variables,
        dict(outcome_variable_to_value_mappings),
        dict(outcome_and_conditioned_variable_names_to_values),
        outcome_and_conditioned_variable_names,
        conditioned_variable_names,
    )


def _transport_conditional_counterfactual_query_line_2(
    *,
    ancestral_components: frozenset[frozenset[Variable]],
    outcome_variables: set[Variable],
    outcome_variable_to_value_mappings: dict[Variable, set[Intervention]],
    target_domain_graph: NxMixedGraph,
) -> tuple[Event, set[Variable]]:
    r"""Set up data structures to process a conditional counterfactual query per Algorithm 3 of [correa22a]_.

    This function is an internal subroutine for transport_conditional_counterfactual_query().

    :param ancestral_components:
        The ancestral components associated with a conditional counterfactual query, encoded as a set of sets of
        counterfactual variables.

        :math: Let $\mathbf{W_{\ast}}$ be a set of counterfactual variables,
           $\mathbf{X_{\ast}} \subseteq \mathbf{W_{\ast}}$,
           and $\mathcal{G}$ be a causal diagram. Then the ancestral components induced by $\mathbf{W_{\ast}}$,
           given $\mathbf{X_{\ast}}$, are sets $\mathbf{A}_{1\ast},\mathbf{A}_{2\ast},\ldots$ that form a partition
           over $An(\mathbf{W_{\ast}})$, made of unions of the ancestral sets
           $An(W_\mathbf{t})_{\mathcal{G}_{\underline{\mathbf{X_{\ast}}(W_\mathbf{t})}}}$,
           $W_{\mathbf{t}} \in \mathbf{W_{\ast}}$. Sets
           $An(W_{1\left[\mathbf{t}_{1}\right]})_{\mathcal{G}_{\underline{\mathbf{X_{\ast}}(W_{1\left[\mathbf{t}_{1}\right]})}}}$
           and
           $An(W_{2\left[\mathbf{t}_{2}\right]})_{\mathcal{G}_{\underline{\mathbf{X_{\ast}}(W_{2\left[\mathbf{t}_{2}\right]})}}}$
           are put together if they are not disjoint or there exists a bidirected arrow in $\mathcal{G}$
           connecting variables in those sets.
    :param outcome_variables:
        "Y_*, a set of counterfactual variables in V.", encoded as a set of Variable objects.
    :param outcome_variable_to_value_mappings:
        A dictionary mapping Variable objects to sets of Intervention objects representing the values
        those variables attain in the query.
    :param target_domain_graph: The associated graph, for the target domain.
    :returns: The union of the ancestral components containing at least one outcome variable:

        :math: "Let $\mathbf{D_{\ast}}$ be the union of the ancestral components containing a variable in
           $\mathbf{Y_{\ast}}$ and $\mathbf{d_{\ast}}$ the corresponding set of values" ([correa22a]_, Algorithm 3).

        This function converts the variables to counterfactual factor form in preparation for calling Line 3 of
        Algorithm 3 of [correa22a]_. It also returns a set of variables representing the target domain graph vertices
        associated with variables in $\mathbf{D_{\ast}}$.
    """
    outcome_ancestral_component_variables_and_values: Event = []
    outcome_variable_ancestral_component_variables: set[Variable] = set()
    for component in ancestral_components:
        if any(variable in outcome_variables for variable in component):
            outcome_variable_ancestral_component_variables.update(set(component))
    for variable in outcome_variable_ancestral_component_variables:
        if variable not in outcome_variable_to_value_mappings:
            outcome_ancestral_component_variables_and_values.append((variable, None))
        else:
            # There could be redundant values for a variable, and Simplify() will catch them
            for value in outcome_variable_to_value_mappings[variable]:
                outcome_ancestral_component_variables_and_values.append((variable, value))
    outcome_ancestral_component_query_in_counterfactual_factor_form: Event = (
        convert_to_counterfactual_factor_form(
            event=outcome_ancestral_component_variables_and_values, graph=target_domain_graph
        )
    )
    outcome_variable_ancestral_component_variable_names = {
        variable.get_base() for variable in outcome_variable_ancestral_component_variables
    }
    return (
        outcome_ancestral_component_query_in_counterfactual_factor_form,
        outcome_variable_ancestral_component_variable_names,
    )


def _validate_transport_conditional_counterfactual_query_line_4_output(
    *,
    simplified_event: Event,
    outcome_and_conditioned_variable_names: set[Variable],
    outcome_and_conditioned_variable_names_to_values: dict[Variable, set[Intervention]],
    outcome_ancestral_component_variables_with_no_values: set[Variable],
    result_expression: Expression,
    result_event: list[tuple[Variable, Intervention]],
    domain_data: list[tuple[Collection[Variable], PopulationProbability]],
) -> None:
    r"""Perform validity checks on output for Algorithm 3 of [correa22a]_.

    This function is an internal subroutine for transport_conditional_counterfactual_query(). It performs
    some validity checks and does not return any value if successful or raises a KeyError or TypeError if not.

    :param simplified_event:
        This is the set of variables $D_{\ast}$ (in counterfactual factor form) and their values $d_{\ast}$
        after getting processed by the simplify() algorithm (Algorithm 1 in [correa22a]_, called from Line 1 in
        Algorithm 2 of [correa22a]_. ) We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param outcome_and_conditioned_variable_names: The graph vertices associated with outcome and conditioned variables
        in the query (stripped of any interventions), represented as a set of Variable objects.
    :param outcome_and_conditioned_variable_names_to_values: a dictionary mapping those variables to their values
        from the query passed in to transport_conditional_counterfactual_query() as a parameter.
    :param outcome_ancestral_component_variables_with_no_values: the graph vertices associated with elements of
        the ancestral components of $\mathbf{Y_{\ast}} \cap \mathbf{X_{\ast}}$ given $\mathbf{X_{\ast}}$
        that are neither outcome variables nor conditioned variables in the query, and therefore do not have
        values assigned to those variables. Represented as a set of Variable objects.
    :param result_expression: the probabilistic expression to be returned by this query (if this function gets called,
        the query is not expected to fail or return a probability of zero due to inconsistent query variable values).
    :param result_event: a list of tuples of variables and their values used to evaluate the result_expression.
    :param domain_data: Corresponding to $\mathcal{Z}$ in [correa22a]_, this is a set of
           $K$ tuples, one for each of the $K$ domains except for the target domain.
           Each tuple contains a set of variables corresponding to
           $\sigma_{\mathbf{Z}_{k}}$ and an expression denoting the probability distribution
           $P^{k}(\mathbf{V};\sigma_{\mathbf{Z}\_{j}})|{\mathbf{Z}_{j}} \in \mathcal{Z}^{i}$.
           Passed in to transport_conditional_counterfactual_query() as an input parameter.
    :raises KeyError: a variable in one of the input parameters should appear in another parameter and does not,
           or should not and does. See each error description for specifics.
    :raises TypeError: a return value used to evaluate the result_expression is None and that should not happen.
    """
    simplified_event_variable_names_to_values: dict[Variable, Intervention | None] = {
        variable.get_base(): value for variable, value in simplified_event
    }
    # 1. Make sure in the simplified event we got back, all of the variables
    #    that have values are either variables in the outcomes or variables
    #    in the conditions.
    if any(
        name not in outcome_and_conditioned_variable_names
        for name in simplified_event_variable_names_to_values.keys()
        if simplified_event_variable_names_to_values[name] is not None
    ):
        raise KeyError(
            "In final checks for transport_conditional_counterfactual_query: a variable "
            + "that transport_unconditional_counterfactual_query() returned to evaluate "
            + "the return expression is not one of the input variables (outcomes and "
            + "conditioned variables."
        )
    # 2. Make sure the values for those variables in the simplified event match
    #    the values for their corresponding outcome or condition variables
    #    in the input for this function.
    # logger.debug("In _validate_transport_conditionalcounterfactual_query_line_4_output: ")
    # logger.debug("    simplified_event: " + str(simplified_event))
    # logger.debug(
    #    "    outcome_and_conditioned_variable_names_to_values: "
    #    + str(outcome_and_conditioned_variable_names_to_values)
    # )
    if any(
        value not in outcome_and_conditioned_variable_names_to_values[variable.get_base()]
        for variable, value in simplified_event
        if simplified_event_variable_names_to_values[variable.get_base()] is not None
    ):
        raise KeyError(
            "In final checks for transport_conditional_counterfactual_query: a value "
            + "of a variable that transport_unconditional_counterfactual_query() returned "
            + "for the purpose of evaluating the return expression is not a value associated"
            + " with one of the input variables (outcomes and conditioned variables) "
            + "that has the same name after ignoring any intervention set."
        )
    # 3. Make sure all the variables in the expression this function will return are either
    #    in the outcomes, the conditions, one of the expressions passed in with the
    #    domain data, or the outcome ancestral component variables excluding outcomes and conditions.
    if not all(
        variable in outcome_ancestral_component_variables_with_no_values
        or variable in outcome_and_conditioned_variable_names
        or any(variable in expression.get_variables() for _, expression in domain_data)
        for variable in result_expression.get_variables()
    ):
        raise KeyError(
            "In final checks for transport_conditional_counterfactual_query: a variable in "
            + "the expression that transport_unconditional_counterfactual_query() "
            + "returned is neither one of the input variables (outcomes and conditioned "
            + "variables) nor a variable in one of the ancestral components containing "
            + "an outcome variable. This validity check ignored interventions in all variables."
        )
    # 4. Make sure we're not going to return any variables with None values. TODO: This check
    #    won't be necessary after we verify that the input outcomes and conditions have no
    #    None values.
    if any(value is None for _, value in result_event):
        raise TypeError(
            "In transport_conditional_counterfactual_query: all returned values "
            + "used to evaluate the query result expression should be actual outcome "
            + "variable values, but at least one is None. The result_event is "
            + str(result_event)
            + ". Also check your inputs."
        )
    # 5. Make sure all the variables in the result_event are in the result_expression.
    if not all(variable in result_expression.get_variables() for variable, _ in result_event):
        raise KeyError(
            "In final checks for transport_conditional_counterfactual_query: at least one variable in "
            + "the event that transport_unconditional_counterfactual_query() "
            + "will return is not a variable in the expression for the probability of the query "
            + "that is to be returned. result_event: "
            + str(result_event)
            + " and return expression: "
            + str(result_expression)
        )


def _transport_conditional_counterfactual_query_line_4(
    *,
    outcome_variable_ancestral_component_variable_names: set[Variable],
    outcome_and_conditioned_variable_names: set[Variable],
    conditioned_variable_names: set[Variable],
    transported_unconditional_query_expression: Expression,
    simplified_event: Event,
    outcome_and_conditioned_variable_names_to_values: dict[Variable, set[Intervention]],
    outcomes: list[tuple[Variable, Intervention]],
    conditions: list[tuple[Variable, Intervention]],
    domain_data: list[tuple[Collection[Variable], PopulationProbability]],
) -> "ConditionalCFTResult":
    r"""Execute Line 4 of Algorithm 3 of [correa22a]_.

    This function is an internal subroutine for transport_conditional_counterfactual_query().

    :param outcome_variable_ancestral_component_variable_names: The graph vertices associated with ancestral
        components containing at least one outcome variable in the query, represented as a set of Variable objects.
    :param outcome_and_conditioned_variable_names: The graph vertices associated with outcome and conditioned variables
        in the query (stripped of any interventions), represented as a set of Variable objects.
    :param conditioned_variable_names: The graph vertices associated with conditioned variables
        in the query (stripped of any interventions), represented as a set of Variable objects.
    :param transported_unconditional_query_expression: the probabilistic expression returned by the call to Algorithm
        2 of [correa22a]_ and represented by $Q$ in the pseudocode for Algorithm 3 of [correa22a]_.
    :param simplified_event:
        This is the set of variables $D_{\ast}$ (in counterfactual factor form) and their values $d_{\ast}$
        after getting processed by the simplify() algorithm (Algorithm 1 in [correa22a]_, called from Line 1 in
        Algorithm 2 of [correa22a]_. ) We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param outcome_and_conditioned_variable_names_to_values: a dictionary mapping those variables to their values
        from the query passed in to transport_conditional_counterfactual_query() as a parameter.
    :param outcomes:
        "Y_*, a set of counterfactual variables in V and y_* a set of
        values for Y_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param conditions:
        "X_*, a set of counterfactual variables in V and x_* a set of
        values for X_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param domain_data: Corresponding to $\mathcal{Z}$ in [correa22a]_, this is a set of
           $K$ tuples, one for each of the $K$ domains except for the target domain.
           Each tuple contains a set of variables corresponding to
           $\sigma_{\mathbf{Z}_{k}}$ and an expression denoting the probability distribution
           $P^{k}(\mathbf{V};\sigma_{\mathbf{Z}\_{j}})|{\mathbf{Z}_{j}} \in \mathcal{Z}^{i}$.
           Passed in to transport_conditional_counterfactual_query() as an input parameter.
    :returns: a tuple containing an expression representing the query result, and a list of
           variables and their values used to evaluate the expression. Per Line 4 of Algorithm 4 of
           [correa22a]_, the return expression has the following form:
        :math: $\frac{\Sigma_{\mathbf{d_{\ast}}\backslash(\mathbf{y_{\ast}} \cup \mathbf{x_{\ast}})}{Q}}
                     {\Sigma_{\mathbf{d_{\ast}}\backslash\mathbf{x_{\ast}}}{Q}}$
    """
    # Line 4: compute the expression to return
    # $\mathbf{d_{\ast}} \backslash (\mathbf{y_{\ast}}\cup\mathbf{x_{\ast}})}$
    outcome_ancestral_component_variables_with_no_values: set[Variable] = (
        outcome_variable_ancestral_component_variable_names - outcome_and_conditioned_variable_names
    )
    # $\mathbf{d_{\ast}} \backslash \mathbf{x_{\ast}}$
    outcome_ancestral_component_variable_names_excluding_outcomes = (
        outcome_variable_ancestral_component_variable_names - conditioned_variable_names
    )
    result_expression: Expression = Fraction(
        Sum.safe(
            transported_unconditional_query_expression,
            outcome_ancestral_component_variables_with_no_values,
        ),
        Sum.safe(
            transported_unconditional_query_expression,
            outcome_ancestral_component_variable_names_excluding_outcomes,
        ),
    )
    logger.debug(
        "In transport_conditional_counterfactual_query: result_expression = "
        + result_expression.to_latex()
    )
    # The input outcome and condition variables and their values, to be used
    # to evaluate the return expression.
    result_event: list[tuple[Variable, Intervention]] = [
        (variable.get_base(), value) for variable, value in itt.chain(outcomes, conditions)
    ]
    _validate_transport_conditional_counterfactual_query_line_4_output(
        simplified_event=simplified_event,
        outcome_and_conditioned_variable_names=outcome_and_conditioned_variable_names,
        outcome_and_conditioned_variable_names_to_values=outcome_and_conditioned_variable_names_to_values,
        outcome_ancestral_component_variables_with_no_values=outcome_ancestral_component_variables_with_no_values,
        result_expression=result_expression,
        result_event=result_event,
        domain_data=domain_data,
    )
    return ConditionalCFTResult(expression=result_expression, event=result_event)


class ConditionalCFTResult(NamedTuple):
    """Represents the result of a conditional counterfactual transportability query."""

    expression: Expression
    event: list[tuple[Variable, Intervention]] | None

    def display(self) -> None:
        """Display this result."""
        from IPython.display import display

        display(event_to_probability(self.event))  # type:ignore
        display(self.expression)


def conditional_cft(
    *,
    outcomes: Variable | list[Variable],
    conditions: Variable | list[Variable],
    target_domain_graph: NxMixedGraph,
    domains: list[CFTDomain],
) -> ConditionalCFTResult | None:
    r"""Run a conditional counterfactual transportability query (Algorithm 3 from [correa22a]_).

    :param outcomes:
        "Y_*, a set of counterfactual variables in V and y_* a set of
        values for Y_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param conditions:
        "X_*, a set of counterfactual variables in V and x_* a set of
        values for X_*." We encode the counterfactual variables as
        CounterfactualVariable objects, and the values as Intervention objects.
    :param target_domain_graph: a graph for the target domain.
    :param domains: A set of $K$ CFTDomain classes, one for each of the $K$ domains. See the
        documentation for CFTDomain for more details.

    :returns: The result of the query as a ConditionalCFTResult object.
    """
    domain_graphs = [
        (domain.graph, domain.ordering or domain.graph.topological_sort()) for domain in domains
    ]
    domain_data = [(domain.policy_variables, domain.population) for domain in domains]
    return transport_conditional_counterfactual_query(
        outcomes=_event_from_counterfactuals_strict(outcomes),
        conditions=_event_from_counterfactuals_strict(conditions),
        target_domain_graph=target_domain_graph,
        domain_graphs=domain_graphs,
        domain_data=domain_data,
    )


def transport_conditional_counterfactual_query(
    *,
    outcomes: list[tuple[Variable, Intervention]],
    conditions: list[tuple[Variable, Intervention]],
    target_domain_graph: NxMixedGraph,
    domain_graphs: list[tuple[NxMixedGraph, list[Variable]]],
    domain_data: list[tuple[Collection[Variable], PopulationProbability]],
) -> ConditionalCFTResult | None:
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
    :returns: FAIL (None) if the algorithm fails, or a probabilistic expression for
           $P^{\ast}(\mathbf{Y_{\ast}}=\mathbf{y_{\ast}} | \mathbf{X_{\ast}}=\mathbf{x_{\ast}}$
           along a mapping of variables to values used to evaluate that expression.
           If that expression evaluated to 0 because input values of some counterfactual
           variables were not consistent, then the algorithm returns Zero (a DSL Expression type)
           and no mapping.
    """
    _validate_transport_conditional_counterfactual_query_input(
        outcomes=outcomes,
        conditions=conditions,
        target_domain_graph=target_domain_graph,
        domain_graphs=domain_graphs,
        domain_data=domain_data,
    )

    # Initialize data structures
    (
        conditioned_variables,
        outcome_variables,
        outcome_and_conditioned_variables,
        outcome_variable_to_value_mappings,
        # outcome_and_conditioned_variable_to_value_mappings,
        outcome_and_conditioned_variable_names_to_values,
        outcome_and_conditioned_variable_names,
        conditioned_variable_names,
    ) = _initialize_conditional_transportability_data_structures(
        outcomes=outcomes, conditions=conditions
    )

    # Line 1: compute $\mathbf{A_{\ast}}$
    ancestral_components: frozenset[frozenset[Variable]] = get_ancestral_components(
        conditioned_variables=conditioned_variables,
        root_variables=outcome_and_conditioned_variables,
        graph=target_domain_graph,
    )

    # Line 2: compute $\mathbf{D_{\ast}}$ and $\mathbf{d_{\ast}}$
    (
        outcome_ancestral_component_query_in_counterfactual_factor_form,
        outcome_variable_ancestral_component_variable_names,
    ) = _transport_conditional_counterfactual_query_line_2(
        ancestral_components=ancestral_components,
        outcome_variables=outcome_variables,
        outcome_variable_to_value_mappings=outcome_variable_to_value_mappings,
        target_domain_graph=target_domain_graph,
    )

    # Line 3
    unconditional_query_result = transport_unconditional_counterfactual_query(
        event=outcome_ancestral_component_query_in_counterfactual_factor_form,
        target_domain_graph=target_domain_graph,
        domain_graphs=domain_graphs,
        domain_data=domain_data,
    )
    if unconditional_query_result is None:
        # Technically the logic of possibly returning FAIL if Algorithm 2 of [correa22a]_ returns FAIL is
        # not in the published version of Algorithm 3, but it's implied by Algorithm 2's return values
        return None

    transported_unconditional_query_expression = unconditional_query_result.expression  # This is Q
    simplified_event: Event | None = unconditional_query_result.event

    # Event has probability of zero due to inconsistent values in the query
    if simplified_event is None:
        return ConditionalCFTResult(
            expression=transported_unconditional_query_expression, event=simplified_event
        )
    else:
        # Line 4: compute the expression to return
        return _transport_conditional_counterfactual_query_line_4(
            outcome_variable_ancestral_component_variable_names=outcome_variable_ancestral_component_variable_names,
            outcome_and_conditioned_variable_names=outcome_and_conditioned_variable_names,
            conditioned_variable_names=conditioned_variable_names,
            transported_unconditional_query_expression=transported_unconditional_query_expression,
            simplified_event=simplified_event,
            outcome_and_conditioned_variable_names_to_values=outcome_and_conditioned_variable_names_to_values,
            outcomes=outcomes,
            conditions=conditions,
            domain_data=domain_data,
        )


def _validate_transport_conditional_counterfactual_query_input(  # noqa:C901
    outcomes: list[tuple[Variable, Intervention]],
    conditions: list[tuple[Variable, Intervention]],
    target_domain_graph: NxMixedGraph,
    domain_graphs: list[tuple[NxMixedGraph, list[Variable]]],
    domain_data: list[tuple[Collection[Variable], PopulationProbability]],
) -> None:
    r"""Conduct pre-processing checks to transport conditional counterfacutal queries (Algorithm 3 from [correa22a]_).

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
    :raises TypeError: a validity check associated with either the input variables
           or the output event failed. See the error message for specifics.
    :raises ValueError: an input variable is of valid type but has an invalid value. See
           the error message for specifics.
    :raises NotImplementedError: this algorithm does not currently handle input graph probability
           Expression objects that are One() or Zero(), or cases where the conditioned and
           outcome variable sets share one or more variables in common.
    """
    # Here are all the checks (numbering is just based on convenience during implementation, and
    #    the numbered order is not necessarily the order of implementation):
    # 1. Type checking for outcomes and conditions
    # 2. Type checking for target_domain_graph
    # 3. Type checking for domain_graphs
    # 4. Type checking for domain_data
    #    4.5. Make sure probabilistic expressions in domain_data aren't Zero() or One()
    # 5. Make sure conditions and outcomes aren't empty
    # 6. (Skipped for the conditional transportability algorithm, included for unconditional
    #    transportability) Make sure at least one event element has a non-None value
    # 7. Check domain_graphs and domain_data aren't empty lists
    # 8. Check all graphs in domain_graphs have nodes
    # 9. Check all topologically sorted lists have entries
    # 9.2. Check that the target domain graph contains no transportability nodes and is a directed acyclic graph
    # 9.5. Check that the domain_graphs and domain_data list lengths are equal
    # 9.7. Check that every domain graph is a directed acyclic graph
    # 10. Check that every topological order list in domain_graphs is a valid topological order,
    #     given the corresponding graph
    # 11. Check the domain graph vertices are all the same as the target domain graph vertices,
    #     net of transportability nodes
    # 12. Check the event vertices are in the target domain graph (given check #11, that
    #     means they're in every graph)
    # 13. Check the conditioned and outcome variables have the same base variable
    #     as the base variable of their corresponding values
    # 14. Domain graphs: make sure the vertex set of the topologically sorted vertex order matches
    #     the set of vertices in each corresponding domain graph
    # 15. It's possible for a graph probability expression to contain vertices not in the graph
    #     due to conditioning on vertices outside the graph. But the graph vertices must all be
    #     represented in that graph probability expression.
    # 15.5. All the policy vertices must be in the graph, for each domain.
    # 16. If the target domain graph is also in the domain_graphs list (i.e., data were collected for
    #     the target domain), then the target domain graph in the domain_graphs list must be
    #     identical to the target_domain_graph parameter.
    # 17. Make sure the conditioned variable and outcome variable sets don't share graph vertices

    # Preliminary checks, starting with type checking for inputs unique to Algorithm 3
    # 1.
    if not (
        isinstance(outcomes, list) and all(isinstance(t, tuple) and len(t) == 2 for t in outcomes)
    ):
        raise TypeError(
            "In _validate_transport_conditional_counterfactual_query_input: the input outcomes "
            + "must be a list of tuples of length 2. Check your inputs."
        )
    if not all(
        isinstance(variable, Variable) and isinstance(value, Intervention)
        for variable, value in outcomes
    ):
        raise TypeError(
            "In _validate_transport_conditional_counterfactual_query_input: each tuple in the input outcomes "
            + "must contain a Variable object and its corresponding value (an Intervention). Check your inputs."
        )
    if not (
        isinstance(conditions, list)
        and all(isinstance(t, tuple) and len(t) == 2 for t in conditions)
    ):
        raise TypeError(
            "In _validate_transport_conditional_counterfactual_query_input: the input conditions "
            + "must be a list of tuples of length 2. Check your inputs."
        )
    if not all(
        isinstance(variable, Variable) and isinstance(value, Intervention)
        for variable, value in conditions
    ):
        raise TypeError(
            "In _validate_transport_conditional_counterfactual_query_input: each tuple in the input conditions "
            + "must contain a Variable object and its corresponding value (an Intervention). Check your inputs."
        )
    # Type checking for inputs consistent with both Algorithms 2 and 3
    # 2.
    if not isinstance(target_domain_graph, NxMixedGraph):
        raise TypeError(
            "In _validate_transport_conditional_counterfactual_query_input: the target_domain_graph "
            + "must be an NxMixedGraph object."
        )

    # Check we have no empty inputs, for inputs unique to Algorithm 3
    # 5.
    if len(conditions) == 0:
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: empty list for "
            + "the conditions. Check your inputs or consider directly calling "
            + "transport_unconditional_counterfactual_query()."
        )
    if len(outcomes) == 0:
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: empty list for "
            + "the outcomes. Check your inputs."
        )
    if len(target_domain_graph.nodes()) == 0:
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: the target "
            + "domain graph contained no nodes. Check your inputs."
        )

    # Type checking for inputs consistent with Algorithms 2,3, and 4
    # 3.
    if not (isinstance(domain_graphs, list) and all(isinstance(t, tuple) for t in domain_graphs)):
        raise TypeError(
            "In _validate_transport_conditional_counterfactual_query_input: the "
            + "domain_graphs input parameter must be a list of tuples."
        )
    if not all(
        isinstance(g, NxMixedGraph)
        and isinstance(l_variables, list)
        and all(isinstance(v, Variable) for v in l_variables)
        for g, l_variables in domain_graphs
    ):
        raise TypeError(
            "In _validate_transport_conditional_counterfactual_query_input: the input domain "
            + "graph tuples must all contain NxMixedGraph objects and lists of variables."
        )
    # 4 and 4.5.
    if not (isinstance(domain_data, list) and all(isinstance(t, tuple) for t in domain_data)):
        raise TypeError(
            "In _validate_transport_conditional_counterfactual_query_input: the "
            + "input domain data must be a list of tuples."
        )
    if any(e == Zero() or e == One() for _, e in domain_data):
        raise NotImplementedError(
            "In _validate_transport_conditional_counterfactual_query_input: this algorithm "
            + "does not currently handle domain_data probability expressions that are of type "
            + "One() or Zero()."
        )
    if not all(
        isinstance(sigma_z, Collection)
        and all(isinstance(v, Variable) for v in sigma_z)
        and isinstance(pp, PopulationProbability)
        for sigma_z, pp in domain_data
    ):
        raise TypeError(
            "In _validate_transport_conditional_counterfactual_query_input: the input "
            + "domain data tuples must all contain Collections of Variable objects "
            + "(first element) and PopulationProbability expressions (second element)."
        )

    # 17.
    # conditioned_variables = {variable.get_base() for variable, _ in conditions}
    # outcome_variables = {variable.get_base() for variable, _ in outcomes}
    # if conditioned_variables.intersection(outcome_variables) != set():
    #    raise NotImplementedError(
    #        "In _validate_transport_conditional_counterfactual_query_input: currently this "
    #        + "algorithm does not allow for the conditioned and outcome variables to share common"
    #        + " vertices. Overlapping graph vertices (i.e., without their interventions): "
    #        + str(conditioned_variables.intersection(outcome_variables))
    #    )

    # Check we have no empty inputs (Algorithms 2, 3, and 4)
    # 7.
    if len(domain_graphs) == 0 or len(domain_data) == 0:
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: empty list for "
            + "either domain_graphs or domain_data. Check your inputs."
        )
    # 8.
    if any(len(g.nodes()) == 0 for g, _ in domain_graphs):
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: at least one input "
            + "domain graph contained no nodes. Check your inputs."
        )
    # 9.
    if any(len(topo) == 0 for _, topo in domain_graphs):
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: an input set of "
            + "topologically sorted vertices was empty. Check your inputs."
        )
    # 9.5.
    if len(domain_graphs) != len(domain_data):
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: the length of the "
            + "domain_graphs and domain_data must be the same."
        )

    # Check the target domain graph contains no transportability nodes and is a directed acyclic graph
    # 9.2.
    if frozenset(target_domain_graph.nodes()) != frozenset(
        _remove_transportability_vertices(vertices=target_domain_graph.nodes())
    ):
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: the target domain graph "
            + "cannot contain a transportability node. Check your inputs."
        )
    if not is_directed_acyclic_graph(target_domain_graph.directed):
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: the directed edges in "
            + "the target domain graph cannot form a cycle. Check your inputs."
        )
    # Check the domain graph vertices are all the same as the target domain graph vertices,
    #    net of transportability nodes
    # 11.
    if any(
        frozenset(target_domain_graph.nodes())
        != frozenset(_remove_transportability_vertices(vertices=domain_graph.nodes()))
        for domain_graph, _ in domain_graphs
    ):
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: a domain graph contained"
            + " different vertices than the target domain graph after excluding transportability "
            + "nodes. Check your inputs."
        )
    # Check the event vertices are in the target domain graph (given the above check, that means they're in every graph)
    # 12.
    if any(variable.get_base() not in target_domain_graph.nodes() for variable, _ in conditions):
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: one of the input "
            + "conditioned variables is not in the target domain graph. Check your inputs. "
        )
    if any(variable.get_base() not in target_domain_graph.nodes() for variable, _ in outcomes):
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: one of the input "
            + "outcome variables is not in the target domain graph. Check your inputs. "
        )
    # 13.
    if any(variable.get_base() != value.get_base() for variable, value in conditions):
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: one of the input "
            + "conditioned variables does not have the same base variable as its corresponding"
            + "value (e.g., your variable is (W @ -X) and its value must be +W or -W, but it's"
            + "-X). Check your inputs."
        )
    if any(variable.get_base() != value.get_base() for variable, value in outcomes):
        raise ValueError(
            "In _validate_transport_conditional_counterfactual_query_input: one of the input "
            + "outcome variables does not have the same base variable as its corresponding"
            + "value (e.g., your variable is (W @ -X) and its value must be +W or -W, but the "
            + "value is -X). Check your inputs."
        )

    # Technically the topologically sorted vertices could be for a superset of the vertices
    # in the input graphs, but we currently require them to be for the vertices in the input graphs.
    for k in range(len(domain_graphs)):
        # logger.debug("k = " + str(k))
        topo_vertices = frozenset(domain_graphs[k][1])
        expression_vertices = frozenset(domain_data[k][1].get_variables())
        graph_vertices = frozenset(domain_graphs[k][0].nodes())
        graph_vertices_without_transportability_nodes = frozenset(
            _remove_transportability_vertices(vertices=graph_vertices)
        )
        policy_vertices = frozenset(domain_data[k][0])
        # 14.
        if topo_vertices != graph_vertices:
            raise ValueError(
                "In _validate_transport_conditional_counterfactual_query_input: the vertices "
                + "in each domain graph must match those in the "
                + "corresponding topologically sorted list of vertices. Check your inputs and "
                + "note that the topologically sorted vertex lists "
                + "must contain any transportability nodes. An easy way to "
                + "generate a usable list is to call "
                + "[graph_name].topological_sort() on the graph. "
                + "Graph vertices: "
                + str(graph_vertices)
                + ". Topologically sorted list of vertices: "
                + str(topo_vertices)
                + ". Domain index (zero-indexed): "
                + str(k)
            )
        # It's possible for a graph probability expression to contain vertices not in the graph
        # due to conditioning on vertices outside the graph. But the graph vertices must all be
        # represented in that graph probability expression.
        # 15.
        if not all(v in expression_vertices for v in graph_vertices_without_transportability_nodes):
            raise ValueError(
                "In _validate_transport_conditional_counterfactual_query_input: some of the "
                + "vertices in a domain graph do not appear in the expression"
                + " for the probability of the graph. Check your inputs. Graph vertices: "
                + str(graph_vertices_without_transportability_nodes)
                + ". Expression vertices: "
                + str(expression_vertices)
                + ". Domain index (zero-indexed): "
                + str(k)
            )
        # 15.5.
        if not all(v in graph_vertices_without_transportability_nodes for v in policy_vertices):
            raise ValueError(
                "In _validate_transport_conditional_counterfactual_query_input: the set of "
                + "vertices for which a policy has been applied for one "
                + "of the domains contains at least one vertex not in the domain graph. Check your inputs. "
                + "Policy vertices: "
                + str(policy_vertices)
                + ". Graph vertices: "
                + str(graph_vertices_without_transportability_nodes)
                + ". Domain index (zero-indexed): "
                + str(k)
            )
        # 9.7. (The directed acyclic graph check must come before the topological order check)
        if not is_directed_acyclic_graph(domain_graphs[k][0].directed):
            raise ValueError(
                "In _validate_transport_conditional_counterfactual_query_input: the directed edges in "
                + "domain graph entry "
                + str(k)
                + " (zero-indexed) form a cycle and the graph must be a "
                + "directed acyclic graph. Check your inputs."
            )
        # 10.
        if not _valid_topo_list(topo=domain_graphs[k][1], graph=domain_graphs[k][0]):
            raise ValueError(
                "In _validate_transport_conditional_counterfactual_query_input: the provided topologically "
                + "sorted order of the vertices ("
                + str(domain_graphs[k][1])
                + ") for domain graph entry "
                + str(k)
                + " (zero-indexed) "
                + "is not valid, given the input domain_graph. Check your inputs."
            )
        # 16. If the target domain graph is also in the domain_graphs list (i.e., data were collected for
        #     the target domain), then the target domain graph in the domain_graphs list must be
        #     identical to the target_domain_graph parameter.
        # TODO: relax the code base to allow Expressions in the domain_data instead of
        #     PopulationProbability types as follows:
        # if isinstance(domain_data[k][1], PopulationProbability) and
        #    str(domain_data[k][1].population)==str(TARGET_DOMAIN):
        #    That covers a corner case where a user wishes to run this algorithm with a single domain and
        #    without specifying a population for the graph probability. In that case the problem reduces
        #    to running ID* anyway, so it's not likely to get much usage.
        if str(domain_data[k][1].population) == str(TARGET_DOMAIN):
            if domain_graphs[k][0] != target_domain_graph:
                raise ValueError(
                    "In _validate_transport_conditional_counterfactual_query_input: the domain_data contain "
                    + 'a graph probability expression from the target domain (i.e., "pi*"), but the '
                    + "corresponding domain_graph is not the same graph as the target_domain_graph. "
                    + "Check your inputs. Domain index (zero-indexed): "
                    + str(k)
                )
    return


def _valid_topo_list(topo: list[Variable], graph: NxMixedGraph) -> bool:
    r"""Verify that a list of vertices is in topologically sorted order for a given graph.

    :param topo: A candidate list of graph vertices. This function assumes every vertex in topo is somewhere
        in the graph and every graph vertex is in topo (that is, this information has already been verified).
    :param graph: The graph in question.
    :returns: True if the list is in a valid topologically sorted order, False otherwise.

    From Cormen, Leiserson, Rivest, and Stein, "Introduction to Algorithms", second edition, p. 549:
    a graph is sorted topologically if for every directed edge in the graph from $u$ to $v$,
    the index of $u$ in the topologically sorted array of vertices is less than the index of $v$.
    We just have to check this definition holds for each edge.
    $O(E*V)$ if we use :func:`list.index` to do this. But :func:`list.index` is $O(n)$,
    so we improve the running time to $O(E+V)$ by creating a hash table first.
    """
    node_to_index = {node: index for index, node in enumerate(topo)}
    return not any(node_to_index[u] > node_to_index[v] for u, v in graph.directed.edges)
