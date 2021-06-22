from .id_std import identify
from ..conditional_independencies import are_d_separated


def idx(outcomes, treatments, conditions, graph):
    if idc_condition():
        ...
        return idx(...)

    # Run ID algorithm
    _ = identify(...)
    return ...


def idc_condition(graph, conditions, outcomes):
    for condition in conditions:
        if are_d_separated(outcomes, ...):
            ...
