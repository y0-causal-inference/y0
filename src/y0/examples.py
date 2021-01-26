# -*- coding: utf-8 -*-

"""Examples from CausalFusion."""

from y0.graph import NxMixedGraph

#: The "backdoor" example
#: Treatment: X
#: Outcome: Y
#: Adjusted: N/A
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 178.
backdoor = NxMixedGraph.from_edges([
    ('Z', 'X'),
    ('Z', 'Y'),
    ('X', 'Y'),
])

#: The "frontdoor" example
#: Treatment: X
#: Outcome: Y
#: Adjusted: N/A
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 81.
frontdoor = NxMixedGraph.from_edges(
    [('X', 'Z'), ('Z', 'Y')],
    [('X', 'Y')],
)

#: The Instrument Variable example
#: Treatment: X
#: Outcome: Y
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 153.
instrumental_variable = NxMixedGraph.from_edges(
    [('Z', 'X'), ('X', 'Y')],
    [('X', 'Y')],
)

#: The Napkin example
#: Treatment: X
#: Outcome: Y
#: Reference: J. Pearl and D. Mackenzie. 2018. "The Book of Why: The New Science of Cause and Effect." Basic Books,
#   p. 240.
napkin = NxMixedGraph.from_edges(
    [('Z2', 'Z1'), ('Z1', 'X'), ('X', 'Y')],
    [('Z2', 'X'), ('Z2', 'Y')],
)

#: The M-Graph example
#: Treatment: X
#: Outcome: Y
#: Reference: S. Greenland, J. Pearl, and J.M. Robins. 1999. "Causal Diagrams for Epidemiologic Research." Epidemiology
#:  Journal, Volume 10, No. 10, pp. 37-48, 1999.
m_graph = NxMixedGraph.from_edges(
    [('X', 'Y')],
    [('X', 'Z'), ('Y', 'Z')],
)

#: The Identifiability 1 example
#: Treatment: X
#: Outcome: Y
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 80.
identifiability_1 = NxMixedGraph.from_edges([
    ('Z1', 'Z2'),
    ('Z1', 'Z3'),
    ('Z2', 'X'),
    ('Z3', 'X'),
    ('Z4', 'X'),
    ('Z4', 'Z5'),
    ('Z3', 'Y'),
    ('X', 'Y'),
    ('Z3', 'Y'),
])

#: The Napkin graph
#: Treatment: X
#: Outcome: Y
#: Reference: E. Bareinboim modification of Identifiability 1.
identifiability_2 = NxMixedGraph.from_edges([
    ('Z1', 'Z2'),
    ('Z1', 'Z3'),
    ('Z2', 'X'),
    ('Z3', 'X'),
    ('Z4', 'X'),
    ('Z4', 'Z5'),
    ('Z3', 'Y'),
    ('X', 'Y'),
    ('Z3', 'Y'),
])
