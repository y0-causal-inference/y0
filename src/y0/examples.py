# -*- coding: utf-8 -*-

"""Examples from CausalFusion."""

from .graph import NxMixedGraph

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
#: Journal, Volume 10, No. 10, pp. 37-48, 1999.
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

# TODO identifiability_2

#: The Identifiability 3 example
#: Treatment: X
#: Outcome: Y
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 92.
identifiability_3 = NxMixedGraph.from_edges(
    [
        ('Z2', 'X'),
        ('Z2', 'Z1'),
        ('Z2', 'Z3'),
        ('X', 'Z1'),
        ('Z3', 'Y'),
        ('Z1', 'Y'),
    ],
    [
        ('Z2', 'X'),
        ('Z2', 'Y'),
        ('X', 'Z3'),
        ('X', 'Y'),
    ],
)

#: The Identifiability 4 example
#: Treatment: X
#: Outcome: Y
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 92.
identifiability_4 = NxMixedGraph.from_edges(
    [
        ('X', 'Z1'),
        ('X', 'Y'),
        ('Z1', 'Z2'),
        ('Z1', 'Y'),
        ('Z2', 'Y'),
    ],
    [
        ('X', 'Z2'),
        ('Z1', 'Y'),
    ],
)

#: The Identifiability 5 example
#: Treatment: X1, X2
#: Outcome: Y
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 119.
identifiability_5 = NxMixedGraph.from_edges(
    [
        ('X1', 'Z'),
        ('X1', 'Y'),
        ('X1', 'X2'),
        ('Z', 'X2'),
        ('X2', 'Y'),
    ],
    [
        ('X1', 'Z'),
        ('Z', 'Y'),
    ],
)

#: The Identifiability 6 example
#: Treatment: X1, X2
#: Outcome: Y
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 125.
identifiability_6 = NxMixedGraph.from_edges(
    [
        ('Z1', 'X1'),
        ('X1', 'X2'),
        ('X2', 'Y'),
        ('Z2', 'Y'),
    ],
    [
        ('Z1', 'Z2'),
        ('Z1', 'X2'),
        ('Z2', 'X2'),
    ],
)

#: The Identifiability 7 example
#: Treatment: X
#: Outcome: Y
#: Reference: J. Tian. 2002. "Studies in Causal Reasoning and Learning." p. 90.
identifiability_6 = NxMixedGraph.from_edges(
    [
        ('W1', 'W2'),
        ('W3', 'W4'),
        ('W2', 'X'),
        ('W4', 'X'),
        ('X', 'Y'),
    ],
    [
        ('W1', 'X'),
        ('W1', 'Y'),
        ('W1', 'W3'),
        ('W3', 'W2'),
        ('W3', 'W5'),
        ('W5', 'W4'),
    ],
)

# TODO Recoverability 1/2 - what is the S node?
# TODO Transportability 1/2 - what are the box nodes?
# TODO g-Identifiability examples
# TODO g-Transportability examples


#: The Verma 1 example
#: Treatment: V3
#: Outcome: V4
#: Reference: T. Verma and J. Pearl. 1990. "Equivalence and Synthesis of Causal Models." In P. Bonissone et al., eds.,
#: Proceedings of the 6th Conference on Uncertainty in Artificial Intelligence. Cambridge, MA: AUAI Press, p. 257.
verma_1 = NxMixedGraph.from_edges(
    [
        ('V1', 'V2'),
        ('V2', 'V3'),
        ('V3', 'V4'),
    ],
    [
        ('V2', 'V4'),
    ],
)

#: The Verma 2 example
#: Treatment: V1
#: Outcome: V5
#: Reference: J. Tian. 2002. "Studies in Causal Reasoning and Learning." p. 70.
verma_2 = NxMixedGraph.from_edges(
    [
        ('V1', 'V2'),
        ('V2', 'V3'),
        ('V3', 'V4'),
        ('V4', 'V5'),
    ],
    [
        ('V1', 'V3'),
        ('V2', 'V4'),
        ('V3', 'V5'),
    ],
)

#: The Verma 3 example
#: Treatment: V1
#: Outcome: V5
#: Reference: J. Tian. 2002. "Studies in Causal Reasoning and Learning." p. 59.
verma_3 = NxMixedGraph.from_edges(
    [
        ('V1', 'V2'),
        ('V2', 'V3'),
        ('V3', 'V4'),
        ('V4', 'V5'),
    ],
    [
        ('V1', 'V5'),
        ('V1', 'V3'),
        ('V2', 'V4'),
    ],
)

#: The Verma 4 example
#: Treatment: V1
#: Outcome: V5
#: Reference: E. Bareinboim modification of Verma 2.
verma_4 = NxMixedGraph.from_edges(
    [
        ('V1', 'V2'),
        ('V2', 'V3'),
        ('V3', 'V4'),
        ('V4', 'V5'),
    ],
    [
        ('V1', 'V5'),
        ('V1', 'V3'),
        ('V2', 'V4'),
        ('V3', 'V5'),
    ],
)

#: The Verma 5 example
#: Treatment: V1
#: Outcome: V5
#: Reference: E. Bareinboim modification of Verma 2.
verma_5 = NxMixedGraph.from_edges(
    [
        ('V1', 'V2'),
        ('V2', 'V3'),
        ('V3', 'V4'),
        ('V4', 'V5'),
        ('V5', 'V6'),
    ],
    [
        ('V0', 'V1'),
        ('V0', 'V6'),
        ('V1', 'V5'),
        ('V1', 'V3'),
        ('V2', 'V4'),
    ],
)

#: The z-Identifiability 1 example
#: Treatment: X
#: Outcome: Y
#: Z*: Z
#: Reference: E. Bareinboim and J. Pearl. 2012. "Causal Inference by Surrogate Experiments: z-Identifiability." In
#: Nando de Freitas and K. Murphy., eds., Proceedings of the 28th Conference on Uncertainty in Artificial Intelligence.
#: Corvallis, OR: AUAI Press, p. 114.
z_identifiability_1 = NxMixedGraph.from_edges(
    [
        ('Z', 'X'),
        ('X', 'Y'),
    ],
    [
        ('Z', 'X'),
        ('Z', 'Y'),
    ],
)

#: The z-Identifiability 2 example
#: Treatment: X
#: Outcome: Y
#: Z*: Z
#: Reference: E. Bareinboim and J. Pearl. 2012. "Causal Inference by Surrogate Experiments: z-Identifiability." In
#: Nando de Freitas and K. Murphy., eds., Proceedings of the 28th Conference on Uncertainty in Artificial Intelligence.
#: Corvallis, OR: AUAI Press, p. 114.
z_identifiability_2 = NxMixedGraph.from_edges(
    [
        ('Z', 'X'),
        ('X', 'Y'),
    ],
    [
        ('X', 'Y'),
        ('Z', 'Y'),
    ],
)

#: The z-Identifiability 3 example
#: Treatment: X
#: Outcome: Y
#: Z*: Z
#: Reference: E. Bareinboim and J. Pearl. 2012. "Causal Inference by Surrogate Experiments: z-Identifiability." In
#: Nando de Freitas and K. Murphy., eds., Proceedings of the 28th Conference on Uncertainty in Artificial Intelligence.
#: Corvallis, OR: AUAI Press, p. 114.
z_identifiability_3 = NxMixedGraph.from_edges(
    [
        ('Z', 'Y'),
        ('X', 'Y'),
    ],
    [
        ('X', 'Z'),
        ('Z', 'Y'),
    ],
)

#: The Identifiability (Linear) 1 example
#: Treatment: X
#: Outcome: Y
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 153.
identifiability_linear_1 = NxMixedGraph.from_edges(
    [
        ('X', 'Z'),
        ('X', 'W'),
        ('W', 'Y'),
        ('Z', 'Y'),
    ],
    [
        ('X', 'Z'),
        ('W', 'Y'),
    ],
)
