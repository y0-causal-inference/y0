# -*- coding: utf-8 -*-
# type: ignore

"""Examples from CausalFusion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backdoor import generate_data_for_backdoor
from .frontdoor import generate_data_for_frontdoor
from .frontdoor_backdoor import generate_data_for_frontdoor_backdoor
from .sars import generate_data_for_covid_case_study
from .smoke_cancer import generate_data_for_smoke_cancer
from .utils import Example
from ..algorithm.identify import Identification, Query
from ..dsl import (
    AA,
    W0,
    W1,
    W2,
    X1,
    X2,
    Y1,
    Y2,
    Z1,
    Z2,
    Z3,
    Z4,
    Z5,
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    M,
    P,
    Q,
    S,
    Sum,
    T,
    Variable,
    W,
    X,
    Y,
    Z,
)
from ..graph import NxMixedGraph
from ..resources import ASIA_PATH
from ..struct import DSeparationJudgement, VermaConstraint

x, y, z, w = -X, -Y, -Z, -W

u_2 = Variable("u_2")
u_3 = Variable("u_3")

#: Treatment: X
#: Outcome: Y
#: Adjusted: N/A
backdoor = NxMixedGraph.from_edges(
    directed=[
        (Z, X),
        (Z, Y),
        (X, Y),
    ]
)

backdoor_example = Example(
    name="Backdoor",
    reference='J. Pearl. 2009. "Causality: Models, Reasoning and Inference.'
    ' 2nd ed." Cambridge University Press, p. 178.',
    graph=backdoor,
    generate_data=generate_data_for_backdoor,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)

#: Treatment: X
#: Outcome: Y
#: Adjusted: N/A
frontdoor = NxMixedGraph.from_edges(
    directed=[
        (X, Z),
        (Z, Y),
    ],
    undirected=[
        (X, Y),
    ],
)
frontdoor_example = Example(
    name="Frontdoor",
    reference='J. Pearl. 2009. "Causality: Models, Reasoning and Inference.'
    ' 2nd ed." Cambridge University Press, p. 81.',
    graph=frontdoor,
    generate_data=generate_data_for_frontdoor,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)

#: Treatment: X
#: Outcome: Y
#: Adjusted: N/A
frontdoor_backdoor = NxMixedGraph.from_edges(
    directed=[
        (X, Z),
        (Z, Y),
        (W, X),
        (W, Y),
    ],
)
frontdoor_backdoor_example = Example(
    name="Frontdoor / Backdoor",
    reference="https://github.com/y0-causal-inference/y0/pull/183",
    graph=frontdoor_backdoor,
    generate_data=generate_data_for_frontdoor_backdoor,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)

#: Treatment: X
#: Outcome: Y
instrumental_variable = NxMixedGraph.from_edges(
    directed=[
        (Z, X),
        (X, Y),
    ],
    undirected=[
        (X, Y),
    ],
)
instrumental_variable_example = Example(
    name="Instrument Variable",
    reference='J. Pearl. 2009. "Causality: Models, Reasoning and Inference.'
    ' 2nd ed." Cambridge University Press, p. 153.',
    graph=instrumental_variable,
)

#: Treatment: X
#: Outcome: Y
napkin = NxMixedGraph.from_edges(
    directed=[
        (Z2, Z1),
        (Z1, X),
        (X, Y),
    ],
    undirected=[
        (Z2, X),
        (Z2, Y),
    ],
)


def generate_napkin_data(
    num_samples: int, treatments: dict[Variable, float] | None = None, *, seed: int | None = None
) -> pd.DataFrame:
    """Generate testing data for the napkin graph.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable
        to. The keys in this dictionary must correspond to variables in the
        napkin graph as defined in :data:`y0.examples.napkin` (i.e.,
        with :data:`y0.dsl.Z1`, :data:`y0.dsl.Z2`, :data:`y0.dsl.X`,
        and :data:`y0.dsl.Y`).
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding to the four
        variable names in the Napkin graph (i.e., ``Z1``, ``Z2``, ``X``,
        and ``Y``)

    Generate _observational_ data with the following:

    >>> from y0.examples.napkin_example
    >>> napkin_example.generate_data(1000)

    Generate interventional data on $X=1$ with the following:

    >>> from y0.dsl import X
    >>> napkin_example.generate_data(1000, treatments={X: 1})

    Multiple treatments can be specified:

    >>> from y0.dsl import X, Z1
    >>> napkin_example.generate_data(1000, treatments={X: 1, Z1: 0})
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)
    # U1 is the latent variable that is a common cause of W and X
    u1 = generator.normal(loc=3, scale=1, size=num_samples)
    # U2 is the latent variable that is a common cause of W and Y
    u2 = generator.normal(loc=5, scale=1, size=num_samples)
    if Z2 in treatments:
        z2 = np.full(num_samples, treatments[Z2])
    else:
        u_linear_combination = 0.3 * u1 + 0.5 * u2
        z2 = generator.gamma(
            shape=u_linear_combination**-2,
            scale=5 * u_linear_combination,
            size=num_samples,
        )
    if Z1 in treatments:
        z1 = np.full(num_samples, treatments[Z1])
    else:
        z1 = generator.normal(loc=z2 * 0.7, scale=6, size=num_samples)
    if X in treatments:
        x = np.full(num_samples, treatments[X])
    else:
        x = generator.binomial(n=1, p=1 / (1 + np.exp(-2 - 0.23 * u1 - 0.1 * z1)), size=num_samples)
    if Y in treatments:
        y = np.full(num_samples, treatments[Y])
    else:
        y = generator.normal(loc=u2 * 0.5 + x * 3, scale=6)
    return pd.DataFrame({Z2.name: z2, Z1.name: z1, X.name: x, Y.name: y})


napkin_example = Example(
    name="Napkin",
    reference='J. Pearl and D. Mackenzie. 2018. "The Book of Why: The New Science of Cause and Effect."'
    " Basic Books, p. 240.",
    graph=napkin,
    generate_data=generate_napkin_data,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
    verma_constraints=[
        VermaConstraint(
            lhs_cfactor=Q[X, Y](Z1, X, Y) / Sum[Y](Q[X, Y](Z1, X, Y)),
            lhs_expr=(
                Sum[Z2](P(Y | (Z1, Z2, X)) * P(X | (Z2, Z1)) * P(Z2))
                / Sum[Z2, Y](P(Y | (Z2, Z1, X)) * P(X | (Z2, Z1)) * P(Z2))
            ),
            rhs_cfactor=Q[Y](X, Y),
            rhs_expr=Sum[u_2, X](P(Y | u_2 | X) * P(X) * P(u_2)),
            variables=(Z1,),
        ),
    ],
)

#: Treatment: X
#: Outcome: Y
#: Reference:
m_graph = NxMixedGraph.from_edges(
    directed=[
        (X, Y),
    ],
    undirected=[
        (X, Z),
        (Y, Z),
    ],
)
m_graph_example = Example(
    name="M-Graph",
    reference='S. Greenland, J. Pearl, and J.M. Robins. 1999. "Causal Diagrams for Epidemiologic Research."'
    " Epidemiology Journal, Volume 10, No. 10, pp. 37-48, 1999.",
    graph=m_graph,
)

# NxMixedGraph containing vertices without edges
vertices_without_edges = Example(
    name="Vertices-without-Edges",
    reference="out of the mind of JZ (patent pending). See NFT for details",
    graph=NxMixedGraph.from_adj(
        directed={W: [], X: [Y], Y: [Z], Z: []},
        undirected={W: [], X: [Z], Y: [], Z: []},
    ),
)

# Line 1 example
line_1_example = Example(
    name="Line 1 of ID algorithm",
    reference="out of the mind of JZ",
    graph=NxMixedGraph.from_edges(
        directed=[
            (Z, Y),
        ]
    ),
    identifications=[
        dict(
            id_in=[
                Identification.from_expression(
                    query=P(Y),
                    estimand=P(Y, Z),
                    graph=NxMixedGraph.from_edges(directed=[(Z, Y)]),
                )
            ],
            id_out=[
                Identification.from_expression(
                    query=P(Y),
                    estimand=Sum[Z](P(Y, Z)),
                    graph=NxMixedGraph.from_edges(directed=[(Z, Y)]),
                )
            ],
        ),
        dict(
            id_in=[
                Identification.from_expression(
                    query=P(Y, Z),
                    estimand=P(Y, Z),
                    graph=NxMixedGraph.from_edges(directed=[(Z, Y)]),
                )
            ],
            id_out=[
                Identification.from_expression(
                    query=P(Y, Z),
                    estimand=P(Y, Z),
                    graph=NxMixedGraph.from_edges(directed=[(Z, Y)]),
                )
            ],
        ),
    ],
)

# Line 2 example
line_2_example = Example(
    name="intervention not ancestral to outcome",
    reference="out of the mind of JZ",
    graph=NxMixedGraph.from_edges(directed=[(Z, Y), (Y, X)], undirected=[(Z, X)]),
    identifications=[
        dict(
            id_in=[
                Identification.from_expression(
                    query=P(Y @ X),
                    estimand=P(X, Y, Z),
                    graph=NxMixedGraph.from_edges(directed=[(Z, Y), (Y, X)], undirected=[(Z, X)]),
                )
            ],
            id_out=[
                Identification.from_expression(
                    query=P(Y),
                    estimand=Sum[X](P(Y, X, Z)),
                    graph=NxMixedGraph.from_edges(directed=[(Z, Y)]),
                )
            ],
        )
    ],
)

line_3_example = Example(
    name="node has no effect on outcome",
    reference="out of the mind of JZ",
    graph=NxMixedGraph.from_edges(directed=[(Z, X), (X, Y)], undirected=[(Z, X)]),
    identifications=[
        dict(
            id_in=[
                Identification.from_expression(
                    query=P(Y @ X),
                    estimand=P(X, Y, Z),
                    graph=NxMixedGraph.from_edges(directed=[(Z, X), (X, Y)], undirected=[(Z, X)]),
                )
            ],
            id_out=[
                Identification.from_expression(
                    query=P(Y @ {X, Z}),
                    estimand=P(X, Y, Z),
                    graph=NxMixedGraph.from_edges(directed=[(Z, X), (X, Y)], undirected=[(Z, X)]),
                )
            ],
        ),
    ],
)

line_4_example = Example(
    name="graph without X decomposes into multiple C components",
    reference="out of the mind of JZ",
    graph=NxMixedGraph.from_edges(
        directed=[(X, M), (Z, X), (Z, Y), (M, Y)],
        undirected=[(Z, X), (M, Y)],
    ),
    identifications=[
        dict(
            id_in=[
                Identification.from_expression(
                    query=P(Y @ X),
                    estimand=P(M, X, Y, Z),
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, M), (Z, X), (Z, Y), (M, Y)],
                        undirected=[(Z, X), (M, Y)],
                    ),
                )
            ],
            id_out=[
                Identification.from_expression(
                    query=P(M @ {X, Z}, Y @ {X, Z}),
                    estimand=P(M, X, Y, Z),
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, M), (Z, X), (Z, Y), (M, Y)],
                        undirected=[(Z, X), (M, Y)],
                    ),
                ),
                Identification.from_expression(
                    query=P(Z @ {M, X, Y}),
                    estimand=P(M, X, Y, Z),
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, M), (Z, X), (Z, Y), (M, Y)],
                        undirected=[(Z, X), (M, Y)],
                    ),
                ),
            ],
        ),
    ],
)

line_5_example = Example(
    name="graph containing a hedge",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.",
    graph=NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[(X, Y)]),
    identifications=[
        dict(
            id_in=[
                Identification.from_expression(
                    query=P(Y @ X),
                    estimand=P(X, Y),
                    graph=NxMixedGraph.from_edges(directed=[(X, Y)], undirected=[(X, Y)]),
                )
            ],
        )
    ],
)

line_6_example = Example(
    name="ID Line 6 Example",
    description="If there are no bidirected arcs from X to the other nodes in the"
    " current subproblem under consideration, then we can replace acting"
    " on X by conditioning, and thus solve the subproblem.",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.",
    graph=NxMixedGraph.from_edges(directed=[(X, Y), (X, Z), (Z, Y)], undirected=[(X, Z)]),
    identifications=[
        dict(
            id_in=[
                Identification.from_expression(
                    query=P(Y @ [X, Z]),
                    estimand=P(X, Y, Z),
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, Y), (X, Z), (Z, Y)],
                        undirected=[(X, Z)],
                    ),
                )
            ],
            id_out=[
                Identification.from_expression(
                    query=P(Y @ {X, Z}),
                    estimand=P(Y | [X, Z]),
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, Y), (X, Z), (Z, Y)],
                        undirected=[(X, Z)],
                    ),
                )
            ],
        ),
        dict(
            id_in=[
                Identification.from_expression(
                    query=P(Y @ X),
                    estimand=P(X, Y),
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, Y)],
                    ),
                )
            ],
            id_out=[
                Identification.from_expression(
                    query=P(Y @ X),
                    estimand=P(Y | X),
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, Y)],
                    ),
                )
            ],
        ),
    ],
)

line_7_example = Example(
    name="ID Line 7 example, figure 5a and b",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.",
    graph=NxMixedGraph.from_edges(directed=[(X, Y1), (W1, Y1)], undirected=[(W1, Y1)]),
    identifications=[
        dict(
            id_in=[
                Identification.from_expression(
                    query=P(Y1 @ [X, W1]),
                    estimand=P(X, Y1, W1),
                    graph=NxMixedGraph.from_edges(
                        directed=[(X, Y1), (W1, X)], undirected=[(W1, Y1)]
                    ),
                )
            ],
            id_out=[
                Identification.from_expression(
                    query=P(Y1 @ W1),
                    estimand=P(Y1 | [X, W1]) * P(W1),
                    graph=NxMixedGraph.from_edges(undirected=[(W1, Y1)]),
                )
            ],
        )
    ],
)

figure_6a = Example(
    name="Causal graph with identifiable conditional effect P(y|do(x),z)",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.",
    graph=NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)], undirected=[(X, Z)]),
    identifications=[
        dict(
            id_in=[
                Identification.from_parts(
                    outcomes={Y},
                    treatments={X},
                    conditions={Z},
                    estimand=P(X, Y, Z),
                    graph=NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)], undirected=[(X, Z)]),
                ),
            ],
            id_out=[
                Identification.from_expression(
                    query=P(Y @ (X, Z)),
                    estimand=P(Y | (X, Z)) / Sum.safe(expression=P(Y | (X, Z)), ranges=(Y,)),
                    graph=NxMixedGraph.from_edges(directed=[(X, Z), (Z, Y)], undirected=list()),
                ),
            ],
        )
    ],
)

tikka_unidentifiable_graph = Example(
    name="Tikka's unidentifiable example",
    reference="Tikka, S. (2020). Identifying Counterfactual Queries with the R Package cfid",
    graph=NxMixedGraph.from_edges(
        directed=[(X, W), (W, Y), (D, Z), (Z, Y), (X, Y)], undirected=[(X, Y)]
    ),
)

tikka_unidentifiable_cfgraph = Example(
    name="Tikka's unidentifiable example",
    reference="Tikka, S. (2020). Identifying Counterfactual Queries with the R Package cfid",
    graph=NxMixedGraph.from_edges(
        directed=[(X @ -x, W @ -x), (W @ -x, Y @ -x), (D, Z), (Z, Y @ -x), (X, Y)],
        undirected=[(X, Y @ -x)],
    ),
)


figure_9a = Example(
    name="Original causal diagram",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.",
    graph=NxMixedGraph.from_edges(directed=[(X, W), (W, Y), (D, Z), (Z, Y)], undirected=[(X, Y)]),
)

figure_9b = Example(
    name="Parallel worlds graph for :math:`P(y_x|x', x_d, d)`",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.",
    graph=NxMixedGraph.from_edges(
        directed=[
            (X @ -X, W @ -X),
            (W @ -X, Y @ -X),
            (D @ -X, Z @ -X),
            (Z @ -X, Y @ -X),
            (X, W),
            (W, Y),
            (D, Z),
            (Z, Y),
            (X @ D, W @ D),
            (W @ D, Y @ D),
            (D @ D, Z @ D),
            (Z @ D, Y @ D),
        ],
        undirected=[
            (X, Y),
            (X @ D, X),
            (Y @ -X, Y),
            (Y, Y @ D),
            (Y @ D, Y @ -X),
            (X, Y @ -X),
            (X @ D, Y),
            (X, Y @ D),
            (X @ D, Y @ -X),
            (X @ D, Y @ D),
            (D @ -X, D),
            (W @ -X, W),
            (W, W @ D),
            (W @ D, W @ -X),
            (Z @ -X, Z),
            (Z, Z @ D),
            (Z @ -X, Z @ D),
        ],
    ),
)

figure_9c = Example(
    name="Counterfactual graph for :math:`P(y_x | x', z_d, d)`",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.",
    graph=NxMixedGraph.from_edges(
        directed=[(X @ -X, W @ -X), (W @ -X, Y @ -X), (D, Z), (Z, Y @ -X)],
        undirected=[(X, Y @ -X)],
    ),
)

tikka_figure_2 = Example(
    name=r"Figure 2: A graph for the example on identifiability of a conditional counterfacual "
    r"query :math:`P(y_x|z_x\wedge x')`",
    reference="Tikka, S (2022) Identifiying Counterfactual Queries with the R package cfid",
    graph=NxMixedGraph.from_edges(directed=[(X, Z), (X, Y), (Z, Y)], undirected=[(X, Z)]),
)

tikka_figure_5 = Example(
    name=r"Figure 5: Counterfactual graph :math:`G'` for :math:`y_x\wedge x'\wedge z_d\wedge d`",
    reference="Tikka, S (2022) Identifiying Counterfactual Queries with the R package cfid",
    graph=NxMixedGraph.from_edges(
        nodes=(X, Y @ -x, D, Z, X @ -x, W @ -x),
        directed=[(D, Z), (Z, Y @ -x), (X @ -x, W @ -x), (W @ -x, Y @ -x)],
        undirected=[(X, Y @ -x)],
    ),
)


tikka_figure_6a = Example(
    name=r"Figure 6a: Parallel worlds graph for :math:`y_x\wedge z_x\wedge x'` (the counterfactual graph)",
    reference="Tikka, S (2022) Identifiying Counterfactual Queries with the R package cfid",
    graph=NxMixedGraph.from_edges(
        directed=[(X, Z), (Z, Y), (X, Y), (X @ -x, Z @ -x), (Z @ -x, Y @ -x), (X @ -x, Y @ -x)],
        undirected=[(X, Z), (X, Z @ -x), (Z, Z @ -x), (Y, Y @ -x)],
    ),
)

tikka_figure_6b = Example(
    name=r"Figure 6b: Parallel worlds graph for :math:`y_{x,z}\wedge x'` (the counterfactual graph)",
    reference="Tikka, S (2022) Identifiying Counterfactual Queries with the R package cfid",
    graph=NxMixedGraph.from_edges(
        directed=[
            (X, Z),
            (Z, Y),
            (X, Y),
            (Z @ (-x, -z), Y @ (-x, -z)),
            (X @ (-x, -z), Y @ (-x, -z)),
        ],
        undirected=[(X, Z), (Y, Y @ (-x, -z))],
    ),
)

figure_9d = Example(
    name="Counterfactual graph resulting from application of make_counterfactual_graph() with"
    " joint distribution from which :math:`P(y_{x,z}|x')` is derived, namely  :math:`P(y_{x,z}, x')`",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.",
    graph=NxMixedGraph.from_edges(
        nodes=(X, X @ (-X, -Z), Z @ (-X, -Z), W @ (-X, -Z), Y @ (-X, -Z)),
        directed=[
            (X @ (-X, -Z), W @ (-X, -Z)),
            (Z @ (-X, -Z), Y @ (-X, -Z)),
            (W @ (-X, -Z), Y @ (-X, -Z)),
        ],
        undirected=[(X, Y @ (-X, -Z))],
    ),
)

figure_9e = Example(
    name="Counterfactual graph for :math:`P(Y @ (~X, Z) | X)`",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.",
    graph=NxMixedGraph.from_edges(
        nodes=(D, X, X @ (~X, Z), Z @ (~X, Z), W @ (~X, Z), Y @ (~X, Z)),
        directed=[(D, Z), (X @ (~X, Z), W @ (~X, Z)), (Z, Y @ (~X, Z)), (W @ (~X, Z), Y @ (~X, Z))],
        undirected=[(X, Y @ (~X, Z))],
    ),
)

figure_11a = Example(
    name="Intermediate graph obtained by **make-cg** in constructing the"
    " counterfactual graph for for :math:`P(y_x|x', z_d, d)` from Figure 9b",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.",
    graph=NxMixedGraph.from_edges(
        directed=[
            (X @ -X, W @ -X),
            (W @ -X, Y @ -X),
            (D, Z @ -X),
            (Z @ -X, Y @ -X),
            (X, W),
            (W, Y),
            (D, Z),
            (Z, Y),
            (X, W @ D),
            (W @ D, Y @ D),
            (D @ D, Z @ D),
            (Z @ D, Y @ D),
        ],
        undirected=[
            (X, Y),
            (Y @ -X, Y),
            (Y, Y @ D),
            (Y @ D, X),
            (X, Y @ -X),
            (Y @ D, Y @ -X),
            (W @ -X, W),
            (W, W @ D),
            (W @ D, W @ -X),
            (Z @ -X, Z),
            (Z, Z @ D),
            (Z @ -X, Z @ D),
        ],
    ),
)

figure_11b = Example(
    name="Intermediate graph obtained by **make-cg** in constructing the"
    " counterfactual graph for for :math:`P(y_x|x', z_d, d)` from Figure 9b",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.",
    graph=NxMixedGraph.from_edges(
        directed=[
            (X @ -X, W @ -X),
            (W @ -X, Y @ -X),
            (D, Z),
            (Z, Y @ -X),
            (Z, Y @ D),
            (Z, Y),
            (X, W),
            (W, Y),
            (W, Y @ D),
        ],
        undirected=[
            (X, Y),
            (Y @ -X, Y),
            (Y, Y @ D),
            (Y @ D, X),
            (Y @ D, Y @ -X),
            (X, Y @ -X),
            (X, Y @ D),
            (W @ -X, W),
        ],
    ),
)

figure_11c = Example(
    name="Intermediate graph obtained by **make-cg** in constructing the counterfactual"
    " graph for for :math:`P(y_x|x', z_d, d)` from Figure 9b",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy.",
    graph=NxMixedGraph.from_edges(
        directed=[
            (X @ -X, W @ -X),
            (W @ -X, Y @ -X),
            (D, Z),
            (Z, Y @ -X),
            (Z, Y),
            (X, W),
            (W, Y),
        ],
        undirected=[
            (X, Y),
            (Y @ -X, Y),
            (X, Y @ -X),
            (W @ -X, W),
        ],
    ),
)

cyclic_directed_example = Example(
    name="Cyclic directed graph",
    reference="out of the mind of JZ and ZW",
    graph=NxMixedGraph.from_edges(directed=[(A, B), (A, C), (B, A)]),
)
#: Treatment: X
#: Outcome: Y
identifiability_1 = NxMixedGraph.from_edges(
    directed=[
        (Z1, Z2),
        (Z1, Z3),
        (Z2, X),
        (Z3, X),
        (Z4, X),
        (Z4, Z5),
        (Z3, Y),
        (X, Y),
        (Z3, Y),
    ],
)
identifiability_1_example = Example(
    name="Identifiability 1",
    reference='J. Pearl. 2009. "Causality: Models, Reasoning and Inference.'
    ' 2nd ed." Cambridge University Press, p. 80.',
    graph=identifiability_1,
    conditional_independencies=(
        DSeparationJudgement.create(X, Z1, [Z2, Z3]),
        DSeparationJudgement.create(X, Z5, [Z4]),
        DSeparationJudgement.create(Y, Z1, [X, Z3, Z4]),
        DSeparationJudgement.create(Y, Z2, [X, Z1, Z3]),
        DSeparationJudgement.create(Y, Z4, [X, Z3, Z5]),
        DSeparationJudgement.create(Z1, Z4),
        DSeparationJudgement.create(Z1, Z5),
        DSeparationJudgement.create(Z2, Z3, [Z1]),
        DSeparationJudgement.create(Z2, Z4),
        DSeparationJudgement.create(Z2, Z5),
        DSeparationJudgement.create(Z3, Z5),
        DSeparationJudgement.create(Y, Z5, [X, Z3]),
        DSeparationJudgement.create(Z3, Z4),
    ),
)

#: Treatment: X
#: Outcome: Y
identifiability_2 = NxMixedGraph.from_edges(
    directed=[
        (Z1, Z2),
        (Z1, Z3),
        (Z2, X),
        (Z3, X),
        (X, W0),
        (W0, Y),
        (Z4, Z3),
        (Z4, Z5),
        (Z5, Y),
        (X, W1),
        (W1, W2),
        (W2, Y),
        (Z4, Z3),
        (Z3, Y),
    ],
    undirected=[
        (Z1, X),
        (Z2, Z3),
        (Z3, Z5),
        (Z4, Y),
    ],
)

identifiability_2_example = Example(
    name="Identifiability 2",
    reference="E. Bareinboim modification of Identifiability 1.",
    graph=identifiability_2,
    verma_constraints=[
        VermaConstraint(
            rhs_cfactor=Q[Z5](Z4, Z5),
            rhs_expr=Sum[u_3, Z4](P(Z5 | (u_3, Z4)) * P(Z4) * P(u_3)),
            lhs_cfactor=Sum[Z3](Q[Z3, Z5](Z1, Z4, Z3, Z5)),
            lhs_expr=Sum[Z3](P(Z5 | (Z1, Z2, Z3, Z4)) * P(Z3 | (Z1, Z2, Z4))),
            variables=(Z1,),
        ),
        VermaConstraint(
            rhs_cfactor=Q[Z5](Z4, Z5),
            rhs_expr=Sum[u_3, Z4](P(Z5 | (u_3, Z4)) * P(Z4) * P(u_3)),
            lhs_cfactor=(Q[Z2, Z5](Z1, Z4, Z2, Z5) / Sum[Z5](Q[Z2, Z5](Z1, Z4, Z2, Z5))),
            lhs_expr=(
                Sum[Z3](P(Z5 | (Z1, Z2, Z3, Z4)) * P(Z3 | (Z1, Z4, Z2)) * P(Z2 | (Z1, Z4)))
                / Sum[Z3, Z5](P(Z5 | (Z1, Z4, Z2, Z3)) * P(Z3 | (Z1, Z4, Z2)) * P(Z2 | (Z1, Z4)))
            ),
            variables=(Z1, Z2),
        ),
    ],
    conditional_independencies=[
        DSeparationJudgement.create(W0, W1, [X]),
        DSeparationJudgement.create(W0, W2, [X]),
        DSeparationJudgement.create(W0, Z1, [X]),
        DSeparationJudgement.create(W0, Z2, [X]),
        DSeparationJudgement.create(W0, Z3, [X]),
        DSeparationJudgement.create(W0, Z4, [X]),
        DSeparationJudgement.create(W0, Z5, [X]),
        DSeparationJudgement.create(W1, Y, [W0, W2, Z3, Z4, Z5]),
        DSeparationJudgement.create(W1, Z1, [X]),
        DSeparationJudgement.create(W1, Z2, [X]),
        DSeparationJudgement.create(W1, Z3, [X]),
        DSeparationJudgement.create(W1, Z4, [X]),
        DSeparationJudgement.create(W1, Z5, [X]),
        DSeparationJudgement.create(W2, X, [W1]),
        DSeparationJudgement.create(W2, Z1, [W1]),
        DSeparationJudgement.create(W2, Z2, [W1]),
        DSeparationJudgement.create(W2, Z3, [W1]),
        DSeparationJudgement.create(W2, Z4, [W1]),
        DSeparationJudgement.create(W2, Z5, [W1]),
        DSeparationJudgement.create(X, Y, [W0, W2, Z3, Z4, Z5]),
        DSeparationJudgement.create(X, Z4, [Z1, Z2, Z3]),
        DSeparationJudgement.create(X, Z5, [Z1, Z2, Z3]),
        DSeparationJudgement.create(Y, Z1, [W0, W2, Z3, Z4, Z5]),
        DSeparationJudgement.create(Y, Z2, [W0, W2, Z3, Z4, Z5]),
        DSeparationJudgement.create(Z1, Z4),
        DSeparationJudgement.create(Z1, Z5),
        DSeparationJudgement.create(Z2, Z4),
        DSeparationJudgement.create(Z2, Z5),
    ],
)

#: The Identifiability 3 example
#: Treatment: X
#: Outcome: Y
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 92.
identifiability_3 = NxMixedGraph.from_edges(
    directed=[
        (Z2, X),
        (Z2, Z1),
        (Z2, Z3),
        (X, Z1),
        (Z3, Y),
        (Z1, Y),
    ],
    undirected=[
        (Z2, X),
        (Z2, Y),
        (X, Z3),
        (X, Y),
    ],
)

#: The Identifiability 4 example
#: Treatment: X
#: Outcome: Y
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 92.
identifiability_4 = NxMixedGraph.from_edges(
    directed=[
        (X, Z1),
        (X, Y),
        (Z1, Z2),
        (Z1, Y),
        (Z2, Y),
    ],
    undirected=[
        (X, Z2),
        (Z1, Y),
    ],
)

#: The Identifiability 5 example
#: Treatment: X1, X2
#: Outcome: Y
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 119.
identifiability_5 = NxMixedGraph.from_edges(
    directed=[
        ("X1", Z),
        ("X1", Y),
        ("X1", "X2"),
        (Z, "X2"),
        ("X2", Y),
    ],
    undirected=[
        ("X1", Z),
        (Z, Y),
    ],
)

#: The Identifiability 6 example
#: Treatment: X1, X2
#: Outcome: Y
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 125.
identifiability_6 = NxMixedGraph.from_edges(
    directed=[
        (Z1, "X1"),
        ("X1", "X2"),
        ("X2", Y),
        (Z2, Y),
    ],
    undirected=[
        (Z1, Z2),
        (Z1, "X2"),
        (Z2, "X2"),
    ],
)

#: The Identifiability 7 example
#: Treatment: X
#: Outcome: Y
#: Reference: J. Tian. 2002. "Studies in Causal Reasoning and Learning." p. 90.
identifiability_7 = NxMixedGraph.from_edges(
    directed=[
        (W1, W2),
        ("W3", "W4"),
        (W2, X),
        ("W4", X),
        (X, Y),
    ],
    undirected=[
        (W1, X),
        (W1, Y),
        (W1, "W3"),
        ("W3", W2),
        ("W3", "W5"),
        ("W5", "W4"),
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
verma_1 = NxMixedGraph.from_str_edges(
    directed=[
        ("V1", "V2"),
        ("V2", "V3"),
        ("V3", "V4"),
    ],
    undirected=[
        ("V2", "V4"),
    ],
)

#: The Verma 2 example
#: Treatment: V1
#: Outcome: V5
#: Reference: J. Tian. 2002. "Studies in Causal Reasoning and Learning." p. 70.
verma_2 = NxMixedGraph.from_str_edges(
    directed=[
        ("V1", "V2"),
        ("V2", "V3"),
        ("V3", "V4"),
        ("V4", "V5"),
    ],
    undirected=[
        ("V1", "V3"),
        ("V2", "V4"),
        ("V3", "V5"),
    ],
)

#: The Verma 3 example
#: Treatment: V1
#: Outcome: V5
#: Reference: J. Tian. 2002. "Studies in Causal Reasoning and Learning." p. 59.
verma_3 = NxMixedGraph.from_str_edges(
    directed=[
        ("V1", "V2"),
        ("V2", "V3"),
        ("V3", "V4"),
        ("V4", "V5"),
    ],
    undirected=[
        ("V1", "V5"),
        ("V1", "V3"),
        ("V2", "V4"),
    ],
)

#: The Verma 4 example
#: Treatment: V1
#: Outcome: V5
#: Reference: E. Bareinboim modification of Verma 2.
verma_4 = NxMixedGraph.from_str_edges(
    directed=[
        ("V1", "V2"),
        ("V2", "V3"),
        ("V3", "V4"),
        ("V4", "V5"),
    ],
    undirected=[
        ("V1", "V5"),
        ("V1", "V3"),
        ("V2", "V4"),
        ("V3", "V5"),
    ],
)

#: The Verma 5 example
#: Treatment: V1
#: Outcome: V5
#: Reference: E. Bareinboim modification of Verma 2.
verma_5 = NxMixedGraph.from_str_edges(
    directed=[
        ("V1", "V2"),
        ("V2", "V3"),
        ("V3", "V4"),
        ("V4", "V5"),
        ("V5", "V6"),
    ],
    undirected=[
        ("V0", "V1"),
        ("V0", "V6"),
        ("V1", "V5"),
        ("V1", "V3"),
        ("V2", "V4"),
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
    directed=[
        (Z, X),
        (X, Y),
    ],
    undirected=[
        (Z, X),
        (Z, Y),
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
    directed=[
        (Z, X),
        (X, Y),
    ],
    undirected=[
        (X, Y),
        (Z, Y),
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
    directed=[
        (Z, Y),
        (X, Y),
    ],
    undirected=[
        (X, Z),
        (Z, Y),
    ],
)

#: The Identifiability (Linear) 1 example
#: Treatment: X
#: Outcome: Y
#: Reference: J. Pearl. 2009. "Causality: Models, Reasoning and Inference. 2nd ed." Cambridge University Press, p. 153.
identifiability_linear_1 = NxMixedGraph.from_edges(
    directed=[
        (X, Z),
        (X, W),
        (W, Y),
        (Z, Y),
    ],
    undirected=[
        (X, Z),
        (W, Y),
    ],
)

d_separation_example = Example(
    name="D-separation example",
    reference="http://web.mit.edu/jmn/www/6.034/d-separation.pdf",
    graph=NxMixedGraph.from_edges(
        directed=[
            (AA, C),
            (B, C),
            (C, D),
            (C, E),
            (D, F),
            (F, G),
        ],
    ),
    conditional_independencies=[
        DSeparationJudgement.create(AA, B),
        DSeparationJudgement.create(AA, D, [C]),
        DSeparationJudgement.create(AA, E, [C]),
        DSeparationJudgement.create(AA, F, [C]),
        DSeparationJudgement.create(AA, G, [C]),
        DSeparationJudgement.create(B, D, [C]),
        DSeparationJudgement.create(B, E, [C]),
        DSeparationJudgement.create(B, F, [C]),
        DSeparationJudgement.create(B, G, [C]),
        DSeparationJudgement.create(C, F, [D]),
        DSeparationJudgement.create(C, G, [D]),
        DSeparationJudgement.create(D, E, [C]),
        DSeparationJudgement.create(D, G, [F]),
        DSeparationJudgement.create(E, F, [C]),
        DSeparationJudgement.create(E, G, [C]),
    ],
)


asia_df = pd.read_csv(ASIA_PATH).replace({"yes": 1, "no": -1})
del asia_df[asia_df.columns[0]]

asia_example = Example(
    name="Asia dataset",
    reference="https://www.bnlearn.com/documentation/man/asia.html",
    graph=NxMixedGraph.from_edges(
        directed=[
            (Variable(u), Variable(v))
            for u, v in [
                ("Asia", "Tub"),
                ("Smoke", "Lung"),
                ("Smoke", "Bronc"),
                ("Tub", "Either"),
                ("Lung", "Either"),
                ("Either", "Xray"),
                ("Either", "Dysp"),
                ("Bronc", "Dysp"),
            ]
        ],
    ),
    data=asia_df,
)

figure_2a_example = Example(
    name="Shpitser et al. (2008), Figure 2A",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy. "
    "Journal of Machine Learning Research.",
    graph=NxMixedGraph.from_edges(
        directed=[(X, Y)],
    ),
)

figure_2b_example = Example(
    name="Shpitser et al. (2008), Figure 2B",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy. "
    "Journal of Machine Learning Research.",
    graph=NxMixedGraph.from_edges(
        directed=[(X, Y), (X, Z), (Z, Y)],
        undirected=[(Y, Z)],
    ),
)

complete_hierarchy_figure_2c_example = Example(
    name="Shpitser et al (2008) figure 2d",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy. "
    "Journal of Machine Learning Research.",
    graph=NxMixedGraph.from_edges(
        directed=[
            (X, Y),
            (Z, X),
            (Z, Y),
        ],
        undirected=[(X, Z)],
    ),
)

complete_hierarchy_figure_2d_example = Example(
    name="Shpitser et al (2008) figure 2d",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy. "
    "Journal of Machine Learning Research.",
    graph=NxMixedGraph.from_edges(
        directed=[
            (X, Y),
            (Z, X),
            (Z, Y),
        ],
        undirected=[(X, Z)],
    ),
)

complete_hierarchy_figure_2e_example = Example(
    name="Shpitser et al (2008) figure 2e",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy. "
    "Journal of Machine Learning Research.",
    graph=NxMixedGraph.from_edges(
        directed=[
            (X, Z),
            (Z, Y),
        ],
        undirected=[(X, Y)],
    ),
)

complete_hierarchy_figure_3a_example = Example(
    name="Shpitser et al 2008 figure 3a",
    reference="Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy."
    " Journal of Machine Learning Research.",
    graph=NxMixedGraph.from_edges(
        directed=[(X, Y1), (W1, X), (W2, Y2)],
        undirected=[(W1, W2), (W1, Y1), (W1, Y2), (X, W2)],
    ),
)

id_sir_example = Example(
    name="Identifiable SIR",
    reference="ASKEM",
    graph=NxMixedGraph.from_str_edges(
        directed=[
            ("Infected", "Hospitalized"),
            ("Hospitalized", "Died"),
        ],
        undirected=[("Infected", "Died")],
    ),
)

nonid_sir_example = Example(
    name="Non-Identifiable SIR",
    reference="ASKEM",
    graph=NxMixedGraph.from_str_edges(
        directed=[
            ("Infected", "Died"),
        ],
        undirected=[("Infected", "Died")],
    ),
)

igf_example = Example(
    name="IGF Graph",
    reference="Jeremy Zucker, Sara Mohammad-Taheri, Kaushal Paneri, Somya Bhargava, Pallavi Kolambkar"
    ", Craig Bakker, Jeremy Teuton, Charles Tapley Hoyt, Kristie Oxford, Robert Ness, and Olga Vitek. 2021."
    "Leveraging Structured Biological Knowledge for Counterfactual Inference: a Case Study of Viral Pathogenesis"
    "- IEEE Journals & Magazine. IEEE Transactions on Big Data (January 2021).",
    graph=NxMixedGraph.from_str_edges(
        nodes=["SOS", "Ras", "Raf", "AKT", "Mek", "Erk", "PI3K"],
        directed=[
            ("SOS", "Ras"),
            ("Ras", "PI3K"),
            ("Ras", "Raf"),
            ("PI3K", "AKT"),
            ("AKT", "Raf"),
            ("Raf", "Mek"),
            ("Mek", "Erk"),
        ],
        undirected=[("SOS", "PI3K")],
    ),
    example_queries=[Query.from_str(treatments="SOS", outcomes="Erk")],
)

sars_large_example = Example(
    name="SARS-CoV-2 Graph",
    reference="Jeremy Zucker, Sara Mohammad-Taheri, Kaushal Paneri, Somya Bhargava, Pallavi Kolambkar"
    ", Craig Bakker, Jeremy Teuton, Charles Tapley Hoyt, Kristie Oxford, Robert Ness, and Olga Vitek. 2021."
    "Leveraging Structured Biological Knowledge for Counterfactual Inference: a Case Study of Viral Pathogenesis"
    "- IEEE Journals & Magazine. IEEE Transactions on Big Data (January 2021).",
    graph=NxMixedGraph.from_str_edges(
        nodes=[
            "SARS_COV2",
            "ACE2",
            "Ang",
            "AGTR1",
            "ADAM17",
            "Toci",
            "Sil6r",
            "EGF",
            "TNF",
            "EGFR",
            "PRR",
            "NFKB",
            "IL6STAT3",
            "IL6AMP",
            "cytok",
            "Gefi",
        ],
        directed=[
            ("SARS_COV2", "ACE2"),
            ("ACE2", "Ang"),
            ("Ang", "AGTR1"),
            ("AGTR1", "ADAM17"),
            ("ADAM17", "EGF"),
            ("ADAM17", "TNF"),
            ("ADAM17", "Sil6r"),
            ("SARS_COV2", "PRR"),
            ("PRR", "NFKB"),
            ("EGFR", "NFKB"),
            ("TNF", "NFKB"),
            ("Sil6r", "IL6STAT3"),
            ("Toci", "Sil6r"),
            ("NFKB", "IL6AMP"),
            ("IL6AMP", "cytok"),
            ("IL6STAT3", "IL6AMP"),
            ("EGF", "EGFR"),
            ("Gefi", "EGFR"),
        ],
        undirected=[
            ("SARS_COV2", "Ang"),
            ("ADAM17", "Sil6r"),
            ("PRR", "NFKB"),
            ("EGF", "EGFR"),
            ("EGFR", "TNF"),
            ("EGFR", "IL6STAT3"),
        ],
    ),
    example_queries=[
        Query.from_str(treatments="Sil6r", outcomes="cytok"),
        Query.from_str(treatments="EGFR", outcomes="cytok"),
    ],
)

SARS_SMALL_GRAPH = NxMixedGraph.from_str_edges(
    directed=[
        ("ADAM17", "EGFR"),
        ("ADAM17", "TNF"),
        ("ADAM17", "Sil6r"),
        ("EGFR", "cytok"),
        ("TNF", "cytok"),
        ("Sil6r", "IL6STAT3"),
        ("IL6STAT3", "cytok"),
    ],
    undirected=[
        ("ADAM17", "cytok"),
        ("ADAM17", "Sil6r"),
        ("EGFR", "TNF"),
        ("EGFR", "IL6STAT3"),
    ],
)

sars_small_example = Example(
    name="SARS-CoV-2 Small Graph",
    reference="Sara!",  # FIXME
    graph=SARS_SMALL_GRAPH,
    generate_data=generate_data_for_covid_case_study,
    example_queries=[Query.from_str(outcomes="cytok", treatments="EGFR")],
)

tikka_trso_figure_8_graph = NxMixedGraph.from_edges(
    undirected=[(X1, Y1), (Z, W), (Z, X2)],
    directed=[
        (X1, Y1),
        (X1, Y2),
        (W, Y1),
        (W, Y2),
        (Z, Y1),
        (Z, X2),
        (X2, Y2),
        (Z, Y2),
    ],
)
tikka_trso_figure_8 = Example(
    name="Tikka TRSO Figure 8",
    reference="https://arxiv.org/abs/1806.07172",
    graph=tikka_trso_figure_8_graph,
)


cancer_example = Example(
    name="Smoking and Cancer",
    reference="https://github.com/y0-causal-inference/y0/pull/183",
    graph=NxMixedGraph.from_edges(directed=[(S, T), (T, C), (S, C)], undirected=[(S, T)]),
    generate_data=generate_data_for_smoke_cancer,
    example_queries=[Query.from_str(outcomes="C", treatments="S")],
)


examples = [v for name, v in locals().items() if name.endswith("_example")]
