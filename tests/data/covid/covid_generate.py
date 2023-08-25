ndp = 1000
seed = 1
import numpy as np
from ananke.estimation import CausalEffect

from y0.graph import NxMixedGraph


def generate_data_for_covid_case_study(treatment_assignment="None"):
    np.random.seed(seed)
    U_ADAM17_Sil6r = np.random.normal(loc=40.0, scale=10.0, size=ndp)
    U_IL6STAT_EGFR = np.random.normal(loc=44.0, scale=10.0, size=ndp)
    U_TNF_EGFR = np.random.normal(loc=40.0, scale=10.0, size=ndp)
    U_ADAM17_cytok = np.random.normal(loc=44.0, scale=10.0, size=ndp)

    np.random.seed(seed)
    beta0_ADAM17 = -1
    beta_U_ADAM17_cytok = 0.04
    beta_U_ADAM17_Sil6r = 0.04
    ADAM17 = np.random.normal(
        loc=100
        / (
            1
            + np.exp(
                -beta0_ADAM17
                - U_ADAM17_cytok * beta_U_ADAM17_cytok
                - U_ADAM17_Sil6r * beta_U_ADAM17_Sil6r
            )
        ),
        scale=1,
        size=ndp,
    )

    np.random.seed(seed)
    beta0_Sil6r = -1.9
    beta_ADAM17ToSil6r = 0.03
    beta_U_ADAM17_Sil6r = 0.05
    Sil6r = np.random.normal(
        loc=100
        / (
            1
            + np.exp(
                -beta0_Sil6r - ADAM17 * beta_ADAM17ToSil6r - U_ADAM17_Sil6r * beta_U_ADAM17_Sil6r
            )
        ),
        scale=1,
    )

    np.random.seed(seed)
    beta0_TNF = -1.8
    beta_ADAM17ToTNF = 0.05
    beta_U_TNF_EGFR = 0.06
    TNF = np.random.normal(
        loc=100
        / (1 + np.exp(-beta0_TNF - ADAM17 * beta_ADAM17ToTNF - U_TNF_EGFR * beta_U_TNF_EGFR)),
        scale=1,
    )

    np.random.seed(seed)
    beta0_EGFR = -1.9
    beta_ADAM17_EGFR = 0.03
    beta_U_IL6STAT_EGFR = -0.04
    beta_U_TNF_EGFR = 0.02
    if treatment_assignment is not None:
        EGFR = np.array([treatment_assignment] * ndp)
    else:
        p = np.array(
            1
            / (
                1
                + np.exp(
                    -beta0_EGFR
                    - ADAM17 * beta_ADAM17_EGFR
                    - U_IL6STAT_EGFR * beta_U_IL6STAT_EGFR
                    - U_TNF_EGFR * beta_U_TNF_EGFR
                )
            )
        )
        EGFR = np.random.binomial(1, p, size=ndp)

    np.random.seed(seed)
    beta0_IL6STAT3 = -1.6
    beta_U_IL6STAT_EGFR = -0.05
    beta_Sil6rToIL6STAT3 = 0.04
    IL6STAT3 = np.random.normal(
        loc=100
        / (
            1
            + np.exp(
                -beta0_IL6STAT3
                - U_IL6STAT_EGFR * beta_U_IL6STAT_EGFR
                - Sil6r * beta_Sil6rToIL6STAT3
            )
        ),
        scale=1,
    )

    np.random.seed(seed)
    beta0_cytok = -1.9
    beta_IL6STAT3Tocytok = 0.02
    beta_EGFRTocytok = 0.06
    beta_TNFTocytok = 0.01
    beta_U_ADAM17_cytok = 0.01
    cytok = np.random.normal(
        loc=100
        / (
            1
            + np.exp(
                -beta0_cytok
                - IL6STAT3 * beta_IL6STAT3Tocytok
                - EGFR * beta_EGFRTocytok
                - TNF * beta_TNFTocytok
                - U_ADAM17_cytok * beta_U_ADAM17_cytok
            )
        ),
        scale=1,
    )

    data = {
        "ADAM17": ADAM17,
        "Sil6r": Sil6r,
        "TNF": TNF,
        "EGFR": EGFR,
        "IL6STAT3": IL6STAT3,
        "cytok": cytok,
    }
    df = pd.DataFrame(data)

    return df


# get observational and interventional data
obs_data = generate_data_for_covid_case_study(treatment_assignment="None")
intv_data_X1 = generate_data_for_covid_case_study(treatment_assignment=1)
intv_data_X0 = generate_data_for_covid_case_study(treatment_assignment=0)

# TRUE ACE
np.mean(intv_data_X1["cytok"]) - np.mean(intv_data_X0["cytok"])

# ananke estimate of ACE
graph_3 = NxMixedGraph.from_str_edges(
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
ace_obj = CausalEffect(graph=graph_3, treatment="EGFR", outcome="cytok")
ace_aipw = ace_obj.compute_effect(obs_data, "aipw")
