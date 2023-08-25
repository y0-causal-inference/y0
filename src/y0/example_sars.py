"""Examples for SARS-CoV-2 and COVID19."""

import numpy as np
import pandas as pd

from y0.dsl import Variable


def _r_exp(x):
    return 1 / (1 + np.exp(x))


def generate_data_for_covid_case_study(
    num_samples: int, treatments: dict[Variable, float] | None = None, *, seed: int | None = None
) -> pd.DataFrame:
    """Generate testing data for the SARS-CoV-2 small graph.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding
        to the variable names SARS-CoV-2 small graph
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)

    u_adam17_sil6r_value = generator.normal(loc=40.0, scale=10.0, size=num_samples)
    U_IL6STAT_EGFR_value = generator.normal(loc=44.0, scale=10.0, size=num_samples)
    U_TNF_EGFR_value = generator.normal(loc=40.0, scale=10.0, size=num_samples)
    U_ADAM17_cytok_value = generator.normal(loc=44.0, scale=10.0, size=num_samples)

    beta0_ADAM17 = -1
    beta_U_ADAM17_cytok = 0.04
    beta_U_ADAM17_Sil6r = 0.04
    ADAM17 = generator.normal(
        loc=100
        / (
            1
            + np.exp(
                -beta0_ADAM17
                - U_ADAM17_cytok_value * beta_U_ADAM17_cytok
                - u_adam17_sil6r_value * beta_U_ADAM17_Sil6r
            )
        ),
        scale=1,
        size=num_samples,
    )

    beta0_Sil6r = -1.9
    beta_ADAM17ToSil6r = 0.03
    beta_U_ADAM17_Sil6r = 0.05
    Sil6r_value = generator.normal(
        loc=100
        / (
            1
            + np.exp(
                -beta0_Sil6r
                - ADAM17 * beta_ADAM17ToSil6r
                - u_adam17_sil6r_value * beta_U_ADAM17_Sil6r
            )
        ),
        scale=1,
    )

    beta0_TNF = -1.8
    beta_ADAM17ToTNF = 0.05
    beta_U_TNF_EGFR = 0.06
    TNF = generator.normal(
        loc=100
        / (1 + np.exp(-beta0_TNF - ADAM17 * beta_ADAM17ToTNF - U_TNF_EGFR_value * beta_U_TNF_EGFR)),
        scale=1,
    )

    beta0_EGFR = -1.9
    beta_ADAM17_EGFR = 0.03
    beta_U_IL6STAT_EGFR = -0.04
    beta_U_TNF_EGFR = 0.02
    if Variable("EGFR") in treatments:
        EGFR = np.full(num_samples, treatments[Variable("EGFR")])
    else:
        p = _r_exp(
            -beta0_EGFR
            - ADAM17 * beta_ADAM17_EGFR
            - U_IL6STAT_EGFR_value * beta_U_IL6STAT_EGFR
            - U_TNF_EGFR_value * beta_U_TNF_EGFR
        )
        EGFR = generator.binomial(1, p, size=num_samples)

    beta0_IL6STAT3 = -1.6
    beta_U_IL6STAT_EGFR = -0.05
    beta_Sil6rToIL6STAT3 = 0.04
    IL6STAT3 = generator.normal(
        loc=100
        / (
            1
            + np.exp(
                -beta0_IL6STAT3
                - U_IL6STAT_EGFR_value * beta_U_IL6STAT_EGFR
                - Sil6r_value * beta_Sil6rToIL6STAT3
            )
        ),
        scale=1,
    )

    beta0_cytok = -1.9
    beta_IL6STAT3Tocytok = 0.02
    beta_EGFRTocytok = 0.06
    beta_TNFTocytok = 0.01
    beta_U_ADAM17_cytok = 0.01
    cytok = generator.normal(
        loc=100
        / (
            1
            + np.exp(
                -beta0_cytok
                - IL6STAT3 * beta_IL6STAT3Tocytok
                - EGFR * beta_EGFRTocytok
                - TNF * beta_TNFTocytok
                - U_ADAM17_cytok_value * beta_U_ADAM17_cytok
            )
        ),
        scale=1,
    )

    data = {
        "ADAM17": ADAM17,
        "Sil6r": Sil6r_value,
        "TNF": TNF,
        "EGFR": EGFR,
        "IL6STAT3": IL6STAT3,
        "cytok": cytok,
    }
    df = pd.DataFrame(data)
    return df
