"""Examples for SARS-CoV-2 and COVID19."""

import numpy as np
import pandas as pd

from y0.dsl import Variable

__all__ = [
    "generate_data_for_covid_case_study",
]


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
    u_il6_stat_egfr_value = generator.normal(loc=44.0, scale=10.0, size=num_samples)
    u_tnf_egfr_value = generator.normal(loc=40.0, scale=10.0, size=num_samples)
    u_adam17_cytok_value = generator.normal(loc=44.0, scale=10.0, size=num_samples)

    beta0_adam17 = -1
    beta_u_adam17_cytok = 0.04
    beta_u_adam17_sil6r = 0.04
    adam17 = generator.normal(
        loc=100
        * _r_exp(
            -beta0_adam17
            - u_adam17_cytok_value * beta_u_adam17_cytok
            - u_adam17_sil6r_value * beta_u_adam17_sil6r
        ),
        scale=1,
        size=num_samples,
    )

    beta0_sil6r = -1.9
    beta_adam17_to_sil6r = 0.03
    beta_u_adam17_sil6r = 0.05
    sil6r_value = generator.normal(
        loc=100
        * _r_exp(
            -beta0_sil6r
            - adam17 * beta_adam17_to_sil6r
            - u_adam17_sil6r_value * beta_u_adam17_sil6r
        ),
        scale=1,
    )

    beta0_tnf = -1.8
    beta_adam17_to_tnf = 0.05
    beta_u_tnf_egfr = 0.06
    tnf = generator.normal(
        loc=100
        * _r_exp(-beta0_tnf - adam17 * beta_adam17_to_tnf - u_tnf_egfr_value * beta_u_tnf_egfr),
        scale=1,
    )

    beta0_egfr = -1.9
    beta_adam17_egfr = 0.03
    beta_u_il6_stat_egfr = -0.04
    beta_u_tnf_egfr = 0.02
    if Variable("EGFR") in treatments:
        egfr = np.full(num_samples, treatments[Variable("EGFR")])
    else:
        p = _r_exp(
            -beta0_egfr
            - adam17 * beta_adam17_egfr
            - u_il6_stat_egfr_value * beta_u_il6_stat_egfr
            - u_tnf_egfr_value * beta_u_tnf_egfr
        )
        egfr = generator.binomial(1, p, size=num_samples)

    beta0_il6_stat3 = -1.6
    beta_u_il6_stat_egfr = -0.05
    beta_sil6r_to_il6_stat3 = 0.04
    il6_stat3 = generator.normal(
        loc=100
        * _r_exp(
            -beta0_il6_stat3
            - u_il6_stat_egfr_value * beta_u_il6_stat_egfr
            - sil6r_value * beta_sil6r_to_il6_stat3
        ),
        scale=1,
    )

    beta0_cytok = -1.9
    beta_il6_stat3_tocytok = 0.02
    beta_egfr_tocytok = 0.06
    beta_tnf_tocytok = 0.01
    beta_u_adam17_cytok = 0.01
    cytok = generator.normal(
        loc=100
        * _r_exp(
            -beta0_cytok
            - il6_stat3 * beta_il6_stat3_tocytok
            - egfr * beta_egfr_tocytok
            - tnf * beta_tnf_tocytok
            - u_adam17_cytok_value * beta_u_adam17_cytok
        ),
        scale=1,
    )

    data = {
        "ADAM17": adam17,
        "Sil6r": sil6r_value,
        "TNF": tnf,
        "EGFR": egfr,
        "IL6STAT3": il6_stat3,
        "cytok": cytok,
    }
    df = pd.DataFrame(data)
    return df
