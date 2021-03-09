# flake8: noqa

# The MIT License (MIT)
#
# Copyright (c) 2013-2017 pgmpy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Originally copied from: https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/estimators/CITests.py
# 20Feb2021

"""Statitistical tests for conditional independence with conditions."""

from warnings import warn
from scipy import stats


def chi_square(X, Y, Z, data, boolean=True, **kwargs):
    r"""
    Chi-square conditional independence test.
    Tests the null hypothesis that X is independent from Y given Zs.

    This is done by comparing the observed frequencies with the expected
    frequencies if X,Y were conditionally independent, using a chisquare
    deviance statistic. The expected frequencies given independence are
    :math:`P(X,Y,Zs) = P(X|Zs)*P(Y|Zs)*P(Zs)`. The latter term can be computed
    as :math:`P(X,Zs)*P(Y,Zs)/P(Zs).

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    If boolean = False, Returns 3 values:
        chi: float
            The chi-squre test statistic.

        p_value: float
            The p_value, i.e. the probability of observing the computed chi-square
            statistic (or an even higher value), given the null hypothesis
            that X \u27C2 Y | Zs.

        dof: int
            The degrees of freedom of the test.

    If boolean = True, returns:
        independent: boolean
            If the p_value of the test is greater than significance_level, returns True.
            Else returns False.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Chi-squared_test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> chi_square(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="pearson", **kwargs
    )


def g_sq(X, Y, Z, data, boolean=True, **kwargs):
    """
    G squared test for conditional independence. Also commonly known as G-test,
    likelihood-ratio or maximum likelihood statistical significance test.
    Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    If boolean = False, Returns 3 values:
        chi: float
            The chi-squre test statistic.

        p_value: float
            The p_value, i.e. the probability of observing the computed chi-square
            statistic (or an even higher value), given the null hypothesis
            that X \u27C2 Y | Zs.

        dof: int
            The degrees of freedom of the test.

    If boolean = True, returns:
        independent: boolean
            If the p_value of the test is greater than significance_level, returns True.
            Else returns False.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/G-test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> g_sq(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> g_sq(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> g_sq(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="log-likelihood", **kwargs
    )


def log_likelihood(X, Y, Z, data, boolean=True, **kwargs):
    """
    Log likelihood ratio test for conditional independence. Also commonly known
    as G-test, G-squared test or maximum likelihood statistical significance
    test.  Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    If boolean = False, Returns 3 values:
        chi: float
            The chi-squre test statistic.

        p_value: float
            The p_value, i.e. the probability of observing the computed chi-square
            statistic (or an even higher value), given the null hypothesis
            that X \u27C2 Y | Zs.

        dof: int
            The degrees of freedom of the test.

    If boolean = True, returns:
        independent: boolean
            If the p_value of the test is greater than significance_level, returns True.
            Else returns False.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/G-test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> log_likelihood(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> log_likelihood(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> log_likelihood(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="log-likelihood", **kwargs
    )


def freeman_tuckey(X, Y, Z, data, boolean=True, **kwargs):
    """
    Freeman Tuckey test for conditional independence [1].
    Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    If boolean = False, Returns 3 values:
        chi: float
            The chi-squre test statistic.

        p_value: float
            The p_value, i.e. the probability of observing the computed chi-square
            statistic (or an even higher value), given the null hypothesis
            that X \u27C2 Y | Zs.

        dof: int
            The degrees of freedom of the test.

    If boolean = True, returns:
        independent: boolean
            If the p_value of the test is greater than significance_level, returns True.
            Else returns False.

    References
    ----------
    [1] Read, Campbell B. "Freeman—Tukey chi-squared goodness-of-fit statistics."
    Statistics & probability letters 18.4 (1993): 271-278.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> freeman_tuckey(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> freeman_tuckey(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> freeman_tuckey(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="freeman-tukey", **kwargs
    )


def modified_log_likelihood(X, Y, Z, data, boolean=True, **kwargs):
    """
    Modified log likelihood ratio test for conditional independence.
    Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    If boolean = False, Returns 3 values:
        chi: float
            The chi-squre test statistic.

        p_value: float
            The p_value, i.e. the probability of observing the computed chi-square
            statistic (or an even higher value), given the null hypothesis
            that X \u27C2 Y | Zs.

        dof: int
            The degrees of freedom of the test.

    If boolean = True, returns:
        independent: boolean
            If the p_value of the test is greater than significance_level, returns True.
            Else returns False.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> modified_log_likelihood(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> modified_log_likelihood(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> modified_log_likelihood(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X,
        Y=Y,
        Z=Z,
        data=data,
        boolean=boolean,
        lambda_="mod-log-likelihood",
        **kwargs,
    )


def neyman(X, Y, Z, data, boolean=True, **kwargs):
    """
    Neyman's test for conditional independence[1].
    Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    If boolean = False, Returns 3 values:
        chi: float
            The chi-squre test statistic.

        p_value: float
            The p_value, i.e. the probability of observing the computed chi-square
            statistic (or an even higher value), given the null hypothesis
            that X \u27C2 Y | Zs.

        dof: int
            The degrees of freedom of the test.

    If boolean = True, returns:
        independent: boolean
            If the p_value of the test is greater than significance_level, returns True.
            Else returns False.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Neyman%E2%80%93Pearson_lemma

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> neyman(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> neyman(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> neyman(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="neyman", **kwargs
    )


def cressie_read(X, Y, Z, data, boolean=True, **kwargs):
    """
    Cressie Read statistic for conditional independence[1].
    Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list (array-like)
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    If boolean = False, Returns 3 values:
        chi: float
            The chi-squre test statistic.

        p_value: float
            The p_value, i.e. the probability of observing the computed chi-square
            statistic (or an even higher value), given the null hypothesis
            that X \u27C2 Y | Zs.

        dof: int
            The degrees of freedom of the test.

    If boolean = True, returns:
        independent: boolean
            If the p_value of the test is greater than significance_level, returns True.
            Else returns False.

    References
    ----------
    [1] Cressie, Noel, and Timothy RC Read. "Multinomial goodness‐of‐fit tests."
    Journal of the Royal Statistical Society: Series B (Methodological) 46.3 (1984): 440-464.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> cressie_read(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> cressie_read(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> cressie_read(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """
    return power_divergence(
        X=X, Y=Y, Z=Z, data=data, boolean=boolean, lambda_="cressie-read", **kwargs
    )


def power_divergence(X, Y, Z, data, boolean=True, lambda_="cressie-read", **kwargs):
    """
    Computes the Cressie-Read power divergence statistic [1]. The null hypothesis
    for the test is X is independent of Y given Z. A lot of the frequency comparision
    based statistics (eg. chi-square, G-test etc) belong to power divergence family,
    and are special cases of this test.

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set

    Y: int, string, hashable object
        A variable name contained in the data set, different from X

    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    lambda_: float or string
        The lambda parameter for the power_divergence statistic. Some values of
        lambda_ results in other well known tests:
            "pearson"             1          "Chi-squared test"
            "log-likelihood"      0          "G-test or log-likelihood"
            "freeman-tuckey"     -1/2        "Freeman-Tuckey Statistic"
            "mod-log-likelihood"  -1         "Modified Log-likelihood"
            "neyman"              -2         "Neyman's statistic"
            "cressie-read"        2/3        "The value recommended in the paper[1]"

    boolean: bool
        If boolean=True, an additional argument `significance_level` must
            be specified. If p_value of the test is greater than equal to
            `significance_level`, returns True. Otherwise returns False.

        If boolean=False, returns the chi2 and p_value of the test.

    Returns
    -------
    If boolean = False, Returns 3 values:
        chi: float
            The chi-squre test statistic.

        p_value: float
            The p_value, i.e. the probability of observing the computed chi-square
            statistic (or an even higher value), given the null hypothesis
            that X \u27C2 Y | Zs.

        dof: int
            The degrees of freedom of the test.

    If boolean = True, returns:
        independent: boolean
            If the p_value of the test is greater than significance_level, returns True.
            Else returns False.

    References
    ----------
    [1] Cressie, Noel, and Timothy RC Read. "Multinomial goodness‐of‐fit tests."
    Journal of the Royal Statistical Society: Series B (Methodological) 46.3 (1984): 440-464.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    >>> data['E'] = data['A'] + data['B'] + data['C']
    >>> chi_square(X='A', Y='C', Z=[], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D'], data=data, boolean=True, significance_level=0.05)
    True
    >>> chi_square(X='A', Y='B', Z=['D', 'E'], data=data, boolean=True, significance_level=0.05)
    False
    """

    # Step 1: Check if the arguments are valid and type conversions.
    if hasattr(Z, "__iter__"):
        Z = list(Z)
    else:
        raise (f"Z must be an iterable. Got object type: {type(Z)}")

    if (X in Z) or (Y in Z):
        raise ValueError(
            f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z."
        )

    # Step 2: Do a simple contingency test if there are no conditional variables.
    if len(Z) == 0:
        chi, p_value, dof, expected = stats.chi2_contingency(
            data.groupby([X, Y]).size().unstack(Y, fill_value=0), lambda_=lambda_
        )

    # Step 3: If there are conditionals variables, iterate over unique states and do
    #         the contingency test.
    else:
        chi = 0
        dof = 0
        for z_state, df in data.groupby(Z):
            try:
                c, _, d, _ = stats.chi2_contingency(
                    df.groupby([X, Y]).size().unstack(Y, fill_value=0), lambda_=lambda_
                )
                chi += c
                dof += d
            except ValueError:
                # If one of the values is 0 in the 2x2 table.
                if isinstance(z_state, str):
                    warn(
                        f"Skipping the test {X} \u27C2 {Y} | {Z[0]}={z_state}. Not enough samples"
                    )
                else:
                    z_str = ", ".join(
                        [f"{var}={state}" for var, state in zip(Z, z_state)]
                    )
                    warn(
                        f"Skipping the test {X} \u27C2 {Y} | {z_str}. Not enough samples"
                    )
        p_value = 1 - stats.chi2.cdf(chi, df=dof)

    # Step 4: Return the values
    if boolean:
        return p_value >= kwargs["significance_level"]
    else:
        return chi, dof, p_value
