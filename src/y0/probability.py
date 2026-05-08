"""Concrete discrete probability distribution for Kolmogorov axiom verification.

Maps Dafny's ``PMF = map<Outcome, real>`` to a pandas DataFrame with
variable columns and a ``prob`` column.  All operations are exact
(no sampling) — suitable for verifying probability axioms numerically.

Ref: probability.dfy IsDistribution / ProbEvent / ProbCond
"""

from __future__ import annotations

from collections.abc import Collection
from itertools import product as cartesian_product
from typing import Any

import numpy as np
import pandas as pd

from .dsl import Variable

__all__ = ["ConcreteDistribution"]


class ConcreteDistribution:
    """A concrete discrete PMF over a finite set of variables.

    The internal representation is a :class:`pd.DataFrame` where each
    row is an outcome (one value per variable) and the ``prob`` column
    holds the probability mass.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        variables: list[Variable],
        *,
        _cpds: dict | None = None,
        _parents: dict | None = None,
    ) -> None:
        self._df = df
        self._variables = list(variables)
        self._cpds = _cpds
        self._parents = _parents

    @classmethod
    def from_random(
        cls,
        variables: list[Variable],
        n_values: int = 2,
        seed: int | None = None,
    ) -> ConcreteDistribution:
        """Generate a random valid PMF over the Cartesian product of variable values.

        Each variable takes integer values in ``range(n_values)``.
        Probabilities are drawn from a Dirichlet(1, ..., 1) distribution
        (uniform over the simplex) to guarantee a valid PMF.

        :param variables: The variables in the distribution.
        :param n_values: Number of discrete values per variable.
        :param seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        names = [v.name for v in variables]
        domain = list(cartesian_product(*[range(n_values)] * len(variables)))
        probs = rng.dirichlet(np.ones(len(domain)))
        rows = [dict(zip(names, vals)) | {"prob": p} for vals, p in zip(domain, probs)]
        df = pd.DataFrame(rows)
        return cls(df, variables)

    @classmethod
    def from_dag(
        cls,
        directed_edges: list[tuple[Variable, Variable]],
        variables: list[Variable],
        n_values: int = 2,
        seed: int | None = None,
    ) -> ConcreteDistribution:
        """Generate a random Markov-compatible distribution from a DAG.

        Generates random conditional probability tables P(V | pa(V)) for
        each variable, then builds the joint PMF as their product.
        The resulting distribution satisfies the Markov factorization.

        Ref: interventional.dfy MarkovFactorization
        """
        rng = np.random.default_rng(seed)

        # Build parent mapping
        parents: dict[Variable, list[Variable]] = {v: [] for v in variables}
        for u, v in directed_edges:
            parents[v].append(u)

        # Generate random CPTs
        cpds: dict[Variable, dict[tuple[int, ...], np.ndarray]] = {}
        for var in variables:
            pa = parents[var]
            cpds[var] = {}
            if not pa:
                cpds[var][()] = rng.dirichlet(np.ones(n_values))
            else:
                for pa_vals in cartesian_product(*[range(n_values)] * len(pa)):
                    cpds[var][pa_vals] = rng.dirichlet(np.ones(n_values))

        # Build joint PMF by multiplying factors
        names = [v.name for v in variables]
        rows = []
        for vals in cartesian_product(*[range(n_values)] * len(variables)):
            assignment = dict(zip(variables, vals))
            prob = 1.0
            for var in variables:
                pa = parents[var]
                pa_vals = tuple(assignment[p] for p in pa)
                var_val = assignment[var]
                prob *= cpds[var][pa_vals][var_val]
            rows.append(dict(zip(names, vals)) | {"prob": prob})

        df = pd.DataFrame(rows)
        return cls(df, variables, _cpds=cpds, _parents=parents)

    def is_valid(self) -> bool:
        """Check AllNonNeg and SumsToOne. Ref: probability.dfy IsDistribution."""
        return bool((self._df["prob"] >= 0.0).all() and abs(self._df["prob"].sum() - 1.0) < 1e-12)

    def prob_event(self, assignment: dict[Variable, Any]) -> float:
        """P(X₁=x₁, X₂=x₂, ...).  Ref: probability.dfy ProbEvent.

        Returns 0.0 if no rows match the assignment.
        """
        mask = pd.Series(True, index=self._df.index)
        for var, val in assignment.items():
            col = var.name
            if col in self._df.columns:
                mask &= self._df[col] == val
            else:
                return 0.0
        return float(self._df.loc[mask, "prob"].sum())

    def prob_marginal(self, assignment: dict[Variable, Any]) -> float:
        """Marginal probability of a partial assignment.

        Equivalent to summing prob_event over all unspecified variables.
        """
        return self.prob_event(assignment)

    def prob_cond(
        self,
        target: dict[Variable, Any],
        given: dict[Variable, Any],
    ) -> float:
        """P(target | given).  Ref: probability.dfy ProbCond.

        :raises ValueError: If P(given) == 0.
        """
        p_given = self.prob_event(given)
        if p_given == 0.0:
            raise ValueError(f"P({given}) = 0; conditional undefined")
        p_joint = self.prob_event({**target, **given})
        return p_joint / p_given

    def values(self, variable: Variable) -> list[Any]:
        """Return the sorted list of values a variable takes."""
        return sorted(self._df[variable.name].unique().tolist())

    def intervene(self, intervention: dict[Variable, Any]) -> ConcreteDistribution:
        """Return a new distribution after do(X=x) — simple filter+renormalize.

        For flat PMFs (no DAG), this is equivalent to conditioning.
        For proper truncated factorization, use ``do_graph()``.
        """
        mask = pd.Series(True, index=self._df.index)
        for var, val in intervention.items():
            mask &= self._df[var.name] == val
        filtered = self._df.loc[mask].copy()
        total = filtered["prob"].sum()
        if total == 0.0:
            raise ValueError(f"Intervention {intervention} has zero probability")
        filtered["prob"] = filtered["prob"] / total
        return ConcreteDistribution(filtered.reset_index(drop=True), self._variables)

    def do_graph(self, intervention: dict[Variable, Any]) -> ConcreteDistribution:
        """Compute P(* | do(X=x)) via truncated factorization.

        Requires the distribution to have been created via ``from_dag()``.
        Removes the factors for intervened variables and recomputes the
        joint, then renormalizes.

        Ref: interventional.dfy TruncatePMF
        """
        if self._cpds is None or self._parents is None:
            raise ValueError("do_graph() requires a distribution created via from_dag()")

        names = [v.name for v in self._variables]
        n_values = len(self._df[names[0]].unique())
        rows = []
        for vals in cartesian_product(*[range(n_values)] * len(self._variables)):
            assignment = dict(zip(self._variables, vals))
            # Skip rows inconsistent with intervention
            if not all(assignment[v] == val for v, val in intervention.items()):
                continue
            # Truncated product: skip factors for intervened variables
            prob = 1.0
            for var in self._variables:
                if var in intervention:
                    continue
                pa = self._parents[var]
                pa_vals = tuple(assignment[p] for p in pa)
                var_val = assignment[var]
                prob *= self._cpds[var][pa_vals][var_val]
            rows.append(dict(zip(names, [assignment[v] for v in self._variables])) | {"prob": prob})

        df = pd.DataFrame(rows)
        total = df["prob"].sum()
        if total > 0:
            df["prob"] = df["prob"] / total
        return ConcreteDistribution(df.reset_index(drop=True), self._variables)
