from functools import partial

import numpy as np

from ivmodels.confidence_set import ConfidenceSet
from ivmodels.quadric import Quadric


class Summary:
    """Class containing summary statistics for a fitted model.

    Parameters
    ----------
    kclass: ivmodels.KClass or child class of ivmodels.models.kclass.KClassMixin
        Fitted model.
    test: str
        Name of the test to be used. One of ``"wald"``, ``"anderson-rubin"``,
        ``"lagrange multiplier"``, ``"likelihood-ratio"``, or
        ``"conditional likelihood-ratio"``.
    alpha: float
        Significance level :math:`\\alpha` for the confidence sets, e.g., 0.05. The
        confidence of the confidence set will be :math:`1 - \\alpha`

    """

    def __init__(self, kclass, test, alpha):
        from ivmodels import KClass  # avoid circular imports

        if not isinstance(kclass, KClass):
            raise ValueError("kclass must be an instance of ivmodels.KClass")

        self.kclass = kclass
        self.test = test
        self.alpha = alpha

    def fit(self, X, y, Z=None, C=None, *args, **kwargs):
        """
        Fit a summary.

        If ``instrument_names`` or ``instrument_regex`` are specified, ``X`` must be a
        pandas DataFrame containing columns ``instrument_names`` and ``Z`` must be
        ``None``. At least one one of ``Z``, ``instrument_names``, and
        ``instrument_regex`` must be specified.
        If ``exogenous_names`` or ``exogenous_regex`` are specified, ``X`` must be a
        pandas DataFrame containing columns ``exogenous_names`` and ``C`` must be
        ``None``.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The training input samples. If ``instrument_names`` or ``instrument_regex``
            are specified, ``X`` must be a pandas DataFrame containing columns
            ``instrument_names``.
        y: array-like, shape (n_samples,)
            The target values.
        Z: array-like, shape (n_samples, n_instruments), optional
            The instrument values. If ``instrument_names`` or ``instrument_regex`` are
            specified, ``Z`` must be ``None``. If ``Z`` is specified,
            ``instrument_names`` and ``instrument_regex`` must be ``None``.
        C: array-like, shape (n_samples, n_exogenous), optional
            The exogenous regressors. If ``exogenous_names`` or ``exogenous_regex`` are
            specified, ``C`` must be ``None``. If ``C`` is specified,
            ``exogenous_names`` and ``exogenous_regex`` must be ``None``.
        """
        if not hasattr(self, "named_coefs_"):
            self.kclass.fit(X, y, Z=Z, C=C)

        import ivmodels.tests as tests  # avoid circular imports

        _TESTS = {
            "wald": (
                partial(tests.wald_test, estimator=self.kclass.kappa),
                partial(tests.inverse_wald_test, estimator=self.kclass.kappa),
            ),
            "anderson-rubin": (
                tests.anderson_rubin_test,
                tests.inverse_anderson_rubin_test,
            ),
            "lagrange multiplier": (
                tests.lagrange_multiplier_test,
                tests.inverse_lagrange_multiplier_test,
            ),
            "likelihood-ratio": (
                tests.likelihood_ratio_test,
                tests.inverse_likelihood_ratio_test,
            ),
            "conditional likelihood-ratio": (
                tests.conditional_likelihood_ratio_test,
                tests.inverse_conditional_likelihood_ratio_test,
            ),
        }

        if not str(self.test).lower() in _TESTS:
            raise ValueError(f"Test {self.test} not recognized.")

        n = X.shape[0]

        test_, inverse_test_ = _TESTS.get(str(self.test).lower())

        self.estimates_ = self.kclass.coef_.tolist()
        self.statistics_ = list()
        self.p_values_ = list()
        self.confidence_sets_ = list()

        (X, Z, C), _ = self.kclass._X_Z_C(X, Z, C, predict=False)

        self.feature_names_ = (
            self.kclass.endogenous_names_ + self.kclass.exogenous_names_
        )

        idx = 0
        if self.kclass.fit_intercept:
            self.feature_names_ = ["intercept"] + self.feature_names_
            self.estimates_ = [self.kclass.intercept_] + self.estimates_
            idx -= 1

        for name in self.feature_names_:
            if name in self.kclass.endogenous_names_:
                mask = np.zeros(len(self.kclass.endogenous_names_), dtype=bool)
                mask[idx] = True

                X_, W_, C_, D_ = X[:, mask], X[:, ~mask], C, np.zeros((n, 0))
                fit_intercept_ = True

            elif name in self.kclass.exogenous_names_:
                mask = np.zeros(len(self.kclass.exogenous_names_), dtype=bool)
                mask[idx - len(self.kclass.endogenous_names_)] = True

                X_, W_, C_, D_ = np.zeros((n, 0)), X, C[:, ~mask], C[:, mask]
                fit_intercept_ = True

            elif name == "intercept":
                X_, W_, C_, D_ = np.zeros((n, 0)), X, C, np.ones((n, 1))
                fit_intercept_ = False

            test_result = test_(
                Z=Z,
                X=X_,
                W=W_,
                y=y,
                C=C_,
                D=D_,
                beta=np.array([0]),
                fit_intercept=fit_intercept_,
                **kwargs,
            )
            self.statistics_.append(test_result[0])
            self.p_values_.append(test_result[1])

            confidence_set = inverse_test_(
                Z=Z,
                X=X_,
                W=W_,
                y=y,
                C=C_,
                D=D_,
                alpha=self.alpha,
                fit_intercept=fit_intercept_,
                **kwargs,
            )

            if isinstance(confidence_set, Quadric):
                confidence_set = ConfidenceSet.from_quadric(confidence_set)

            self.confidence_sets_.append(confidence_set)
            idx += 1

        self.statistic_, self.p_value_ = test_(
            Z=Z, X=X, y=y, C=C, beta=np.zeros(X.shape[1]), fit_intercept=True
        )
        self.f_statistic_, self.f_p_value_ = tests.rank_test(
            Z, X, C=C, fit_intercept=True
        )
        return self

    def __format__(self, format_spec: str) -> str:  # noqa D
        if not hasattr(self, "estimates_"):
            return "Summary not fitted yet."

        def format_p_value(x):
            return f"{x:{format_spec}}" if np.isnan(x) or x > 1e-16 else "<1e-16"

        estimate_str = [f"{e:{format_spec}}" for e in self.estimates_]
        statistics_str = [f"{s:{format_spec}}" for s in self.statistics_]
        p_values_str = [format_p_value(p) for p in self.p_values_]
        cis_str = [f"{cs:{format_spec}}" for cs in self.confidence_sets_]

        names_len = max(len(name) for name in self.feature_names_)
        coefs_len = max(max(len(e) for e in estimate_str), len("estimate"))
        statistics_len = max(max(len(s) for s in statistics_str), len("statistic"))
        p_values_len = max(max(len(p) for p in p_values_str), len("p-value"))
        cis_len = max(max(len(ci) for ci in cis_str), len("conf. set"))

        string = f"""Summary based on the {self.test} test.

{'': <{names_len}}  {'estimate': >{coefs_len}}  {'statistic': >{statistics_len}}  {'p-value': >{p_values_len}}  {'conf. set': >{cis_len}}
"""
        # total_len = names_len + statistics_len + coefs_len + p_values_len + cis_len + 4
        # string += "-" * total_len + "\n"

        for name, estimate, statistic, p_value, ci in zip(
            self.feature_names_, estimate_str, statistics_str, p_values_str, cis_str
        ):
            string += f"{name: <{names_len}}  {estimate: >{coefs_len}}  {statistic: >{statistics_len}}  {p_value: >{p_values_len}}  {ci: >{cis_len}}\n"

        string += f"""
Endogenous model statistic: {self.statistic_:{format_spec}}, p-value: {format_p_value(self.p_value_)}
(Multivariate) F-statistic: {self.f_statistic_:{format_spec}}, p-value: {format_p_value(self.f_p_value_)}"""
        return string

    def __str__(self):  # noqa D
        return f"{self:.4g}"
