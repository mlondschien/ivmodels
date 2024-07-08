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
        Name of the test to be used.
    alpha: float
        Significance level :math:`\\alpha` for the confidence sets, e.g., 0.05. The
        confidence of the confidence set will be :math:`1 - \\alpha`


    """

    def __init__(self, kclass, test, alpha):
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
        import ivmodels.tests as tests  # avoid circular imports

        _TESTS = {
            "wald (liml)": (
                partial(tests.wald_test, estimator="liml"),
                partial(tests.inverse_wald_test, estimator="liml"),
            ),
            "wald (tsls)": (
                partial(tests.wald_test, estimator="tsls"),
                partial(tests.inverse_wald_test, estimator="tsls"),
            ),
            "anderson rubin": (
                tests.anderson_rubin_test,
                tests.inverse_anderson_rubin_test,
            ),
            "AR": (tests.anderson_rubin_test, tests.inverse_anderson_rubin_test),
            "lagrange multiplier": (
                tests.lagrange_multiplier_test,
                tests.inverse_lagrange_multiplier_test,
            ),
        }

        if not hasattr(self, "named_coefs_"):
            self.kclass.fit(X, y, Z=Z, C=C)

        self.estimates_ = self.kclass.coef_.tolist()
        self.statistics_ = list()
        self.p_values_ = list()
        self.confidence_sets_ = list()

        (X, Z, C), _ = self.kclass._X_Z_C(X, Z, C, predict=False)

        test_, inverse_test_ = _TESTS[self.test]

        self.feature_names_ = self.kclass.endogenous_names_
        # + self.kclass.exogenous_names_

        idx = 0

        # if self.kclass.fit_intercept:
        #     self.feature_names_ = ["intercept"] + self.feature_names_
        #     self.estimates_ = [self.kclass.intercept_] + self.estimates_
        #     idx -= 1

        for name in self.feature_names_:

            if name in self.kclass.endogenous_names_:
                mask = np.zeros(len(self.kclass.endogenous_names_), dtype=bool)
                mask[idx] = True

                X_, W_, C_, Z_ = X[:, mask], X[:, ~mask], C, Z
                fit_intercept_ = True
                beta_ = np.array([self.kclass.coef_[idx]])

            elif name in self.kclass.exogenous_names_:
                mask = np.zeros(len(self.kclass.exogenous_names_), dtype=bool)
                mask[idx - len(self.kclass.endogenous_names_)] = True

                X_, W_, C_, Z_ = C[:, mask], X, C[:, ~mask], np.hstack([Z, C[:, mask]])
                fit_intercept_ = True
                beta_ = np.array([self.kclass.coef_[idx]])

            elif name == "intercept":
                X_, W_, C_, Z_ = np.ones((X.shape[0], 1)), X, C, Z
                fit_intercept_ = False
                beta_ = np.array([self.kclass.intercept_])

            test_result = test_(
                Z=Z_,
                X=X_,
                W=W_,
                y=y,
                C=C_,
                beta=beta_,
                fit_intercept=fit_intercept_,
                **kwargs,
            )
            self.statistics_.append(test_result[0])
            self.p_values_.append(test_result[1])

            confidence_set = inverse_test_(
                Z=Z_,
                X=X_,
                W=W_,
                y=y,
                C=C_,
                alpha=self.alpha,
                fit_intercept=fit_intercept_,
                **kwargs,
            )

            if isinstance(confidence_set, Quadric):
                confidence_set = ConfidenceSet.from_quadric(confidence_set)

            self.confidence_sets_.append(confidence_set)
            idx += 1

        return self

    def __format__(self, format_spec: str) -> str:  # noqa D
        if not hasattr(self, "estimates_"):
            return "Summary not fitted yet."

        estimate_str = [f"{e:{format_spec}}" for e in self.estimates_]
        statistics_str = [f"{s:{format_spec}}" for s in self.statistics_]
        p_values_str = [f"{p:{format_spec}}" for p in self.p_values_]
        cis_str = [f"{cs:{format_spec}}" for cs in self.confidence_sets_]

        names_len = max(len(name) for name in self.feature_names_)
        coefs_len = max(max(len(e) for e in estimate_str), len("estimate"))
        statistics_len = max(max(len(s) for s in statistics_str), len("statistic"))
        p_values_len = max(max(len(p) for p in p_values_str), len("p-value"))
        cis_len = max(max(len(ci) for ci in cis_str), len("conf. set"))

        string = f"{'': <{names_len}} {'estimate': >{coefs_len}} {'statistic': >{statistics_len}} {'p-value': >{p_values_len}} {'conf. set': >{cis_len}}\n"
        # total_len = names_len + statistics_len + coefs_len + p_values_len + cis_len + 4
        # string += "-" * total_len + "\n"

        for name, estimate, statistic, p_value, ci in zip(
            self.feature_names_, estimate_str, statistics_str, p_values_str, cis_str
        ):
            string += f"{name: <{names_len}} {estimate: >{coefs_len}} {statistic: >{statistics_len}} {p_value: >{p_values_len}} {ci: >{cis_len}}\n"

        return string

    def __str__(self):  # noqa D
        return f"{self:.4g}"
