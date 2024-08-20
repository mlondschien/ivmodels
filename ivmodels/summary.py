from functools import partial

import numpy as np

from ivmodels.confidence_set import ConfidenceSet
from ivmodels.quadric import Quadric
from ivmodels.utils import _check_inputs


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
    feature_names: list of str, optional
        Names of the features to be included in the summary. If not specified, all
        features will be included.

    Attributes
    ----------
    coefficient_table_: CoefficientTable
        Table containing the estimates, test statistics, p-values, and confidence sets
        for each feature.
    statistic_: float
        Test statistic with null hypothesis that coefficients corresponding to the
        endogenous regressors are jointly zero.
    p_value_: float
        P-value of the test statistic.
    f_statistic_: float
        F-statistic (or multivariate extension, see
        :func:`~ivmodels.tests.rank.rank_test`) with null hypothesis that the
        first-stage coefficient is of reduced rank.
    f_p_value_: float
        P-value of the F-statistic (or multivariate extension).
    """

    def __init__(self, kclass, test, alpha, feature_names=None):
        from ivmodels import KClass  # avoid circular imports

        if not isinstance(kclass, KClass):
            raise ValueError("kclass must be an instance of ivmodels.KClass")

        self.kclass = kclass
        self.test = test
        self.alpha = alpha
        self.feature_names = feature_names

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

        all_feature_names = self.kclass.endogenous_names_ + self.kclass.exogenous_names_
        if self.kclass.fit_intercept:
            all_feature_names = ["intercept"] + all_feature_names

        if self.feature_names is None:
            feature_names = all_feature_names
        else:
            extra_names = [n for n in self.feature_names if n not in all_feature_names]
            if len(extra_names) > 0:
                raise ValueError(
                    f"Extra feature names {extra_names}. Available feature names are "
                    f"{all_feature_names}. Got feature names {self.feature_names}."
                )
            feature_names = self.feature_names

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

        estimates = list()
        statistics = list()
        p_values = list()
        confidence_sets = list()

        (X, Z, C), _ = self.kclass._X_Z_C(X, Z, C, predict=False)
        Z, X, y, _, C, _, _ = _check_inputs(Z, X, y, C=C)

        mx = len(self.kclass.endogenous_names_)
        fit_intercept = self.kclass.fit_intercept

        for name in feature_names:

            idx = np.where(np.array(all_feature_names) == name)[0][0]
            mask = np.zeros(len(all_feature_names), dtype=bool)
            mask[idx] = True

            if fit_intercept:
                mask = mask[1:]  # the first element corresponds to the intercept

            if name == "intercept":
                estimates.append(self.kclass.intercept_)
            else:
                estimates.append(self.kclass.coef_[idx - fit_intercept])

            test_result = test_(
                Z=Z,
                X=X[:, mask[:mx]],
                W=X[:, ~mask[:mx]],
                y=y,
                C=C[:, ~mask[mx:]],
                D=C[:, mask[mx:]] if name != "intercept" else np.ones((n, 1)),
                beta=np.array([0]),
                fit_intercept=fit_intercept and name != "intercept",
                **kwargs,
            )
            statistics.append(test_result[0])
            p_values.append(test_result[1])

            confidence_set = inverse_test_(
                Z=Z,
                X=X[:, mask[:mx]],
                W=X[:, ~mask[:mx]],
                y=y,
                C=C[:, ~mask[mx:]],
                D=C[:, mask[mx:]] if name != "intercept" else np.ones((n, 1)),
                fit_intercept=fit_intercept and name != "intercept",
                alpha=self.alpha,
                **kwargs,
            )
            if isinstance(confidence_set, Quadric):
                confidence_set = ConfidenceSet.from_quadric(confidence_set)

            confidence_sets.append(confidence_set)

        self.coefficient_table_ = CoefficientTable(
            feature_names, estimates, statistics, p_values, confidence_sets
        )

        self.statistic_, self.p_value_ = test_(
            Z=Z, X=X, y=y, C=C, beta=np.zeros(X.shape[1]), fit_intercept=fit_intercept
        )
        self.f_statistic_, self.f_p_value_ = tests.rank_test(
            Z, X, C=C, fit_intercept=fit_intercept
        )
        self.j_statistic_, self.j_p_value_ = tests.j_test(
            Z, X, y=y, C=C, fit_intercept=fit_intercept
        )

        return self

    def __format__(self, format_spec: str) -> str:  # noqa D
        if not hasattr(self, "coefficient_table_"):
            return "Summary not fitted yet."

        string = f"Summary based on the {self.test} test.\n\n"
        string += self.coefficient_table_.__format__(format_spec)

        string += f"""

Endogenous model statistic: {self.statistic_:{format_spec}}, p-value: {_format_p_value(self.p_value_, format_spec)}
(Multivariate) F-statistic: {self.f_statistic_:{format_spec}}, p-value: {_format_p_value(self.f_p_value_, format_spec)}
J-statistic (LIML): {self.j_statistic_:{format_spec}}, p-value: {_format_p_value(self.j_p_value_, format_spec)}"""
        return string

    def __str__(self):  # noqa D
        return f"{self:.4g}"


class CoefficientTable:
    """
    Table with estimates, statistics, p-values, and confidence sets for each feature.

    Parameters
    ----------
    feature_names: list of str
        Names of the features.
    estimates: list of float
        Estimates of the coefficients.
    statistics: list of float
        Test statistics.
    p_values: list of float
        P-values of the test statistics.
    confidence_sets: list of ivmodels.confidence_set.ConfidenceSet
        Confidence sets for the coefficients.
    """

    def __init__(self, feature_names, estimates, statistics, p_values, confidence_sets):
        self.feature_names = feature_names
        self.estimates = estimates
        self.statistics = statistics
        self.p_values = p_values
        self.confidence_sets = confidence_sets

    def __format__(self, format_spec: str) -> str:  # noqa D
        estimate_str = [f"{e:{format_spec}}" for e in self.estimates]
        statistics_str = [f"{s:{format_spec}}" for s in self.statistics]
        p_values_str = [_format_p_value(p, format_spec) for p in self.p_values]
        cis_str = [f"{cs:{format_spec}}" for cs in self.confidence_sets]

        names_len = max(len(name) for name in self.feature_names)
        coefs_len = max(max(len(e) for e in estimate_str), len("estimate"))
        statistics_len = max(max(len(s) for s in statistics_str), len("statistic"))
        p_values_len = max(max(len(p) for p in p_values_str), len("p-value"))
        cis_len = max(max(len(ci) for ci in cis_str), len("conf. set"))

        string = f"{'': <{names_len}}  {'estimate': >{coefs_len}}  {'statistic': >{statistics_len}}  {'p-value': >{p_values_len}}  {'conf. set': >{cis_len}}"

        for name, estimate, statistic, p_value, ci in zip(
            self.feature_names, estimate_str, statistics_str, p_values_str, cis_str
        ):
            string += f"\n{name: <{names_len}}  {estimate: >{coefs_len}}  {statistic: >{statistics_len}}  {p_value: >{p_values_len}}  {ci: >{cis_len}}"

        return string

    def __str__(self):  # noqa D
        return f"{self:.4g}"


def _format_p_value(x, format_spec):
    return f"{x:{format_spec}}" if np.isnan(x) or x > 1e-16 else "<1e-16"
