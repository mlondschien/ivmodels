import numpy as np
import pytest

from ivmodels.models.kclass import KClass
from ivmodels.simulate import simulate_gaussian_iv


@pytest.mark.parametrize(
    "test",
    [
        "anderson-rubin",
        "wald",
        "likelihood-ratio",
        "lagrange multiplier",
        "conditional likelihood-ratio",
    ],
)
@pytest.mark.parametrize(
    "n, mx, k, mc, fit_intercept", [(100, 2, 3, 0, False), (100, 4, 10, 2, True)]
)
def test_kclass_summary(test, n, mx, k, mc, fit_intercept):
    Z, X, y, C, _, _ = simulate_gaussian_iv(
        n=n, mx=mx, k=k, mc=mc, mw=0, include_intercept=fit_intercept
    )

    kclass = KClass(kappa="liml", fit_intercept=fit_intercept)
    kclass.fit(X, y, Z, C)

    summary = kclass.summary(X, y, Z, C, test=test)

    assert len(summary.estimates_) == len(kclass.coef_) + (1 if fit_intercept else 0)
    assert len(summary.statistics_) == len(summary.estimates_)
    assert len(summary.p_values_) == len(summary.estimates_)
    assert len(summary.confidence_sets_) == len(summary.estimates_)

    if fit_intercept:
        assert summary.estimates_[0] == kclass.intercept_
        assert np.all(summary.estimates_[1:] == kclass.coef_)
        names = ["intercept"] + kclass.endogenous_names_ + kclass.exogenous_names_
    else:
        assert np.all(summary.estimates_ == kclass.coef_)
        names = kclass.endogenous_names_ + kclass.exogenous_names_

    assert summary.feature_names_ == names

    summary_string = str(summary)
    for name in names:
        assert name in summary_string


@pytest.mark.parametrize(
    "n, mx, k, mc, fit_intercept, names",
    [
        (100, 2, 3, 3, False, ["endogenous_0", "exogenous_2"]),
        (100, 4, 10, 2, True, ["intercept", "exogenous_1"]),
    ],
)
def test_kclass_summary_names(n, mx, k, mc, fit_intercept, names):
    Z, X, y, C, _, _ = simulate_gaussian_iv(
        n=n, mx=mx, k=k, mc=mc, mw=0, include_intercept=fit_intercept
    )

    kclass = KClass(kappa="liml", fit_intercept=fit_intercept)
    kclass.fit(X, y, Z, C=C)

    summary = kclass.summary(X, y, Z, C=C, test="wald", feature_names=names)

    assert len(summary.estimates_) == len(names)
    assert len(summary.statistics_) == len(names)
    assert len(summary.p_values_) == len(names)
    assert len(summary.confidence_sets_) == len(names)

    assert summary.feature_names_ == names

    summary_string = str(summary)
    for name in names:
        assert name in summary_string
