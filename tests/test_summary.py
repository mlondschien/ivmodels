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
    table = summary.coefficient_table_

    assert len(table.estimates) == len(kclass.coef_) + (1 if fit_intercept else 0)
    assert len(table.statistics) == len(table.estimates)
    assert len(table.p_values) == len(table.estimates)
    assert len(table.confidence_sets) == len(table.estimates)

    if fit_intercept:
        assert table.estimates[0] == kclass.intercept_
        assert np.all(table.estimates[1:] == kclass.coef_)
        names = ["intercept"] + kclass.endogenous_names_ + kclass.exogenous_names_
    else:
        assert np.all(table.estimates == kclass.coef_)
        names = kclass.endogenous_names_ + kclass.exogenous_names_

    assert table.feature_names == names

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
    table = summary.coefficient_table_

    assert len(table.estimates) == len(names)
    assert len(table.statistics) == len(names)
    assert len(table.p_values) == len(names)
    assert len(table.confidence_sets) == len(names)

    assert table.feature_names == names

    summary_string = str(summary)
    for name in names:
        assert name in summary_string
