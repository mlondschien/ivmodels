import numpy as np
import pandas as pd
import pytest
import scipy
from glum import GeneralizedLinearRegressor
from sklearn.linear_model import LinearRegression

from ivmodels.models.kclass import KClass, KClassMixin
from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests.anderson_rubin import anderson_rubin_test
from ivmodels.utils import oproj, proj


@pytest.mark.parametrize(
    "kappa, expected",
    [
        ("fuller(1)", 1),
        ("fuller(0.2)", 0.2),
        ("FULLER(4)", 4),
        ("fuller", 1),
        ("FulLeR", 1),
        ("liml", 0),
        ("LIML", 0),
    ],
)
def test__fuller_alpha(kappa, expected):
    assert KClassMixin()._fuller_alpha(kappa) == expected


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("alpha, l1_ratio", [(0, 0), (1, 0), (1, 0.5), (1, 1)])
@pytest.mark.parametrize(
    "n, mx, r, k, u", [(100, 2, 1, 4, 1), (100, 2, 0, 2, 2), (100, 2, 0, 2, 2)]
)
def test_k_class_equal_to_ols(fit_intercept, alpha, l1_ratio, n, mx, r, k, u):
    n = 100

    Z, X, y, C, _ = simulate_gaussian_iv(n, mx, k, u, mw=0, r=r)
    XC = np.hstack([X, C])

    kclass = KClass(
        kappa=0,
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
    ).fit(X=X, C=C, Z=Z, y=y)

    ols = GeneralizedLinearRegressor(
        family="gaussian", alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept
    ).fit(XC, y)

    assert np.allclose(kclass.predict(X=X, C=C), ols.predict(XC))
    assert np.allclose(kclass.coef_, ols.coef_)
    assert np.allclose(kclass.intercept_, ols.intercept_)


@pytest.mark.parametrize(
    "n, mx, r, k, u", [(100, 2, 2, 5, 1), (100, 0, 3, 4, 3), (100, 2, 0, 3, 2)]
)
@pytest.mark.parametrize("kappa", ["tsls", "ols", "liml", 0.5, 1.5])
def test_k_class_intercept_equiv_to_all_ones_in_C(kappa, n, mx, r, k, u):
    n = 100

    Z, X, y, C, _ = simulate_gaussian_iv(n, mx, k=k, u=u, r=r)

    kclass_with = KClass(kappa=kappa, fit_intercept=True)
    kclass_without = KClass(kappa=kappa, fit_intercept=False)

    C1 = np.hstack([C, np.ones((n, 1))])

    kclass_with.fit(X=X, C=C, y=y, Z=Z)
    kclass_without.fit(X=X, C=C1, y=y, Z=Z)

    assert np.allclose(kclass_with.predict(X, C=C), kclass_without.predict(X, C=C1))
    assert np.allclose(kclass_with.coef_, kclass_without.coef_[:-1])
    assert np.allclose(kclass_with.intercept_, kclass_without.coef_[-1])


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("n, mx, k, r, u", [(100, 2, 2, 2, 1), (100, 3, 5, 2, 1)])
@pytest.mark.parametrize("kappa", ["tsls", "ols", "liml", 0.5, 1.5])
def test_equivalence_exogenous_covariates_and_fitting_on_residuals(
    fit_intercept, kappa, n, mx, k, r, u
):
    # It should be equivalent to include exogenous covariates C or to fit a model on the
    # residuals M_C Z, M_C X, M_C y.
    Z, X, y, C, _ = simulate_gaussian_iv(n, mx, k, u, r=r)

    if fit_intercept:
        C = C - C.mean(axis=0)

    kclass1 = KClass(kappa=kappa, fit_intercept=fit_intercept)
    kclass2 = KClass(kappa=kappa, fit_intercept=fit_intercept)

    kclass1.fit(X=X, C=C, y=y, Z=Z)
    kclass2.fit(X=oproj(C, X), y=oproj(C, y), Z=oproj(C, Z))

    np.testing.assert_allclose(kclass1.coef_[:mx], kclass2.coef_)
    np.testing.assert_allclose(kclass1.intercept_, kclass2.intercept_)

    if r > 0:
        kclass3 = KClass(kappa=0, fit_intercept=fit_intercept)
        kclass3.fit(X=C, y=y - kclass2.predict(X))
        np.testing.assert_allclose(kclass1.coef_[mx:], kclass3.coef_)


@pytest.mark.parametrize("n, mx, k, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_ar_min_same_with_shortcut(n, mx, k, u):
    Z, X, y, _, _ = simulate_gaussian_iv(n, mx, k, u)

    X_proj = proj(Z, X)
    y_proj = proj(Z, y)

    ar_min = KClass.ar_min(X, y, Z)
    assert np.allclose(ar_min, KClass.ar_min(X, y, X_proj=X_proj, y_proj=y_proj))
    assert np.allclose(ar_min, KClass.ar_min(X, y, Z=Z, X_proj=X_proj))
    assert np.allclose(ar_min, KClass.ar_min(X, y, Z=Z, y_proj=y_proj))
    assert np.allclose(ar_min, KClass.ar_min(X, y, Z=Z))


@pytest.mark.parametrize("n, mx, k, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_ar_min_positive(n, mx, k, u):
    # If k > mx, then kappa ~ chi2(k-mx) > 0. Else kappa = 0.
    Z, X, y, _, _ = simulate_gaussian_iv(n, mx, k, u)

    ar_min = KClass.ar_min(X, y, Z)

    if k > mx:
        assert ar_min > 0
    else:
        assert np.allclose(ar_min, 0)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("n, mx, k, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
def test_liml_minimizes_anderson_rubin(fit_intercept, n, mx, k, u):
    Z, X, y, _, _ = simulate_gaussian_iv(n, mx, k, u)

    kclass = KClass(kappa="liml", fit_intercept=fit_intercept)
    kclass.fit(X, y, Z=Z)

    if fit_intercept:
        X = X - X.mean(axis=0)
        y = y - y.mean()
        Z = Z - Z.mean(axis=0)

    # TODO: Use C once implemented for tests.
    def ar(beta):
        return anderson_rubin_test(Z, X, y, beta)[0]

    grad = scipy.optimize.approx_fprime(kclass.coef_, ar, 1e-8)
    np.testing.assert_allclose(grad, 0, atol=1e-4)


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("n, mx, k, u", [(100, 2, 2, 1), (100, 2, 5, 2)])
@pytest.mark.parametrize("kappa", [0.2, 0.8])
def test_k_class_normal_equations(fit_intercept, kappa, n, mx, k, u):
    Z, X, y, _, _ = simulate_gaussian_iv(n, mx, k, u)

    k_class = KClass(kappa=kappa, fit_intercept=fit_intercept)
    k_class.fit(X, y, Z)

    if fit_intercept:
        X = X - X.mean(axis=0)
        y = y - y.mean()
        Z = Z - Z.mean(axis=0)

    X_proj = proj(Z, X)

    assert np.allclose(k_class.coef_, k_class._solve_normal_equations(X, y, X_proj))


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("n, mx, k, r, u", [(100, 2, 2, 0, 1), (100, 2, 5, 1, 2)])
def test_k_class_normal_equations_2(fit_intercept, n, mx, k, r, u):
    # If kappa <=1, then the algorithm uses OLS on transformed data. If kappa > 1, then
    # it uses the normal equations. The result should be the same
    Z, X, y, C, _ = simulate_gaussian_iv(n, mx, k, u, r=r)

    kclass1 = KClass(kappa=1 - 1e-10, fit_intercept=fit_intercept)
    kclass2 = KClass(kappa=1 + 1e-10, fit_intercept=fit_intercept)

    kclass1.fit(X, y, Z, C=C)
    kclass2.fit(X, y, Z, C=C)

    assert np.allclose(kclass1.coef_, kclass2.coef_)
    assert np.allclose(kclass1.intercept_, kclass2.intercept_)


@pytest.mark.parametrize("n, mx, k, r, u", [(100, 2, 2, 0, 1), (100, 4, 4, 1, 2)])
def test_liml_equal_to_tsls_in_just_identified_setting(n, mx, k, r, u):
    Z, X, y, C, _ = simulate_gaussian_iv(n, mx, k, u, r=r)
    y = y.flatten()

    liml = KClass(kappa="liml")
    liml.fit(X, y, Z, C=C)

    Xhat = np.hstack(
        [LinearRegression().fit(np.hstack([Z, C]), X).predict(np.hstack([Z, C])), C]
    )
    tsls = LinearRegression().fit(Xhat, y)

    assert np.allclose(liml.coef_, tsls.coef_)
    assert np.allclose(liml.intercept_, tsls.intercept_)


@pytest.mark.parametrize("n, mx, k, r, u", [(100, 2, 2, 0, 1), (100, 4, 4, 1, 2)])
def test_anderson_rubin_at_liml_is_equal_to_ar_min(n, mx, k, r, u):
    Z, X, y, _, _ = simulate_gaussian_iv(n, mx, k, u, r=r)
    y = y.flatten()

    liml = KClass(kappa="liml", fit_intercept=False)
    liml.fit(X, y, Z)

    # TODO: Use C once available for test.
    assert np.allclose(
        anderson_rubin_test(Z, X, y, liml.coef_)[0],
        (n - k) / k * liml.ar_min_,
        atol=1e-8,
    )


# We fit on df with feature names, but predict on X without feature names
def test_kclass_X_Z_C_raises():
    Z = np.random.normal(size=(100, 2))
    X = Z @ np.random.normal(size=(2, 2)) + np.random.normal(size=(100, 2))
    C = np.random.normal(size=(100, 1))
    Y = X @ np.random.normal(size=2) + C @ np.random.normal(size=1) + 1

    df = pd.DataFrame(np.hstack([X, Z, C]), columns=["X1", "X2", "Z1", "Z2", "C1"])

    kclass_1 = KClass(kappa=1, instrument_names=["Z1", "Z2"], exogenous_names=["C1"])
    with pytest.raises(ValueError, match="`Z` must be None"):
        kclass_1.fit(df, Y, Z=Z)

    with pytest.raises(ValueError, match="`C` must be None"):
        kclass_1.fit(df, Y, C=C)

    with pytest.raises(ValueError, match="not found in X: {'Z1'}"):
        kclass_1.fit(df.drop(columns=["Z1"]), Y)
    with pytest.raises(ValueError, match="not found in X: {'C1'}"):
        kclass_1.fit(df.drop(columns=["C1"]), Y)

    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        kclass_1.fit(X, Y)

    kclass_1.fit(df, Y)
    _ = kclass_1.predict(df)
    _ = kclass_1.predict(df.drop(columns=["Z1", "Z2"]))

    kclass_2 = KClass(kappa=1, instrument_regex="Z", exogenous_regex="C")

    with pytest.raises(ValueError, match="`Z` must be None"):
        kclass_2.fit(df, Y, Z=Z)
    with pytest.raises(ValueError, match="`C` must be None"):
        kclass_2.fit(df, Y, C=C)

    with pytest.raises(ValueError, match="No columns in X matched the regex Z"):
        kclass_2.fit(df.drop(columns=["Z1", "Z2"]), Y)
    with pytest.raises(ValueError, match="No columns in X matched the regex C"):
        kclass_2.fit(df.drop(columns=["C1"]), Y)

    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        kclass_2.fit(X, Y)

    kclass_2.fit(df, Y)
    _ = kclass_2.predict(df)
    _ = kclass_2.predict(df.drop(columns=["Z1", "Z2"]))

    kclass_3 = KClass(kappa=1, instrument_regex="1", exogenous_regex="1")
    with pytest.raises(ValueError, match="mes` and `exogenous_regex` must be disjoint"):
        kclass_3.fit(df, Y)
