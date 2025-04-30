import numpy as np
import scipy
from sklearn.ensemble import RandomForestRegressor

from ivmodels.models.kclass import KClass
from ivmodels.utils import _check_inputs, proj


def residual_prediction_test(
    Z,
    X,
    y,
    C=None,
    nonlinear_model=None,
    kappa="tsls",
    fit_intercept=True,
    train_fraction=None,
    upper_clipping_quantile=0.9,
    lower_clipping_value=0.1,
    seed=0,
):
    """
    Perform the residual prediction test :cite:p:`scheidegger2025residual` model specification.

    This uses a nonlinear model to test

    .. math::
       H_0: \\exists \\beta_0 \\in \\mathbb{R}^p \\mathrm{\\ such \\ that \\ } \\mathbb{E}[y - X \\beta | Z] = 0.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    C: np.ndarray of dimension (n, mc) or None, optional, default = None
        Included exogenous regressors.
    nonlinear_model: object, optional, default = None
        Object with a ``fit`` and ``predict`` method. If ``None``, uses an
        ``sklearn.ensemble.RandomForestRegressor()``.
    kappa: str, optional, default = "tsls"
        The instrumental variables estimator to use for the test. E.g., ``"tsls"`` or
        ``"liml"``.
    fit_intercept: bool, optional, default = True
        Whether to include an intercept. This is equivalent to centering the inputs.
    train_fraction: float, optional, default = None
        Fraction of data to use to train the nonlinear model (`estimator`). Must be
        between 0 and 1. The remaining data is used to compute the test statistic. If
        ``None``, 0.5 or ``e / log(n)`` is used, whichever is smaller.
    upper_clipping_quantile: float, optional, default = 0.9
        Asymptotic normality requires the nonlinear model's prediction not to put too
        much weight in the tails. To avoid this, we clip its "test set" predictions by
        a certail threshold in absolute value. The threshold is the
        ``upper_clipping_quantile`` of predictions on the "train" data. Must be between 0
        and 1.
    lower_clipping_value: float, optional, default = 0.1
        Asymptotic normality requires that the test statistics variance is not too
        small. We do a pre-test for this. Clipping the predictions of the nonlinear
        model to ``lower_clipping_value`` in absolute value, after rescaling, let's us
        choose a reasonable threshold. Must be between 0 and 1. Set this to 0.0 to
        disable the pre-test.
    seed: int, optional, default = 0
        Seed used to generate the random train / test split.

    Returns
    -------
    statistic: float
        The test statistic :math:`\\mathrm{AR}(\\beta)`.
    p_value: float
        The p-value of the test.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.
    ValueError:
        If ``train_fraction`` is not in (0, 1).
    ValueError:
        If ``nonlinear_model`` does not have a ``fit`` and ``predict`` method.

    References
    ----------
    .. bibliography::
       :filter: False

       scheidegger2025residual
    """
    Z, X, y, _, C, _, _ = _check_inputs(Z, X, y, C=C)
    ZC = np.hstack([Z, C])

    n, _ = Z.shape

    if train_fraction is None:
        train_fraction = min(0.5, np.e / np.log(n))
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")

    if nonlinear_model is None:
        nonlinear_model = RandomForestRegressor(n_estimators=20, random_state=seed)
    elif not hasattr(nonlinear_model, "fit") or not hasattr(nonlinear_model, "predict"):
        raise ValueError(
            "nonlinear_model must have a `fit` and `predict` method. If you want to "
            "use a different model, please use the `sklearn` interface."
        )

    # We split the data into 2 samples: _a and _b. A good choice for n_a is
    # e * n / log(n). We fit a linear iv estimator beta_a : ya ~ Xa | Za and a nonlinear
    # model (ya - Xa @ beta_a) ~ Za. Under H0: E[y - X beta_a | Z] = 0, there should be
    # no signal in the residuals ya - Xa @ beta_a for the nonlinear model to learn.
    # We fit another linear iv estimator beta_b : yb ~ Xb | Zb. Let
    # wb := nonlinear_model.predict(Zb). Then, under H0, a properly rescaled version of
    # 1 / sqrt{nb} * wb.T @ (yb - Xb @ beta_b) / sigma_hat ~ N(0, 1)
    rng = np.random.default_rng(seed=seed)
    mask = np.zeros(n, dtype=bool)
    mask[: int(n * train_fraction)] = True
    rng.shuffle(mask)

    Xa, ya, Za, Ca, ZCa = X[mask], y[mask], Z[mask], C[mask], ZC[mask]
    Xb, yb, Zb, Cb, ZCb = X[~mask], y[~mask], Z[~mask], C[~mask], ZC[~mask]

    iv_model_a = KClass(kappa, fit_intercept=fit_intercept).fit(Xa, ya, Za, C=Ca)
    residuals_a = ya - iv_model_a.predict(Xa, C=Ca)
    nonlinear_model.fit(X=ZCa, y=residuals_a)

    iv_model_b = KClass(kappa, fit_intercept=fit_intercept).fit(Xb, yb, Zb, C=Cb)
    residuals_b = yb - iv_model_b.predict(Xb, C=Cb).flatten()

    predictions_a = nonlinear_model.predict(X=ZCa).flatten()
    upper_clipping_value = np.quantile(np.abs(predictions_a), upper_clipping_quantile)
    wb = nonlinear_model.predict(X=ZCb).flatten()
    wb = np.sign(wb) * np.clip(np.abs(wb), lower_clipping_value, upper_clipping_value)

    gamma = 0.1 * np.mean(residuals_b**2) * lower_clipping_value**2

    XCb_proj = np.hstack([proj(np.hstack([Zb, Cb]), Xb), Cb])
    # pinv(X) = (X^T @ X)^(-1) @ X^T
    sigma_sq_hat = (
        np.mean(
            (wb - np.linalg.pinv(XCb_proj).T @ XCb_proj.T @ wb) ** 2 * residuals_b**2
        )
        - np.mean(wb * residuals_b) ** 2
    )

    if sigma_sq_hat < gamma:  # Pre-test for variance
        return -np.inf, 1

    stat = wb.T @ residuals_b / np.sqrt(sigma_sq_hat) / np.sqrt(n)
    p_value = 1 - scipy.stats.norm.cdf(stat)
    return stat, p_value
