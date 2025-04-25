import numpy as np
import scipy

from ivmodels.models.kclass import KClass
from ivmodels.utils import _check_inputs, proj


def scheidegger_test(
    Z,
    X,
    y,
    nonlinear_model,
    C=None,
    kappa="tsls",
    fit_intercept=True,
    train_fraction=None,
    clipping_quantile=0.8,
    seed=0,
):
    """
    Perform the residual prediction test for well-specification of the model.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    nonlinear_model: object
        Object with a `fit` and `predict` method.
    C: np.ndarray of dimension (n, mc) or None, optional, default = None
        Included exogenous regressors.
    kappa: str, optional, default = "tsls"
        The instrumental variables estimator to use for the test. E.g., `"tsls"` or
        `"liml"`.
    fit_intercept: bool, optional, default = True
        Whether to include an intercept. This is equivalent to centering the inputs.
    train_fraction: float, optional, default = None
        Fraction of data to use to train the nonlinear model (`estimator`). Must be
        between 0 and 1. The remaining data is used to compute the test statistic. If
        ``None``, 0.5 or ``e / log(n)`` is used, whichever is smaller.
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
        If `train_fraction` is not in (0, 1).

    References
    ----------
    .. bibliography::
       :filter: False

       scheidegger2025residual
    """
    Z, X, y, _, C, _, _ = _check_inputs(Z, X, y, C=C)
    ZC = np.hstack([Z, C])

    n, k = Z.shape

    if train_fraction is None:
        train_fraction = min(0.5, np.e / np.log(n))
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")

    # We split the data into 2 samples: _a and _b. A good choice for n_a is n / log(n).
    # We fit a linear iv estimator beta_a : ya ~ Xa | Za and a nonlinear model
    # (ya - Xa @ beta_a) ~ Za. Under H0: E[y - X beta_a | Z] = 0, there should be no
    # signal in the residuals ya - Xa @ beta_a for the nonlinear model to learn.
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
    clip = np.quantile(np.abs(predictions_a), clipping_quantile)
    wb = np.clip(nonlinear_model.predict(X=ZCb) / clip, -1, 1)

    XCb_proj = np.hstack([proj(np.hstack([Zb, Cb]), Xb), Cb])
    # pinv(X) = (X^T @ X)^(-1) @ X^T
    sigma_sq_hat = (
        np.mean(
            (wb - np.linalg.pinv(XCb_proj).T @ XCb_proj.T @ wb) ** 2 * residuals_b**2
        )
        - np.mean(wb * residuals_b) ** 2
    )

    stat = wb.T @ residuals_b / np.sqrt(sigma_sq_hat) / np.sqrt(n)
    p_value = 1 - scipy.stats.norm.cdf(stat)
    return stat, p_value
