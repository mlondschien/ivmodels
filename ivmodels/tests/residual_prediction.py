import numpy as np
import scipy
import scipy.optimize
from sklearn.ensemble import RandomForestRegressor

from ivmodels.confidence_set import ConfidenceSet
from ivmodels.models.kclass import KClass
from ivmodels.utils import _check_inputs, _find_roots, oproj, proj


def residual_prediction_test(
    Z,
    X,
    y,
    C=None,
    robust=False,
    nonlinear_model=None,
    fit_intercept=True,
    train_fraction=None,
    upper_clipping_quantile=0.9,
    gamma=0.05,
    seed=0,
):
    """
    Perform the residual prediction test :cite:p:`scheidegger2025residual` for model specification.

    This uses a nonlinear model to test the well specification of an IV model: "Is the
    linear IV model appropriate for the data"? Formally, the null hypothesis is:

    .. math::
       H_0: \\exists \\beta_0 \\in \\mathbb{R}^p \\mathrm{\\ such \\ that \\ } \\mathbb{E}[y - X \\beta | Z] = 0.

    The tests splits the data according to ``train_fraction`` into :math:`y_a, X_a, Z_a`
    and :math:`y_b, X_b, Z_b` and fits a nonlinear model regressing
    :math:`\\hat{\\varepsilon}_a \\sim Z_a` on the residuals
    :math:`\\hat{\\varepsilon}_a := y_a - X_a \\hat \\beta_a` of a two-stage
    least-squares (TSLS) estimator :math:`\\hat \\beta_a: y_a \\sim X_a | Z_a` on
    the "train" data :math:`y_a, X_a, Z_a`.
    The fitted nonlinear model is then used to predict the residuals on the "test" data
    :math:`y_b, X_b, Z_b`, yielding weights
    :math:`\\hat w_b := \\mathrm{nonlinear\\_model}(Z_b)`. Let
    :math:`\\hat \\varepsilon_b := y_b - X_b \\hat \\beta_b` be the residuals of a
    TSLS estimator :math:`\\hat \\beta_b: y_b \\sim X_b | Z_b` on the "test" data and
    :math:`\\hat \\sigma^2` be an estimate of the variance of
    :math:`\\hat w_b \\cdot \\hat \\varepsilon_b` under the null hypothesis. The test
    statistic
    is

    .. math::
         T = \\frac{1}{\\sqrt{n_b}} \\frac{w_b^T \\hat \\varepsilon_b}{\\sqrt{\\hat \\sigma^2}}.

    This is asymptotically standard Gaussian distributed under the null.

    See also the test's `R implementation <https://github.com/cyrillsch/RPIV>`_ by
    Cyrill Scheidegger.

    To avoid the :math:`p`-value lottery due to the random train / test split used in
    the residual prediction test, :cite:t:`scheidegger2025residual` suggest aggregating
    the :math:`p`-values from multiple random splits by taking 2 times the median. This
    results in a conservative :math:`p`-value :cite:t:`meinshausen2009p`.

    Example
    -------
    >>> import numpy as np
    >>> from ivmodels.tests import residual_prediction_test
    >>> from ivmodels.simulate import simulate_gaussian_iv
    >>>
    >>> Z, X, y = ...
    >>>
    >>> ps = np.empty(50)
    >>> for i in range(50):
    ...     _, ps[i] = residual_prediction_test(Z, X, y, seed=i)
    >>>
    >>> print(f"Residual prediction test p-value: {2 * np.median(ps):.3f}")

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
    robust: bool or string, optional, default = False
        Whether to use a heteroskedasticity-robust method to estimate the noise level
        :math:`\\hat \\sigma^2`.
    nonlinear_model: object, optional, default = None
        Object with a ``fit`` and ``predict`` method. If ``None``, uses an
        ``sklearn.ensemble.RandomForestRegressor()``.
    fit_intercept: bool, optional, default = True
        Whether to include an intercept. This is equivalent to centering the inputs.
    train_fraction: float, optional, default = None
        Fraction of data to use to train the nonlinear model. Must be
        between 0 and 1. The remaining data is used to compute the test statistic. If
        ``None``, 0.5 or :math:`e / \\log(n)` is used, whichever is smaller.
    upper_clipping_quantile: float, optional, default = 0.9
        Asymptotic normality requires the nonlinear model's prediction not to put too
        much weight in the tails. To avoid this, we clip its "test set" predictions by
        a certail threshold in absolute value. The threshold is the
        ``upper_clipping_quantile`` of predictions on the "train" data. Must be between 0
        and 1.
    gamma: float, optional, default = 0.05
        A non-negative scalar. Limits the minimum variance of the test statistic to
        gamma times the noise level.
    seed: int, optional, default = 0
        Seed used to generate the random train / test split.

    Returns
    -------
    statistic: float
        The test statistic :math:`\\mathrm{T}`.
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
       meinshausen2009p
    """
    Z, X, y, _, C, _, _ = _check_inputs(Z, X, y, C=C)

    if fit_intercept:
        C = np.hstack([np.ones((Z.shape[0], 1), dtype=C.dtype), C])

    ZC = np.hstack([Z, C])

    n, _ = Z.shape

    if train_fraction is None:
        train_fraction = min(0.5, np.e / np.log(n))
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")

    if nonlinear_model is None:
        nonlinear_model = RandomForestRegressor(random_state=0)
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

    iv_model_a = KClass("tsls", fit_intercept=False).fit(Xa, ya, Za, C=Ca)
    residuals_a = ya - iv_model_a.predict(Xa, C=Ca)
    nonlinear_model.fit(X=ZCa, y=residuals_a)

    iv_model_b = KClass("tsls", fit_intercept=False).fit(Xb, yb, Zb, C=Cb)
    residuals_b = yb - iv_model_b.predict(Xb, C=Cb).flatten()

    predictions_a = nonlinear_model.predict(X=ZCa).flatten()
    upper_clipping_value = np.quantile(np.abs(predictions_a), upper_clipping_quantile)
    wb = nonlinear_model.predict(X=ZCb).flatten()

    if upper_clipping_value == 0:
        wb = np.sign(wb)
    else:
        wb = (
            np.sign(wb)
            * np.minimum(np.abs(wb), upper_clipping_value)
            / upper_clipping_value
        )

    gamma_ = gamma * np.mean(residuals_b**2)

    XCb_proj = np.hstack([proj(np.hstack([Zb, Cb]), Xb), Cb])
    XCb = np.hstack([Xb, Cb])

    if robust:
        # pinv(X) = (X^T @ X)^(-1) @ X^T
        sigma_sq_hat = (
            np.mean(
                (wb - np.linalg.pinv(XCb_proj).T @ XCb.T @ wb) ** 2 * residuals_b**2
            )
            - np.mean(wb * residuals_b) ** 2
        )
    else:
        sigma_sq_hat = np.mean((wb - np.linalg.pinv(XCb_proj).T @ XCb.T @ wb) ** 2)
        sigma_sq_hat *= np.mean(residuals_b**2)

    sigma_sq_hat = max(sigma_sq_hat, gamma_)

    stat = wb.T @ residuals_b / np.sqrt(sigma_sq_hat) / np.sqrt(Xb.shape[0])
    p_value = 1 - scipy.stats.norm.cdf(stat)
    return stat, p_value


def weak_residual_prediction_test(
    Z,
    X,
    y,
    beta,
    C=None,
    robust=False,
    nonlinear_model=None,
    fit_intercept=True,
    train_fraction=None,
    upper_clipping_quantile=0.9,
    gamma=0.05,
    seed=0,
):
    """
    Perform the weak-IV-robust residual prediction test at a fixed ``beta`` :cite:p:`scheidegger2025residual`.

    Unlike :func:`residual_prediction_test`, this test does not estimate :math:`\\beta`
    via TSLS. Instead, it tests the null hypothesis

    .. math::
       H_0(\\beta_0): \\exists \\theta \\in \\mathbb{R}^q \\text{ s.t. }
       \\mathbb{E}[y - X^T \\beta_0 - C^T \\theta \\mid Z, C] = 0

    at the supplied value :math:`\\beta_0`. The test statistic is asymptotically
    standard Gaussian under the null and remains valid under weak or many instruments.

    See also the test's `R implementation <https://github.com/cyrillsch/RPIV>`_ by
    Cyrill Scheidegger (``weak_RPIV_test``).

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, mx)
        Regressors.
    y: np.ndarray of dimension (n,)
        Outcomes.
    beta: np.ndarray of dimension (mx,)
        Coefficients to test.
    C: np.ndarray of dimension (n, mc) or None, optional, default = None
        Included exogenous regressors.
    robust: bool, optional, default = False
        Whether to use the heteroskedasticity-robust variance estimator.
    nonlinear_model: object, optional, default = None
        Object with a ``fit`` and ``predict`` method. If ``None``, uses an
        ``sklearn.ensemble.RandomForestRegressor()``.
    fit_intercept: bool, optional, default = True
        Whether to include an intercept. This is equivalent to centering the inputs.
    train_fraction: float, optional, default = None
        Fraction of data used to train the nonlinear model. If ``None``, uses
        :math:`\\min(0.5, e / \\log(n))`.
    upper_clipping_quantile: float, optional, default = 0.9
        Asymptotic normality requires the nonlinear model's prediction not to put too
        much weight in the tails. To avoid this, we clip its "test set" predictions by
        a certain threshold in absolute value. The threshold is the
        ``upper_clipping_quantile`` of predictions on the "train" data. Must be between 0
        and 1.
    gamma: float, optional, default = 0.05
        A non-negative scalar. Limits the minimum variance of the test statistic to
        gamma times the noise level.
    seed: int, optional, default = 0
        Seed for the random train / test split.

    Returns
    -------
    statistic: float
        The test statistic.
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
    Z, X, y, _, C, _, beta = _check_inputs(Z, X, y, C=C, beta=beta)

    n = Z.shape[0]

    if fit_intercept:
        C = np.hstack([np.ones((n, 1), dtype=C.dtype), C])

    if train_fraction is None:
        train_fraction = min(0.5, np.e / np.log(n))
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")

    if nonlinear_model is None:
        nonlinear_model = RandomForestRegressor(random_state=seed)
    elif not hasattr(nonlinear_model, "fit") or not hasattr(nonlinear_model, "predict"):
        raise ValueError(
            "nonlinear_model must have a `fit` and `predict` method. If you want to "
            "use a different model, please use the `sklearn` interface."
        )

    rng = np.random.default_rng(seed=seed)
    na = int(n * train_fraction)
    mask = np.zeros(n, dtype=bool)
    mask[:na] = True
    rng.shuffle(mask)

    Za, Xa, ya, Ca = Z[mask], X[mask], y[mask], C[mask]
    Zb, Xb, yb, Cb = Z[~mask], X[~mask], y[~mask], C[~mask]

    # The incl. exog. regressors in C are passed to the nonlinear model as input, but
    # are partialed out from the residuals later. This allows us to test only for the
    # causal parameter `beta`, treating the parameter corresponding to the incl. exog.
    # variables as a nuisance parameter.
    if Ca.shape[1] > 0:
        MXa, Mya = oproj(Ca, Xa, ya)
    else:
        MXa, Mya = Xa, ya

    ZCa_ml = np.hstack([Za, Ca[:, fit_intercept:]])
    ZCb_ml = np.hstack([Zb, Cb[:, fit_intercept:]])

    resid_a = Mya - MXa @ beta
    nonlinear_model.fit(X=ZCa_ml, y=resid_a)
    pred_train = nonlinear_model.predict(ZCa_ml)
    upper_clipping_value = np.quantile(np.abs(pred_train), upper_clipping_quantile)
    wb = nonlinear_model.predict(ZCb_ml)

    if upper_clipping_value == 0:
        wb = np.sign(wb)
    else:
        wb = (
            np.sign(wb)
            * np.minimum(np.abs(wb), upper_clipping_value)
            / upper_clipping_value
        )

    rb = yb - Xb @ beta
    if Cb.shape[1] > 0:
        rb_tilde, wb_tilde = oproj(Cb, rb, wb)
    else:
        rb_tilde, wb_tilde = rb, wb

    gamma_ = gamma * np.mean(rb_tilde**2)

    N_val = np.sum(wb_tilde * rb_tilde) / np.sqrt(n - na)

    # Differently to the residual_prediction_test, no variance adjustment due to
    # the estimation of the TSLS is needed here.
    if robust:
        sigma_sq = (
            np.mean(wb_tilde**2 * rb_tilde**2) - np.mean(wb_tilde * rb_tilde) ** 2
        )
    else:
        sigma_sq = np.mean(wb_tilde**2) * np.mean(rb_tilde**2)

    sigma_sq = max(sigma_sq, gamma_)

    stat = N_val / np.sqrt(sigma_sq)
    p_value = 1 - scipy.stats.norm.cdf(stat)
    return stat, p_value


def inverse_weak_residual_prediction_test(
    Z,
    X,
    y,
    C=None,
    alpha=0.05,
    robust=False,
    nonlinear_model=None,
    fit_intercept=True,
    train_fraction=None,
    upper_clipping_quantile=0.9,
    gamma=0.05,
    seed=0,
    tol=1e-6,
    max_value=1e6,
    max_eval=1000,
):
    """
    Compute confidence set for the weak-IV residual prediction test :cite:p:`scheidegger2025residual`.

    This implements weak-IV robust confidence sets for the causal parameter based on
    the `weak_residual_prediction_test` :cite:p:`scheidegger2025residual`. For each
    candidate :math:`\\beta_0`, it tests

    .. math::
       H_0(\\beta_0): \\exists \\theta \\in \\mathbb{R}^q \\text{ s.t. }
       \\mathbb{E}[y - X^T \\beta_0 - C^T \\theta \\mid Z, C] = 0,

    and returns the confidence set

    .. math::
       C_\\alpha = \\{\\beta_0 \\mid \\mathrm{pval}(\\beta_0) \\geq \\alpha\\}.

    Unlike :func:`residual_prediction_test`, this test does not require estimating
    :math:`\\beta` via TSLS and therefore remains valid under weak or many instruments.
    If :math:`C_\\alpha` is empty, the model is misspecified for all candidate values.

    The function is only implemented for scalar :math:`X` (:math:`p = 1`). The
    confidence set boundaries are located by root-finding starting from the TSLS
    estimate of :math:`\\beta` on the full sample.

    Parameters
    ----------
    Z: np.ndarray of dimension (n, k)
        Instruments.
    X: np.ndarray of dimension (n, 1)
        Endogenous regressor (must be scalar).
    y: np.ndarray of dimension (n,)
        Outcomes.
    C: np.ndarray of dimension (n, mc) or None, optional, default = None
        Included exogenous regressors.
    alpha: float, optional, default = 0.05
        Significance level.
    robust: bool, optional, default = False
        Whether to use the heteroskedasticity-robust variance estimator.
    nonlinear_model: object, optional, default = None
        Object with a ``fit`` and ``predict`` method. If ``None``, uses an
        ``sklearn.ensemble.RandomForestRegressor()``.
    fit_intercept: bool, optional, default = True
        Whether to include an intercept. This is equivalent to centering the inputs.
    train_fraction: float, optional, default = None
        Fraction of data used to train the nonlinear model. If ``None``, uses
        :math:`\\min(0.5, e / \\log(n))`.
    upper_clipping_quantile: float, optional, default = 0.9
        Asymptotic normality requires the nonlinear model's prediction not to put too
        much weight in the tails. To avoid this, we clip its "test set" predictions by
        a certain threshold in absolute value. The threshold is the
        ``upper_clipping_quantile`` of predictions on the "train" data. Must be between 0
        and 1.
    gamma: float, optional, default = 0.05
        A non-negative scalar. Limits the minimum variance of the test statistic to
        gamma times the noise level.
    seed: int, optional, default = 0
        Seed for the random train / test split.
    tol: float, optional, default = 1e-6
        Tolerance for the root-finding algorithm used to locate the confidence set
        boundaries.
    max_value: float, optional, default = 1e6
        Maximum absolute value of :math:`\\beta_0` to consider. If the confidence set
        boundary lies beyond this value, the boundary is reported as
        :math:`\\pm \\infty`.
    max_eval: int, optional, default = 1000
        Maximum number of evaluations of the test statistic for the root-finding
        algorithm.

    Returns
    -------
    ConfidenceSet
        The confidence set :math:`C_\\alpha`.

    Raises
    ------
    ValueError:
        If the dimensions of the inputs are incorrect.
    ValueError:
        If ``X`` has more than one column.
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

    if X.shape[1] != 1:
        raise ValueError(
            "inverse_weak_residual_prediction_test only supports scalar X (p=1). "
            f"Got X with {X.shape[1]} columns."
        )

    n = Z.shape[0]

    if fit_intercept:
        C = np.hstack([np.ones((n, 1), dtype=C.dtype), C])

    if train_fraction is None:
        train_fraction = min(0.5, np.e / np.log(n))
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1).")

    if nonlinear_model is None:
        nonlinear_model = RandomForestRegressor(random_state=seed)
    elif not hasattr(nonlinear_model, "fit") or not hasattr(nonlinear_model, "predict"):
        raise ValueError(
            "nonlinear_model must have a `fit` and `predict` method. If you want to "
            "use a different model, please use the `sklearn` interface."
        )

    rng = np.random.default_rng(seed=seed)
    mask = np.zeros(n, dtype=bool)
    na = int(n * train_fraction)
    mask[:na] = True
    rng.shuffle(mask)

    Za, Xa, ya, Ca = Z[mask], X[mask], y[mask], C[mask]
    Zb, Xb, yb, Cb = Z[~mask], X[~mask], y[~mask], C[~mask]

    if Ca.shape[1] > 0:
        MXa, Mya = oproj(Ca, Xa, ya)
    else:
        MXa, Mya = Xa, ya

    ZCa_ml = np.hstack([Za, Ca[:, fit_intercept:]])
    ZCb_ml = np.hstack([Zb, Cb[:, fit_intercept:]])

    # TSLS on the full sample provides a starting point expected to be inside the
    # confidence set under the null. C already contains the intercept column if needed.
    beta_tsls = KClass("tsls", fit_intercept=False).fit(X, y, Z, C=C).coef_[0]

    def f(b0):
        beta = np.array([b0])

        resid_a = Mya - MXa @ beta

        nonlinear_model.fit(X=ZCa_ml, y=resid_a)
        pred_train = nonlinear_model.predict(ZCa_ml)
        upper_clipping_value = np.quantile(np.abs(pred_train), upper_clipping_quantile)
        wb = nonlinear_model.predict(ZCb_ml)
        if upper_clipping_value == 0:
            wb = np.sign(wb)
        else:
            wb = (
                np.sign(wb)
                * np.minimum(np.abs(wb), upper_clipping_value)
                / upper_clipping_value
            )

        rb = yb - Xb @ beta
        if Cb.shape[1] > 0:
            rb_tilde, wb_tilde = oproj(Cb, rb, wb)
        else:
            rb_tilde, wb_tilde = rb, wb

        gamma_ = gamma * np.mean(rb_tilde**2)

        N_val = np.sum(wb_tilde * rb_tilde) / np.sqrt(n - na)

        if robust:
            sigma_sq = (
                np.mean(wb_tilde**2 * rb_tilde**2) - np.mean(wb_tilde * rb_tilde) ** 2
            )
        else:
            sigma_sq = np.mean(wb_tilde**2) * np.mean(rb_tilde**2)

        sigma_sq = max(sigma_sq, gamma_)

        stat = N_val / np.sqrt(sigma_sq)
        p_value = 1 - scipy.stats.norm.cdf(stat)
        return alpha - p_value  # negative means inside confidence set

    if f(beta_tsls) >= 0:
        res = scipy.optimize.minimize_scalar(
            f, bounds=(beta_tsls - max_value, beta_tsls + max_value), method="bounded"
        )
        if f(res.x) >= 0:
            return ConfidenceSet([])
        beta_tsls = res.x

    roots = _find_roots(
        f, beta_tsls, -np.inf, tol=tol, max_value=max_value, max_eval=max_eval
    )
    roots += _find_roots(
        f, beta_tsls, np.inf, tol=tol, max_value=max_value, max_eval=max_eval
    )

    roots = sorted(roots)
    assert len(roots) % 2 == 0
    boundaries = [(left, right) for left, right in zip(roots[::2], roots[1::2])]
    return ConfidenceSet(boundaries=boundaries)
