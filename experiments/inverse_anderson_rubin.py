import click
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ivmodels import PULSE, KClass
from ivmodels.tests import (
    anderson_rubin_test,
    asymptotic_confidence_interval,
    bounded_inverse_anderson_rubin,
    inverse_anderson_rubin,
)


@click.command()
@click.option("--n", default=100)
@click.option("--p", default=2)
@click.option("--q", default=2)
@click.option("--u", default=1)
@click.option("--seed", default=0)
def main(n, p, q, u, seed):  # noqa D

    rng = np.random.RandomState(seed)

    delta = rng.normal(0, 1, (u, p))
    gamma = rng.normal(0, 1, (u, 1))

    beta = rng.normal(0, 0.1, (p, 1))
    Pi = rng.normal(0, 1, (q, p))

    U = rng.normal(0, 1, (n, u))
    Z = rng.normal(0, 1, (n, q))
    X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, p))
    y = X @ beta + U @ gamma + rng.normal(0, 1, (n, 1))
    y = y.flatten()

    Z = Z - Z.mean(axis=0)
    X = X - X.mean(axis=0)
    y = y - y.mean()

    # Compute LIML, TSLS, OLS, and PULSE estimates
    liml = KClass(kappa="liml").fit(X, y, Z=Z)
    ar_liml = anderson_rubin_test(Z, y - liml.predict(X))

    ols = KClass(kappa=0).fit(X, y, Z=Z)
    ar_ols = anderson_rubin_test(Z, y - ols.predict(X))

    tsls = KClass(kappa=1).fit(X, y, Z=Z)
    ar_tsls = anderson_rubin_test(Z, y - tsls.predict(X))

    pulse = PULSE(rtol=0.01).fit(X, y, Z=Z)
    ar_pulse = anderson_rubin_test(Z, y - pulse.predict(X))

    ar_truth = anderson_rubin_test(Z, y.reshape(-1, 1) - X @ beta)

    print(
        f"""
    truth: {beta.flatten()} with AR(beta) = {ar_truth[0]}, p-value = {ar_truth[1]}
    LIML:  {liml.coef_.flatten()} with AR(beta) = {ar_liml[0]}, p-value = {ar_liml[1]}
    OLS:   {ols.coef_.flatten()} with AR(beta) = {ar_ols[0]}, p-value = {ar_ols[1]}
    TSLS:  {tsls.coef_.flatten()} with AR(beta) = {ar_tsls[0]}, p-value = {ar_tsls[1]}
    Pulse: {pulse.coef_.flatten()} with AR(beta) = {ar_pulse[0]}, p-value = {ar_pulse[1]}
    """
    )

    n_kclass = 20
    kclass_coefs = np.zeros(shape=(n_kclass, p))

    kappa_kclasses = np.linspace(0, liml.kappa_, n_kclass)
    for it in range(n_kclass):
        kclass_coefs[it, :] = KClass(kappa=kappa_kclasses[it]).fit(X, y, Z=Z).coef_

    # Verify that the LIML minimizes the AR test statistic by
    # 1. approximating d_beta AR(beta) at beta = b_liml
    ar = lambda beta: anderson_rubin_test(Z, y - X @ beta)[0]  # noqa: E731

    grad = scipy.optimize.approx_fprime(liml.coef_.flatten(), ar, 1e-8)
    print(f"Gradient of AR statistic at the LIML estimate: {grad}")

    # 2. approximating min_beta AR(beta)
    beta_min = scipy.optimize.minimize(ar, np.zeros(p)).x
    print(f"\nbeta_min = {beta_min}, AR(beta_min) = {ar(beta_min):.4f}")

    # Verify that (n - q) / q * l_liml / (1 - l_liml) = min_beta AR(beta) = AR(beta_liml)
    print(
        f"lambda_liml: {liml.lambda_liml_}\n(n - q) / q * l_liml / (1 - l_liml) = {(n - q) / q * liml.lambda_liml_ / (1 - liml.lambda_liml_)}"
    )

    quadric_05 = inverse_anderson_rubin(Z, X, y.flatten(), 0.05)
    boundary_05 = quadric_05._boundary(error=False)

    quadric_01 = inverse_anderson_rubin(Z, X, y.flatten(), 0.01)
    boundary_01 = quadric_01._boundary(error=False)

    asymp_quadric_05 = asymptotic_confidence_interval(
        Z, X, y.flatten(), liml.coef_, 0.05
    )
    asymp_boundary_05 = asymp_quadric_05._boundary(error=False)

    asymp_quadric_01 = asymptotic_confidence_interval(
        Z, X, y.flatten(), liml.coef_, 0.01
    )

    p_val = bounded_inverse_anderson_rubin(Z, X)

    first_stage = LinearRegression().fit(Z, X)
    r2 = r2_score(X, first_stage.predict(Z), multioutput="raw_values")
    f_val = (r2 * (n - q)) / ((1 - r2) * 1)

    print(f"p={p_val}, f={f_val}")
    print(
        f"Volumes: AR(0.05): {quadric_05.volume()}, AR(0.01): {quadric_01.volume()}, AS(0.05): {asymp_quadric_05.volume()}, AS(0.01): {asymp_quadric_01.volume()}"
    )
    fig, ax = plt.subplots(figsize=(3 * 10, 3 * 4.5), ncols=2)

    for idx, delta in enumerate([1, 6]):
        x_ = np.linspace(beta[0] - delta, beta[0] + delta, 100)
        y_ = np.linspace(beta[1] - delta, beta[1] + delta, 100)

        xx, yy = np.meshgrid(x_, y_)
        zz = np.zeros(xx.shape)

        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = np.log(
                    anderson_rubin_test(
                        Z, y.flatten() - X @ np.array([xx[i, j], yy[i, j]])
                    )[1]
                )

        im = ax[idx].contourf(xx, yy, zz)

        ax[idx].scatter(beta[0], beta[1], color="black", label="Truth", marker="x")
        ax[idx].scatter(pulse.coef_[0], pulse.coef_[1], color="orange", label="PULSE")
        ax[idx].scatter(
            liml.coef_[0], liml.coef_[1], color="red", label="LIML", marker="o"
        )
        ax[idx].scatter(
            ols.coef_[0], ols.coef_[1], color="blue", label="OLS", marker="x"
        )
        ax[idx].scatter(
            tsls.coef_[0], tsls.coef_[1], color="green", label="TSLS", marker="x"
        )

        ax[idx].plot(
            kclass_coefs[:, 0],
            kclass_coefs[:, 1],
            color="black",
            label="KClass",
            linestyle="dotted",
        )
        ax[idx].plot(
            boundary_05[:, 0], boundary_05[:, 1], color="black", label="AR = 0.05"
        )
        ax[idx].plot(
            boundary_01[:, 0],
            boundary_01[:, 1],
            color="black",
            label="AR = 0.01",
            linestyle="--",
        )
        ax[idx].plot(
            asymp_boundary_05[:, 0],
            asymp_boundary_05[:, 1],
            color="black",
            label="AS = 0.05",
            linestyle="dotted",
        )

        ax[idx].set_xlabel("x1")
        ax[idx].set_ylabel("x2")

        ax[idx].set_xlim(beta[0] - delta, beta[0] + delta)
        ax[idx].set_ylim(beta[1] - delta, beta[1] + delta)

    ax[-1].legend()
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.set_label("log(p-value)", rotation=270, labelpad=10)

    fig.suptitle(
        f"n={n}, p={p}, q={q}, u={u}, $\\alpha_{{name}}$={bounded_inverse_anderson_rubin(Z, X):.3f}"
    )
    fig.savefig(f"figures/inverse_ar_{n}_{p}_{q}_{u}_{seed}.png")
    fig.savefig(f"figures/inverse_ar_{n}_{p}_{q}_{u}_{seed}.eps")


if __name__ == "__main__":
    main()
