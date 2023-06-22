import matplotlib.pyplot as plt
import numpy as np
import scipy

from anchor_regression import KClass
from anchor_regression.utils import (
    anderson_rubin_test,
    asymptotic_confidence_interval,
    inverse_anderson_rubin,
)

# Simulate data
seed = 0

n = 1000
p = 2
q = 3
u = 1

rng = np.random.RandomState(0)

delta = rng.normal(0, 1, (u, p))
gamma = rng.normal(0, 1, (u, 1))

beta = rng.normal(0, 0.1, (p, 1))
Pi = rng.normal(0, 1, (q, p))

U = rng.normal(0, 1, (n, u))
Z = rng.normal(0, 1, (n, q))
X = Z @ Pi + U @ delta + rng.normal(0, 1, (n, p))
y = X @ beta + U @ gamma + rng.normal(0, 1, (n, 1))
y = y.flatten()

# Compute LIML, TSLS, and OLS estimators
liml = KClass(kappa="liml").fit(X, y, Z=Z)
ar_liml = anderson_rubin_test(Z, y - liml.predict(X))

ols = KClass(kappa=0).fit(X, y, Z=Z)
ar_ols = anderson_rubin_test(Z, y - ols.predict(X))

tsls = KClass(kappa=1).fit(X, y, Z=Z)
ar_tsls = anderson_rubin_test(Z, y - tsls.predict(X))

ar_truth = anderson_rubin_test(Z, y - X @ beta)

print(
    f"""
truth: {beta.flatten()} with AR(beta) = {ar_truth[0]}, p-value = {ar_truth[1]}
LIML:  {liml.coef_.flatten()} with AR(beta) = {ar_liml[0]}, p-value = {ar_liml[1]}
OLS:   {ols.coef_.flatten()} with AR(beta) = {ar_ols[0]}, p-value = {ar_ols[1]}
TSLS:  {tsls.coef_.flatten()} with AR(beta) = {ar_tsls[0]}, p-value = {ar_tsls[1]}
"""
)

# Verify that the LIML minimizes the AR test statistic by
# 1. approximating d_beta AR(beta) at beta = b_liml


def ar(beta):  # noqa D
    return anderson_rubin_test(Z, y - X @ beta.reshape(-1, 1))[0]


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

asymp_quadric_05 = asymptotic_confidence_interval(Z, X, y.flatten(), liml.coef_, 0.05)
asymp_boundary_05 = asymp_quadric_05._boundary(error=False)

fig, ax = plt.subplots(figsize=(10, 4.5), ncols=2)

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
    ax[idx].scatter(liml.coef_[0], liml.coef_[1], color="red", label="LIML", marker="o")
    ax[idx].scatter(ols.coef_[0], ols.coef_[1], color="blue", label="OLS", marker="x")
    ax[idx].scatter(
        tsls.coef_[0], tsls.coef_[1], color="green", label="TSLS", marker="x"
    )

    ax[idx].plot(boundary_05[:, 0], boundary_05[:, 1], color="black", label="AR = 0.05")
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
cbar.set_label("log(p-value)", rotation=270)

fig.suptitle(f"n={n}, p={p}, q={q}, u={u}, seed={seed}")
fig.savefig(f"figures/inverse_ar_{n}_{p}_{q}_{u}_{seed}.png")
fig.savefig(f"figures/inverse_ar_{n}_{p}_{q}_{u}_{seed}.eps")
