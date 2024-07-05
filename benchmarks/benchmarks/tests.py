from asv_runner.benchmarks.mark import skip_for_params

from ivmodels.simulate import simulate_gaussian_iv, simulate_guggenberger12
from ivmodels.tests import (
    anderson_rubin_test,
    conditional_likelihood_ratio_test,
    inverse_anderson_rubin_test,
    inverse_lagrange_multiplier_test,
    inverse_likelihood_ratio_test,
    inverse_wald_test,
    lagrange_multiplier_test,
    likelihood_ratio_test,
    rank_test,
    wald_test,
)


class Tests:
    # data is (k, mx, mw, mc)
    params = [
        [1000],
        [(1, 1, 0, 0), (100, 1, 4, 0), (4, 2, 2, 2), "guggenberger12 (k=10)"],
    ]
    param_names = ["n", "data"]

    def setup(self, n, data):

        if data == "guggenberger12 (k=10)":
            Z, X, y, C, W, beta = simulate_guggenberger12(
                n=n, k=10, seed=0, return_beta=True
            )
        else:
            k, mx, mw, mc = data
            Z, X, y, C, W, beta = simulate_gaussian_iv(
                n=n, mx=mx, k=k, u=mx, mc=mc, mw=mw, return_beta=True
            )

        self.data = {"Z": Z, "X": X, "y": y, "C": C, "W": W, "beta": beta}

    def time_anderson_rubin_test(self, n, data):
        _, _ = anderson_rubin_test(**self.data)

    def time_inverse_anderson_rubin_test(self, n, data):
        data = {k: v for k, v in self.data.items() if k != "beta"}
        _ = inverse_anderson_rubin_test(**data)

    @skip_for_params(
        [
            (100, (1, 1, 0, 0)),
            (1000, (1, 1, 0, 0)),
            (10000, (1, 1, 0, 0)),
        ]
    )
    def time_anderson_rubin_test_guggenberger19(self, n, data):
        _, _ = anderson_rubin_test(**self.data, critical_values="guggenberger19")

    def time_lagrange_multiplier_test(self, n, data):
        _, _ = lagrange_multiplier_test(**self.data)

    @skip_for_params(
        [
            (100, (4, 2, 2, 2)),
            (1000, (4, 2, 2, 2)),
            (10000, (4, 2, 0, 2)),
        ]
    )
    def time_inverse_lagrange_multiplier_test(self, n, data):
        data = {k: v for k, v in self.data.items() if k != "beta"}
        _ = inverse_lagrange_multiplier_test(**data)

    def time_likelihood_ratio_test(self, n, data):
        _, _ = likelihood_ratio_test(**self.data)

    def time_inverse_likelihood_ratio_test(self, n, data):
        data = {k: v for k, v in self.data.items() if k != "beta"}
        _ = inverse_likelihood_ratio_test(**data)

    def time_conditional_likelihood_ratio_test_numerical_integration(self, n, data):
        _, _ = conditional_likelihood_ratio_test(
            **self.data, method="numerical_integration"
        )

    def time_conditional_likelihood_ratio_test_power_series(self, n, data):
        _, _ = conditional_likelihood_ratio_test(**self.data, method="power_series")

    def time_wald_test_tsls(self, n, data):
        _, _ = wald_test(**self.data, estimator="tsls")

    def time_inverse_wald_test_tsls(self, n, data):
        data = {k: v for k, v in self.data.items() if k != "beta"}
        _ = inverse_wald_test(**data, estimator="tsls")

    def time_wald_test_liml(self, n, data):
        _, _ = wald_test(**self.data, estimator="liml")

    def time_inverse_wald_test_liml(self, n, data):
        data = {k: v for k, v in self.data.items() if k != "beta"}
        _ = inverse_wald_test(**data, estimator="liml")

    def test_rank_test(self, n, data):
        _, _ = rank_test(Z=self.data["Z"], X=self.data["X"])
