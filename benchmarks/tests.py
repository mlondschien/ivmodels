from asv_runner.benchmarks.mark import skip_for_params

from ivmodels.simulate import simulate_gaussian_iv
from ivmodels.tests import (
    anderson_rubin_test,
    conditional_likelihood_ratio_test,
    lagrange_multiplier_test,
    likelihood_ratio_test,
    rank_test,
    wald_test,
)


class Tests:
    # dims is (k, mx, mw, mc)
    params = [
        [100, 1000, 10000],
        [(1, 1, 0, 0), (4, 2, 0, 0), (10, 1, 4, 0), (4, 2, 2, 2)],
    ]
    param_names = ["n", "dims"]

    def setup(self, n, dims):

        k, mx, mw, mc = dims
        Z, X, y, C, W, beta = simulate_gaussian_iv(
            n=n, mx=mx, k=k, u=mx, mc=mc, mw=mw, return_beta=True
        )

        self.data = {"Z": Z, "X": X, "y": y, "C": C, "W": W, "beta": beta}

    def time_anderson_rubin_test(self, n, dims):
        _, _ = anderson_rubin_test(**self.data)

    @skip_for_params(
        [
            (100, (1, 1, 0, 0)),
            (100, (4, 2, 0, 0)),
            (100, (4, 2, 0, 2)),
            (1000, (1, 1, 0, 0)),
            (1000, (4, 2, 0, 0)),
            (1000, (4, 2, 0, 2)),
            (10000, (1, 1, 0, 0)),
            (10000, (4, 2, 0, 0)),
            (10000, (4, 2, 0, 2)),
        ]
    )
    def time_anderson_rubin_test_guggenberger19(self, n, dims):
        _, _, mw, _ = dims
        _, _ = anderson_rubin_test(**self.data, critical_values="guggenberger19")

    def time_lagrange_multiplier_test(self, n, dims):
        _, _ = lagrange_multiplier_test(**self.data)

    def time_likelihood_ratio_test(self, n, dims):
        _, _ = likelihood_ratio_test(**self.data)

    def time_conditional_likelihood_ratio_test_numerical_integration(self, n, dims):
        _, _ = conditional_likelihood_ratio_test(
            **self.data, method="numerical_integration"
        )

    def time_conditional_likelihood_ratio_test_power_series(self, n, dims):
        _, _ = conditional_likelihood_ratio_test(**self.data, method="power_series")

    def time_wald_test_tsls(self, n, dims):
        _, _ = wald_test(**self.data, estimator="tsls")

    def time_wald_test_liml(self, n, dims):
        _, _ = wald_test(**self.data, estimator="liml")

    def test_rank_test(self, n, dims):
        _, _ = rank_test(Z=self.data["Z"], X=self.data["X"])
