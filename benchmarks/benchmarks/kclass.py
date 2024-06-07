from ivmodels.models.kclass import KClass as ivmodelsKClass
from ivmodels.simulate import simulate_gaussian_iv, simulate_guggenberger12


class KClass:
    # dims is (n, mx, k, mc)
    params = [
        ["liml", "tsls", 0.5],
        [(1000, 1, 1, 0), (1000, 2, 100, 0), (1000, 2, 4, 2), "guggenberger12 (k=10)"],
    ]
    param_names = ["kappa", "data"]

    def setup(self, kappa, data):

        if data == "guggenberger12 (k=10)":
            Z, X, y, C, _ = simulate_guggenberger12(n=1000, k=10, seed=0)
        else:
            n, mx, k, mc = data
            Z, X, y, C, _ = simulate_gaussian_iv(n=n, mx=mx, k=k, u=mx, mc=mc)

        self.data = {
            "Z": Z,
            "X": X,
            "y": y,
            "C": C,
        }

        self.kclass = ivmodelsKClass(kappa=kappa)

    def time_fit(self, kappa, data):
        _ = self.kclass.fit(**self.data)
