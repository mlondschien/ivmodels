from ivmodels.models.kclass import KClass as ivmodelsKClass
from ivmodels.simulate import simulate_gaussian_iv


class KClass:
    # dims is (n, mx, k, mc)
    params = [
        ["liml", "tsls", 0.5],
        [(100, 1, 1, 0), (100, 2, 4, 0), (100, 2, 4, 2), (1000, 1, 1, 0)],
    ]
    param_names = ["kappa", "dims"]

    def setup(self, kappa, dims):

        n, mx, k, mc = dims
        Z, X, y, C, _ = self.data = simulate_gaussian_iv(n=n, mx=mx, k=k, u=mx, mc=mc)

        self.data = {
            "Z": Z,
            "X": X,
            "y": y,
            "C": C,
        }

        self.kclass = ivmodelsKClass(kappa=kappa)

    def time_fit(self, kappa, dims):
        _ = self.kclass.fit(**self.data)
