# Instrumental Variables Regression in Python

`ivmodels` implements

 - *K-Class* estimators, including the Limited Information Maximum Likelihood (*LIML*) and the Two-Stage Least Squares (*TSLS*) estimator.
 - Tests and confidence sets for the parameters of the model, including the *Anderson-Rubin test*, the *Lagrange multiplier test*, the *(conditional) likelihood-ratio test*, and the *Wald test*.
 - Auxiliary tests such as *Anderson's (1951) test of reduced rank* (a multivariate extension to the first-stage F-test), the *J-test* (including its LIML variant), and [Scheidegger et al.'s residual prediction test of well-specification](https://www.arxiv.org/abs/2506.12771).

See the [docs](https://ivmodels.readthedocs.io/en/latest/) and the examples therein for more details.
See [this document](https://arxiv.org/pdf/2508.12474) for an introduction to the estimators, tests, and their properties.

If you use this code, consider citing
```
@article{londschien2025statistician,
  title={A statistician's guide to weak-instrument-robust inference in instrumental variables regression with illustrations in {Python}},
  author={Londschien, Malte},
  journal={arXiv preprint arXiv:2508.12474},
  year={2025}
}
```
and
```
@article{londschien2024weak,
  title={Weak-instrument-robust subvector inference in instrumental variables regression: A subvector Lagrange multiplier test and properties of subvector Anderson-Rubin confidence sets},
  author={Londschien, Malte and B{\"u}hlmann, Peter},
  journal={arXiv preprint arXiv:2407.15256},
  year={2024}
}
```

## Installation

You can install `ivmodels` from `conda` (recommended):
```
conda install -c conda-forge ivmodels
```
or `pip`:
```
pip install ivmodels
```