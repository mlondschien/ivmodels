Changelog
=========

Unreleased
----------

**New features:**

- New method :func:`~ivmodels.simulate.simulate_guggenberger12` to draw from the data
  generating process of Guggenberger (2012).

- The utility functions :func:`~ivmodels.utils.proj` and :func:`~ivmodels.utils.oproj`
  now accept multiple args to be projected. Usage of this results in performance
  improvements.

**Other changes:**

- The utility functions :func:`~ivmodels.utils.proj` and :func:`~ivmodels.utils.oproj`
  now use the `scipy.linalg(..., lapack_driver="gelsy")`. This results in a speedup.

- The numerical integration function
  :func:`~ivmodels.tests.conditional_likelihood_ratio.conditional_likelihood_ratio_critical_value_function`
  has been reparametrized, yielding a speedup.