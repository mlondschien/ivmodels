Changelog
=========

0.3.1 - 2024-07-30
------------------

** Bug fixes:**

- Fixed bug in
  :class:`~ivmodels.tests.conditional_likelihood_ratio.inverse_conditional_likelihood_ratio_test`.

0.3.0 - 2024-07-23
------------------

**New features:**

- New functions :func:`~ivmodels.tests.inverse_lagrange_multiplier_test` and
  :func:`~ivmodels.tests.inverse_conditional_likelihood_ratio_test` to approximate the
  1 dimensional confidence sets by inverting the corresponding (subvector) tests.

- New class :class:`~ivmodels.confidence_sets.ConfidenceSet`.

- New class :class:`~ivmodels.summary.Summary` holding information about the model fit.

- New class :class:`~ivmodels.summary.CoefficientTable` holding a table of coefficients
  and their p-values.

- New method :func:`~ivmodels.models.kclass.KClass.summary` to create a summary of the
  model fit.

- The :class:`~ivmodels.models.kclass.KClass` gets new attributes after fitting a model:
  `endogenous_names_`, `exogenous_names_`,  and `instrument_names_`. If pandas is
  installed, there's also `names_coefs_`.

- The tests :func:`~ivmodels.tests.anderson_rubin_test`,
  :func:`~ivmodels.tests.lagrange_multiplier_test`,
  :func:`~ivmodels.tests.likelihood_ratio_test`, and
  :func:`~ivmodels.tests.wald_test` and their inverses
  :func:`~ivmodels.tests.inverse_anderson_rubin_test`,
  :func:`~ivmodels.tests.inverse_lagrange_multiplier_test`,
  :func:`~ivmodels.tests.inverse_likelihood_ratio_test`, and
  :func:`~ivmodels.tests.inverse_wald_test` now support an additional parameter `D`
  of exogenous covariates to be included in the test. This is not supported for
  the conditional likelihood ratio test.

**Other changes:**

- The function :func:`~ivmodels.tests.lagrange_multiplier_test` is now slightly faster.

- :class:`~ivmodels.models.kclass.KClass` now accepts `pandas.Series` as arguments to
  `y`.

0.2.0 - 2024-06-07
------------------

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