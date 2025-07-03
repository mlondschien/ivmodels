Changelog
=========

0.7.0 - 2025-06-03
------------------

**New features:**

- The :func:`~ivmodels.tests.anderson_rubin.inverse_anderson_rubin_test` now
  supports the GKM critical values by passing ``critical_values="gkm"``.

0.6.0 - 2025-05-22
------------------

**New features:**

- Added the :func:`~ivmodels.tests.residual_prediction.residual_prediction_test` for
  model misspecification.

0.5.3 - 2025-04-21
------------------

**Bug fixes:**

- The classes :class:`~ivmodels.models.kclass.KClass` and
  :class:`~ivmodels.models.anchor_regression.AnchorRegression` now set attributes
  ``n_features_in_`` and ``feature_names_in_`` to comply with sckit-learn SLEP 7 and 10.

0.5.2 - 2024-10-03
------------------

**Bug fixes:**

- The :class:`~ivmodels.summary.Summary` now correctly includes the rank and J test results.

0.5.1 - 2024-09-16
------------------

**Bug fixes:**

- Fixed the ``setuptools`` configuration.

0.5.0 - 2024-08-27
------------------

**New features:**

- The Wald test now supports robust covariance estimation.

- New method ``length`` for :class:`~ivmodels.confidence_set.ConfidenceSet`.

**Other changes:**

- One can now pass the tolerance parameter ``tol`` to the optimization algorithm in
  :func:`~ivmodels.tests.lagrange_multiplier.lagrange_multiplier_test` and
  :func:`~ivmodels.tests.lagrange_multiplier.inverse_lagrange_multiplier_test` via the
  ``kwargs``.

- :class:`~ivmodels.models.kclass.KClass` now raises if ``kappa >= 1`` (as for the
  LIML and TSLS estimators) and the number of instruments is less than the number of
  endogenous regressors.

- The :class:`~ivmodels.summary.Summary` now only includes and prints the results of the
  J-statistic and (multivariate) F-test for instrument strength if this makes sense.

- The docs have been updated and include examples.

0.4.0 - 2024-08-08
------------------

**New features:**

- New test :func:`~ivmodels.tests.j.j_test` of the overidentifying restrictions.

- The tests :func:`~ivmodels.tests.lagrange_multiplier.inverse_lagrange_multiplier_test`
  and
  :func:`~ivmodels.tests.conditional_likelihood_ratio.inverse_conditional_likelihood_ratio_test`
  now possibly return unions of intervals, instead of one conservative large interval.

**Bug fixes:**

- Fixed bug in :func:`~ivmodels.models.kclass.KClass.fit` when ``C`` is not ``None`` and
  :math:`M_{[Z, C]} X` is not full rank.

- Fixed bug in
  :func:`~ivmodels.tests.conditional_likelihood_ratio.inverse_conditional_likelihood_ratio_test`
  when ``k == mw + mx`` and ``C`` is not ``None``.

- Fixed bug in :func:`~ivmodels.utils._characteristic_roots` if
  ``b == np.array([[0]])``. This now correctly returns ``np.inf``.

**Other changes:**

- The :class:`~ivmodels.summary.Summary` now additionally reports the LIML variant of
  the J-statistic.

0.3.1 - 2024-07-30
------------------

**Bug fixes:**

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
  ``endogenous_names_``, ``exogenous_names_``,  and ``instrument_names_``. If pandas is
  installed, there's also ``names_coefs_``.

- The tests :func:`~ivmodels.tests.anderson_rubin_test`,
  :func:`~ivmodels.tests.lagrange_multiplier_test`,
  :func:`~ivmodels.tests.likelihood_ratio_test`, and
  :func:`~ivmodels.tests.wald_test` and their inverses
  :func:`~ivmodels.tests.inverse_anderson_rubin_test`,
  :func:`~ivmodels.tests.inverse_lagrange_multiplier_test`,
  :func:`~ivmodels.tests.inverse_likelihood_ratio_test`, and
  :func:`~ivmodels.tests.inverse_wald_test` now support an additional parameter ``D``
  of exogenous covariates to be included in the test. This is not supported for
  the conditional likelihood ratio test.

**Other changes:**

- The function :func:`~ivmodels.tests.lagrange_multiplier_test` is now slightly faster.

- :class:`~ivmodels.models.kclass.KClass` now accepts ``pandas.Series`` as arguments to
  ``y``.

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
  now use the ``scipy.linalg(..., lapack_driver="gelsy")``. This results in a speedup.

- The numerical integration function
  :func:`~ivmodels.tests.conditional_likelihood_ratio.conditional_likelihood_ratio_critical_value_function`
  has been reparametrized, yielding a speedup.