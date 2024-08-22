ivmodels
========

ivmodels is a Python library for instrumental variable estimation.
It implements

 - K-Class estimators, including the Limited Information Maximum Likelihood (LIML) estimator and the Two-Stage Least Squares (TSLS) estimator.
 - Tests and confidence sets for the parameters of the model, including the Anderson-Rubin test, the the Lagrange multiplier test, the (conditional) likelihood-ratio test, and the Wald test.
 - Auxiliary tests such as Anderson's (1951) test of reduced rank (a multivariate extension to the first-stage F-test) and the J-statistic (including its LIML variant).


.. toctree::
   :maxdepth: 1
   :caption: Examples

   Card (1993) Using Geographic Variation in College Proximity to Estimate the Return to Schooling <examples/card1993using.ipynb>
   Tanaka, Camerer, Nguyen (2010) Risk and time preferences: Linkining experimental and household survey data from vietnam <examples/tanaka2010risk.ipynb>
   Angrist and Krueger (1991) Does Compulsory School Attendance Affect Schooling and Earnings? <examples/angrist1991does.ipynb>

Estimators
=============

.. autoclass:: ivmodels.KClass
   :members: fit
   :noindex:

.. autoclass:: ivmodels.models.AnchorRegression
   :members: fit
   :noindex:

.. autoclass:: ivmodels.models.PULSE
   :members: fit
   :noindex:

.. autoclass:: ivmodels.models.SpaceIV
   :members: fit
   :noindex:

Tests
=====

.. automodule:: ivmodels.tests
   :members:
   :noindex:

Summary
=======

.. autoclass:: ivmodels.summary.Summary
   :members:
   :noindex:

.. autoclass:: ivmodels.summary.CoefficientTable
   :members:
   :noindex:

ConfidenceSet
=============

.. autoclass:: ivmodels.confidence_set.ConfidenceSet
   :members:
   :noindex:

Quadric
=======

.. autoclass:: ivmodels.quadric.Quadric
   :members:
   :noindex:


Bibliography
============

.. bibliography::
