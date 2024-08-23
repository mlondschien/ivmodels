ivmodels
========

ivmodels is a Python library for instrumental variable estimation.
It implements

 - K-Class estimators, including the Limited Information Maximum Likelihood (LIML) estimator and the Two-Stage Least Squares (TSLS) estimator.
 - Tests and confidence sets for the parameters of the model, including the Anderson-Rubin test, the the Lagrange multiplier test, the (conditional) likelihood-ratio test, and the Wald test.
 - Auxiliary tests such as Anderson's (1951) test of reduced rank (a multivariate extension of the first-stage F-test) and the J-statistic (including its LIML variant).


Installation
============

You can install the package through conda

::

   conda install ivmodels -c conda-forge

or through pip

::

   pip install ivmodels


.. toctree::
   :maxdepth: 1
   :caption: Examples

   Card (1993) Using Geographic Variation in College Proximity to Estimate the Return to Schooling <examples/card1993using.ipynb>
   Tanaka, Camerer, Nguyen (2010) Risk and time preferences: Linkining experimental and household survey data from vietnam <examples/tanaka2010risk.ipynb>
   Angrist and Krueger (1991) Does Compulsory School Attendance Affect Schooling and Earnings? <examples/angrist1991does.ipynb>
   Acemoglu, Johnson, and Robinson (2001)  The Colonial Origins of Comparative Development: An Empirical Investigation <examples/acemoglu2001colonial.ipynb>

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api


.. toctree::
   :maxdepth: 1
   :caption: Other

   GitHub <https://github.com/mlondschien/ivmodels>
   changelog
