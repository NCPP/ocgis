.. _computation_headline:

Computation
===========

.. warning:: The computational API is considered to be in an `alpha` development stage and may change rapidly. Suggestions are welcome!

OpenClimateGIS offers an extensible computation framework that supports:
  1. NumPy-based array calculations
  2. Temporal grouping (level grouping not supported)
  3. Argumented and multivariate functions (e.g. heat index)
  4. Overload hooks for spatial aggregation - temporal aggregation implicit to calculation function

Using Computations
------------------

.. warning:: Always use `NumPy masked array functions`_!! Standard array functions may not be compatible with masked variables.

Computations are used by providing a dict list to the :ref:`calc_headline` argument of the :class:`~ocgis.OcgOperations` object. The other two important arguments are :ref:`calc_raw_headline` and :ref:`calc_grouping_headline`.

A function dict is composed of a `func` key and a `name` key. The `func` key corresponds to the `name` attribute of the function class. (This is a bit confusing and will be fixed in a later release.) The `name` key in the function dict is required and is a user-supplied alias. This is required to allow multiple calculations to performed with different parameters. Software-generated function names would be confusing.

For example to calculate a monthly mean and median on a hypothetical daily climate dataset (dumped back into a NetCDF), an OpenClimateGIS call may look like:

>>> from ocgis import OcgOperations, RequestDataset
...
>>> rd = RequestDataset('/path/to/data','tas')
>>> calc = [{'func':'mean','name':'monthly_mean'},{'func':'median','name':'monthly_median'}]
>>> ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=['month'],output_format='nc',prefix='my_calculation')
>>> path = ops.execute()

A calculation with arguments includes a `kwds` key in the function dictionary:

>>> calc = [{'func':'between','name':'between_5_10','kwds':{'lower':5,'upper':10}}]

Available Functions
-------------------

Basic Statistics
~~~~~~~~~~~~~~~~

.. autoclass:: ocgis.calc.library.SampleSize()
   :members: _calculate_

Thresholds
~~~~~~~~~~

Multivariate Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _NumPy masked array functions: http://docs.scipy.org/doc/numpy/reference/maskedarray.html
