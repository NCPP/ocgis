.. _computation_headline:

Computation
===========

.. warning:: The computational API is considered to be in an `alpha` development stage and may change rapidly. Suggestions are welcome!

OpenClimateGIS offers an extensible computation framework that supports:
  1. NumPy-based array calculations
  2. Temporal grouping (level grouping not supported)
  3. Argumented and multivariate functions (e.g. heat index)
  4. Overload hooks for aggregation operations

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

Defining Custom Functions
-------------------------

Currently, custom calculations must be added to the module :mod:`ocgis.calc.library` to be available to the software. This is a known inconvenience...

Inheritance Structure
~~~~~~~~~~~~~~~~~~~~~

All calculations are classes that inherit from one of three abstract base classes:
 1. :class:`~ocgis.calc.base.OcgFunction`: Functions with no required parameters.
 2. :class:`~ocgis.calc.base.OcgArgFunction`: Functions `with` required parameters.
 3. :class:`~ocgis.calc.base.OcgCvArgFunction`: Functions with or without parameters, but requiring a mulivariate input. A heat index requiring both temperature and humidity is a good example.

-------------------------------------------------

.. autoclass:: ocgis.calc.base.OcgFunction
   :show-inheritance:
   :members: _calculate_, _aggregate_spatial_

-------------------------------------------------

.. autoclass:: ocgis.calc.base.OcgArgFunction
   :show-inheritance:

-------------------------------------------------

.. autoclass:: ocgis.calc.base.OcgCvArgFunction
   :show-inheritance:
   :members: _calculate_, _aggregate_temporal_

Available Functions
-------------------

Click on `Show Source` to the right of the function to get descriptive information and see class-level definitions.

Basic Statistics
~~~~~~~~~~~~~~~~

.. autoclass:: ocgis.calc.library.Max
   :show-inheritance:
   :members: _calculate_
   :undoc-members:

.. autoclass:: ocgis.calc.library.Mean
   :show-inheritance:
   :members: _calculate_
   :undoc-members:

.. autoclass:: ocgis.calc.library.Median
   :show-inheritance:
   :members: _calculate_
   :undoc-members:

.. autoclass:: ocgis.calc.library.Min
   :show-inheritance:
   :members: _calculate_
   :undoc-members:

.. autoclass:: ocgis.calc.library.SampleSize
   :show-inheritance:
   :members: _calculate_
   :undoc-members:

.. autoclass:: ocgis.calc.library.StandardDeviation
   :show-inheritance:
   :members: _calculate_
   :undoc-members:

Multivariate Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ocgis.calc.library.HeatIndex
   :show-inheritance:
   :members: _calculate_
   :undoc-members:

Percentiles
~~~~~~~~~~~

.. autoclass:: ocgis.calc.library.FrequencyPercentile
   :show-inheritance:
   :members: _calculate_
   :undoc-members:

Thresholds
~~~~~~~~~~

.. autoclass:: ocgis.calc.library.Between
   :show-inheritance:
   :members: _calculate_
   :undoc-members:

.. autoclass:: ocgis.calc.library.Duration
   :show-inheritance:
   :members: _calculate_
   :undoc-members:

.. autoclass:: ocgis.calc.library.FrequencyDuration
   :show-inheritance:
   :members: _calculate_
   :undoc-members:

.. autoclass:: ocgis.calc.library.Threshold
   :show-inheritance:
   :members: _calculate_
   :undoc-members:

.. _NumPy masked array functions: http://docs.scipy.org/doc/numpy/reference/maskedarray.html
