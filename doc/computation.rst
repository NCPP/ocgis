.. _computation_headline:

=============
 Computation
=============

.. warning:: The computational API is considered to be in an `alpha` development stage and may change rapidly. Suggestions are welcome! The API is stable but there will be changes to the inheritance structure, functionality, and attribution of the class structure. All changes will be backwards compatible.

OpenClimateGIS offers an extensible computation framework that supports:
  1. NumPy-based array calculations
  2. Temporal grouping (level grouping not supported)
  3. Parameters (e.g. threshold) and multivariate functions (e.g. heat index)
  4. Overload hooks for aggregation operations

Computations Described
======================

Computations are applied following any initial subsetting by time, level, or geometry. If data is spatially aggregated, any computation is applied to the aggregated data values unless :ref:`calc_raw_headline` is set to ``True``. Computations are applied to "temporal groups" within the target data defined by the :ref:`calc_grouping_headline` parameter. A "temporal group" is a unique set of date parts (e.g. the month of August, the year 2002, January 2004). Data is summarized within the temporal group to produce a single value within the temporal aggregation for each level and spatial coordinate.

As a functional example, the following code replicates (in principle) the computational process used in OpenClimateGIS for calculating the mean of non-leveled (i.e. three-dimensional) data with temporal aggregation:

>>> import numpy as np
>>> from datetime import datetime
...
>>> ## generate a random three-dimensional dataset (time, latitude/Y, longitude/X)
>>> data = np.ma.array(np.random.rand(4,2,2),mask=False)
>>> ## this is an example temporal dimension
>>> temporal = np.array([datetime(2001,8,1),datetime(2001,8,2),datetime(2001,9,1),datetime(2001,9,2)])
>>> ## assuming a calc_grouping of ['month'], split the data into monthly groups (OpenClimateGIS uses a boolean array here)
>>> aug,sept = data[0:2,:,:],data[2:,:,:]
>>> ## calculate means along the temporal axis
>>> mu_aug,mu_sept = [np.ma.mean(d,axis=0) for d in [aug,sept]]
>>> ## recombine the data
>>> ret = np.vstack((mu_aug,mu_sept))
>>> ret.shape
(2, 2, 2)

It is possible to write functions that do not use a temporal aggregation. In these cases, the function output will have the same shape as the input - as opposed to being reduced by temporal aggregation.

In addition, sample size is always calculated and returned in any calculation output file (not currently supported for multivariate calculations).

Masked data is respected throughout the computational process. These data are assumed to be missing. Hence, they are not used in the sample size calculation.

Temporal and Spatial Aggregation
--------------------------------

It is possible to overload methods for temporal and/or spatial aggregation in any function. This is described in greater detail in the section :ref:`defining_custom_functions`. If the source code method is not defined (i.e. not overloaded), it is a mean (for temporal) and a weighted average (for spatial). For ease-of-programming and potential speed-ups through NumPy, temporal aggregation is performed within the function unless that function may operate on single values (i.e. mean v. logarithm). In this case, a method overload is required to accomodate temporal aggregations.

Using Computations
==================

.. warning:: Always use `NumPy masked array functions`_!! Standard array functions may not be compatible with masked variables.

Computations are applied by passing a list of "function dictionaries" to the :ref:`calc_headline` argument of the :class:`~ocgis.OcgOperations` object. The other two relevant arguments are :ref:`calc_raw_headline` and :ref:`calc_grouping_headline`.

In its simplest form, a "function dictionary" is composed of a ``'func'`` key and a ``'name'`` key. The ``'func'`` key corresponds to the ``key`` attribute of the function class. The ``'name'`` key in the "function dictionary" is required and is a user-supplied alias. This is required to allow multiple calculations with the same function names to be performed with different parameters (in a single request).

Functions currently available are listed below: :ref:`available_functions`. In the case where a function does not expose a ``key`` attribute, the ``'func'`` value is the lower case string of the function's class name (i.e. Mean = 'mean').

For example to calculate a monthly mean and median on a hypothetical daily climate dataset (written to CSV format), an OpenClimateGIS call may look like:

>>> from ocgis import OcgOperations, RequestDataset
...
>>> rd = RequestDataset('/path/to/data', 'tas')
>>> calc = [{'func': 'mean', 'name': 'monthly_mean'}, {'func': 'median', 'name': 'monthly_median'}]
>>> ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=['month'], output_format='csv', prefix='my_calculation')
>>> path = ops.execute()

A calculation with arguments includes a ``'kwds'`` key in the function dictionary:

>>> calc = [{'func': 'between', 'name': 'between_5_10', 'kwds': {'lower': 5, 'upper': 10}}]

If a function takes parameters, those parameters are documented in the :ref:`available_functions` section. The keyword parameter name maps directly to its keyword name in the ``calculate`` method.

There are also keyword arguments common to all calculations:

 * ``'meta_attrs'``: A dictionary containing metadata attributes to attach to the output calculation variable (e.g. NetCDF attributes)

>>> calc = [{'func': 'mean', 'name': 'mean', 'meta_attrs': {'new_attribute': 'the_value'}}]
>>> calc = [{'func': 'mean', 'name': 'mean', 'meta_attrs': {'new_attribute': 5, 'hello': 'attribute'}}]

.. _defining_custom_functions:

Defining Custom Functions
=========================

String-Based Function Expressions
---------------------------------

String-based functions composed of variable aliases and selected NumPy functions are also allowed for the :ref:`calc_headline` argument. The list of enabled NumPy functions is found in the :attr:`ocgis.constants.enabled_numpy_ufuncs` attribute. The string on the left-hand side of the expression will be the name of the output variable. Some acceptable string-based functions are:

>>> calc = 'tas_added=tas+4'
>>> calc = 'es=6.1078*exp(17.08085*(tas-273.16)/(234.175+(tas-273.16)))'
>>> calc = 'diff=tasmax-tasmin'

.. note:: It is not possible to perform any temporal aggregations using string-based function expressions.

Subclassing OpenClimateGIS Function Classes
-------------------------------------------

Once a custom calculation is defined, it must be appended to :class:`ocgis.FunctionRegistry`.

>>> from my_functions import MyCustomFunction
>>> from ocgis import FunctionRegistry
>>> FunctionRegistry.append(MyCustomFunction)

Inheritance Structure
~~~~~~~~~~~~~~~~~~~~~

All calculations are classes that inherit from the following abstract base classes:
 1. :class:`~ocgis.calc.base.AbstractUnivariateFunction`: Functions with no required parameters operating on a single variable.
 2. :class:`~ocgis.calc.base.AbstractUnivariateSetFunction`: Functions with no required parameters opearting on a single variable and reducing along the temporal axis.
 3. :class:`~ocgis.calc.base.AbstractParameterizedFunction`: Functions with input parameters. Functions do not inherit directly from this base class. It used as part of a 'mix-in' to indiciate a function has parameters.
 4. :class:`~ocgis.calc.base.AbstractMultivariateFunction`: Functions operating on two or more variables.

-------------------------------------------------

.. autoclass:: ocgis.calc.base.AbstractFunction
   :show-inheritance:
   :members: calculate, execute, aggregate_spatial, aggregate_temporal, get_output_units, validate, validate_units

-------------------------------------------------

.. autoclass:: ocgis.calc.base.AbstractUnivariateFunction
   :show-inheritance:
   :members: required_units

-------------------------------------------------

.. autoclass:: ocgis.calc.base.AbstractUnivariateSetFunction
   :show-inheritance:
   :members: aggregate_temporal

-------------------------------------------------

.. autoclass:: ocgis.calc.base.AbstractParameterizedFunction
   :show-inheritance:
   :members: parms_definition

-------------------------------------------------

.. autoclass:: ocgis.calc.base.AbstractMultivariateFunction
   :show-inheritance:
   :members: required_units,required_variables

.. _available_functions:

Available Functions
===================

Click on `Show Source` to the right of the function to get descriptive information and see class-level definitions.

Mathematical Operations
-----------------------

.. autoclass:: ocgis.calc.library.math.Sum
   :show-inheritance:
   :members: calculate
   :undoc-members:

.. autoclass:: ocgis.calc.library.math.Convolve1D
   :show-inheritance:
   :members: calculate
   :undoc-members:

Basic Statistics
----------------

.. autoclass:: ocgis.calc.library.statistics.FrequencyPercentile
   :show-inheritance:
   :members: calculate
   :undoc-members:

.. autoclass:: ocgis.calc.library.statistics.Max
   :show-inheritance:
   :members: calculate
   :undoc-members:

.. autoclass:: ocgis.calc.library.statistics.Mean
   :show-inheritance:
   :members: calculate
   :undoc-members:

.. autoclass:: ocgis.calc.library.statistics.Median
   :show-inheritance:
   :members: calculate
   :undoc-members:

.. autoclass:: ocgis.calc.library.statistics.Min
   :show-inheritance:
   :members: calculate
   :undoc-members:

.. autoclass:: ocgis.calc.library.statistics.StandardDeviation
   :show-inheritance:
   :members: calculate
   :undoc-members:

Moving Window / Kernel-Based
----------------------------

.. autoclass:: ocgis.calc.library.statistics.MovingWindow
   :show-inheritance:
   :members: calculate
   :undoc-members:

Multivariate Calculations / Indices
-----------------------------------

.. autoclass:: ocgis.calc.library.index.duration.FrequencyDuration
   :show-inheritance:
   :members: calculate
   :undoc-members:

.. autoclass:: ocgis.calc.library.index.duration.Duration
   :show-inheritance:
   :members: calculate
   :undoc-members:

.. autoclass:: ocgis.calc.library.index.dynamic_kernel_percentile.DynamicDailyKernelPercentileThreshold
   :show-inheritance:
   :members: calculate, get_daily_percentile
   :undoc-members:

.. autoclass:: ocgis.calc.library.index.heat_index.HeatIndex
   :show-inheritance:
   :members: calculate
   :undoc-members:

Thresholds
----------

.. autoclass:: ocgis.calc.library.thresholds.Between
   :show-inheritance:
   :members: calculate
   :undoc-members:

.. autoclass:: ocgis.calc.library.thresholds.Threshold
   :show-inheritance:
   :members: calculate
   :undoc-members:

Calculation using ``icclim`` for ECA Indices
============================================

The optional Python library ``icclim`` (http://icclim.readthedocs.org/en/latest) may be used to calculate the full suite of European Climate Assessment (ECA) indices. To select an ``icclim`` calculation, prefix the name of the indice with the prefix ``'icclim_'``. A list of indices computable with ``icclim`` is available here: http://icclim.readthedocs.org/en/latest/python_api.html#icclim-indice-compute-indice.

For example, to calculate the *TG* indice (mean of daily mean temperature), select the calculation like:

>>> calc = [{'func': 'icclim_TG', 'name': 'TG'}]

.. _NumPy masked array functions: http://docs.scipy.org/doc/numpy/reference/maskedarray.html
