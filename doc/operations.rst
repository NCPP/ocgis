:tocdepth: 4

.. _operations-api:

==============
Operations API
==============

This page provides an overview of OpenClimateGIS's operations interface. A :ref:`class_reference-label` and :ref:`function_reference-label` are also available. OpenClimateGIS's combined functionality is generally accessed using the operations interface (:class:`~ocgis.OcgOperations`). Potential keyword arguments are described below in `Keyword Arguments`_. Please see :ref:`examples-label` for an overview of using operations with real data.

.. _default-coordinate-system:

Default Coordinate System
=========================

The default coordinate system for OpenClimateGIS is :class:`~ocgis.crs.Spherical`. The representative PROJ.4 string is ``{'a': 6370997, 'no_defs': True, 'b': 6370997, 'proj': 'longlat', 'towgs84': '0,0,0,0,0,0,0'}``. Prior to OpenClimateGIS ``v2.x``, the `WGS84` datum was used by default. This default coordinate system also applies to bounding boxes. If a bounding box coordinate system differs from the default, this must be converted to a dictionary representation:

>>> import shapely
>>> import ocgis
>>> bbox = [-120, 30, -130, 40]
>>> bbox = shapely.geometry.box(*bbox)
>>> geom = {'geom': bbox, 'crs': ocgis.crs.WGS84()}

The correct coordinate system will always be read from file by the driver. The default coordinate system used by OpenClimateGIS may be changed using :ref:`env.DEFAULT_COORDSYS <env.DEFAULT_COORDSYS>`.

``OcgOperations`` Reference
===========================

The basic operations syntax is:

>>> import ocgis
>>> ops = ocgis.OcgOperations(**kwargs)
>>> res = ops.execute()

Some notable features of the operations object:
 1. The only required keyword argument is `dataset`_.
 2. All keyword arguments are exposed as public attributes which may be arbitrarily set using standard syntax:

>>> import ocgis
>>> rd = ocgis.RequestDataset(uri='/path/to/some/dataset', variable='foo')
>>> ops = OcgOperations(dataset=rd)
>>> ops.aggregate = True

 3. Operations may be run in parallel using MPI. See :ref:`parallel-operations` for guidance.
 4. The object **SHOULD NOT** be reused following an execution as the software may add/modify attribute contents. Instantiate a new object following an execution or copy the object appropriately.

Keyword Arguments
-----------------

Additional information on arguments are found in their respective sections.

abstraction
~~~~~~~~~~~

.. note:: OpenClimateGIS uses the ``bounds`` attribute of a NetCDF variable to construct ``'polygon'`` representations of regular grids. If no ``bounds`` attribute is found, the software defaults to the ``'point'`` geometry abstraction.

======================= =============================================================
Value                   Description
======================= =============================================================
``'polygon'`` (default) Represent cells as :class:`shapely.geometry.Polygon` objects.
``'point'``             Represent cells as :class:`shapely.geometry.Point` objects.
======================= =============================================================

add_auxiliary_files
~~~~~~~~~~~~~~~~~~~

If ``True`` (the default), create a new directory and add metadata and other informational files in addition to the converted file. If ``False``, write the target file only to :ref:`dir_output` and do not create a new directory.

aggregate
~~~~~~~~~

=================== ========================================================================================
Value               Description
=================== ========================================================================================
``True``            Selected geometries are combined into a single geometry (see :ref:`appendix-aggregate`).
``False`` (default) Selected geometries are not combined.
=================== ========================================================================================

.. _agg_selection:

agg_selection
~~~~~~~~~~~~~

=================== ===============================================
Value               Description
=================== ===============================================
``True``            Aggregate (union) `geom`_ to a single geometry.
``False`` (default) Leave `geom`_ as is.
=================== ===============================================

The purpose of this data manipulation is to ease the method required to aggregate (union) geometries into arbitrary regions. A simple example would be unioning the U.S. state boundaries of Utah, Nevada, Arizona, and New Mexico into a single polygon representing a "Southwestern Region".

allow_empty
~~~~~~~~~~~

================= ====================================================================================================
Value             Description
================= ====================================================================================================
`True`            Allow the empty set for geometries not geographically coincident with a source geometry.
`False` (default) Raise :class:`~ocgis.exc.EmptyDataNotAllowed` if the empty set is encountered.
================= ====================================================================================================

.. _calc_headline:

calc
~~~~

See the :ref:`computation_headline` page for more details.

.. _calc_grouping_headline:

calc_grouping
~~~~~~~~~~~~~

There are three forms for this argument:

1. **Date Part Grouping**: Any combination of ``'day'``, ``'month'``, and ``'year'``.

>>> calc_grouping = ['day']
>>> calc_grouping = ['month','year']
>>> calc_grouping = ['day','year']

Temporal aggregation splits date/time coordinates into parts and groups them according to `unique combinations` of those parts. If data is grouped by month, then all of the January times would be in one group with all of the August times in another. If a grouping of month and year are applied, then all of the January 2000 times would be in a group with all of the January 2001 times and so on.

Any temporal aggregation applied to a dataset should be consistent with the input data's temporal resolution. For example, aggregating by day, month, and year on daily input dataset is not a reasonable aggregation as the data selected for aggregation will have a sample size of one (i.e. one day per aggregation group).

2. **Summarize Over All**: The string ``'all'`` indicates the entire time domain should be summarized.

>>> calc_grouping = 'all'

3. **Seasonal Groups**: A sequence of integer sequences. Element sequences must be mutually exclusive (i.e. no repeated integers). Representative times for the climatology are chosen as the center month in a sequence (i.e. January in the sequence [12,1,2]).

Month integers map as expected (1=January, 2=February, etc.). The example below constructs a single season composed of March, April, and May. Note the nested lists.

>>> calc_grouping = [[3, 4, 5]]

The next example consumes all the months in a year.

>>> calc_grouping = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

Unique, time sequential seasons are possible with the ``'unique'`` flag:

>>> calc_grouping = [[12, 1, 2], 'unique']

A *unique* season has at least one value associated with each month in the season. If a month is missing, the season will be dropped. The season specification above returns a calculation based on values with date coordinates in:
 * Dec 1900, Jan 1901, Feb 1901
 * Dec 1901, Jan 1902, Feb 1902

It is also possible to group the seasons by year.

>>> calc_grouping = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], 'year']

For example, this returns a calculation based on values with date coordinates in:
 * 1900: Dec, Jan, Feb
 * 1901: Dec, Jan, Feb
 * 1902: Dec, Jan, Feb

.. _calc_raw_headline:

calc_raw
~~~~~~~~

=================== =======================================================================================================
Value               Description
=================== =======================================================================================================
``True``            If :attr:`ocgis.OcgOperations.aggregate` is ``True``, perform computations on raw, unaggregated values.
``False`` (default) Use aggregated values during computation.
=================== =======================================================================================================

callback
~~~~~~~~

A callback function that may be used for custom messaging. This function integrates with the log handler and will receive messages at or above the :attr:`logging.INFO` level.

>>> def callback(percent, message):
>>>     print(percent, message)

conform_units_to
~~~~~~~~~~~~~~~~

Destination units for conversion. If this parameter is set, then the :mod:`cf_units` module must be installed. Setting this parameter will override conformed units set on ``dataset`` objects.

dataset
~~~~~~~

This is the only required parameter. All elements of ``dataset`` will be processed.

A ``dataset`` is the target file(s) or object(s) containing data to process. A ``dataset`` may be:
 1. A file on the local machine or network location accessible by the software (use :class:`~ocgis.RequestDataset`).
 2. A URL to an unsecured OpenDAP dataset (use :class:`~ocgis.RequestDataset`).
 3. An OpenClimateGIS field object (use :class:`~ocgis.Field`). If a :class:`~ocgis.Field` object is used, be aware operations may modify the object inplace.

>>> # A keyword argument dictionary can be used in place of an actual request object.
>>> dataset = {'uri': '/path/to/my/data.nc'}
>>> # Use variable auto-discovery.
>>> from ocgis import RequestDataset
>>> dataset = RequestDataset(uri='/path/to/my/data.nc'}
>>> # Specify the target variable directly.
>>> dataset = RequestDataset(uri='/path/to/my/data.nc', variable='tas')

In version ``v2.x`` the :class:`RequestDatasetCollection` was removed. Use sequences of request dataset or field objects in their place.

.. _dir_output:

dir_output
~~~~~~~~~~

This sets the output folder for any disk formats. If this is ``None`` and :attr:``ocgis.env.DIR_OUTPUT`` is ``None``, then output will be written to the current working directory.

.. _geom:

geom
~~~~

.. warning:: Unless ``aggregate`` or ``agg_selection`` is True, subsetting with multiple geometries to netCDF will raise an error.

If a geometry(s) is provided, it is used to subset `every` :class:`~ocgis.RequestDataset` or :class:`~ocgis.Field` object. Supplying a value of ``None`` (the default) results in the return of the entire spatial domain.

There are a number of ways to parameterize the ``geom`` keyword argument:

1. Bounding Box

This is a list of floats corresponding to: ``[min_x, min_y, max_x, max_y]``. See :ref:`default-coordinate-system` for guidance on coordinate system defaults and usages.

>>> geom = [-120.4, 30.0, -110.3, 41.4]

2. Point

This is a list of floats corresponding to: ``[longitude, latitude]``. See :ref:`default-coordinate-system` for guidance on coordinate system defaults and usages.

>>> geom = [-120.4, 36.5]

3. Using :class:`~ocgis.GeomCabinetIterator`

>>> from ocgis import GeomCabinetIterator
>>> geom = GeomCabinetIterator('state_boundaries', geom_select_uid=[16])

.. _geom key:

4. Using a :class:`~ocgis.GeomCabinet` key

>>> geom = 'state_boundaries'

5. Custom Sequence of Shapely Geometry Dictionaries

The ``'crs'`` key is optional. If it is not included, WGS84 is assumed. The ``'properties'`` key is also optional. See :ref:`default-coordinate-system` for guidance on coordinate system defaults and usages.

>>> geom = [{'geom': Point(x,y), 'properties': {'UGID': 23, 'NAME': 'geometry23'}, 'crs': CoordinateReferenceSystem(epsg=4326)} ,...]

6. Path to a GIS file

>>> geom = '/path/to/shapefile.shp'

.. _geom_select_uid:

geom_select_sql_where
~~~~~~~~~~~~~~~~~~~~~

.. warning:: Single quotes must be used inside double quotes!

If provided, this string will be used as part of a ``SQL WHERE`` clause to select geometries from the source. See the section titled "WHERE" for documentation on supported statements: http://www.gdal.org/ogr_sql.html. This works only for geometries read from file.

>>> geom_select_sql_where = "STATE_NAME = 'Wisconsin'"
>>> geom_select_sql_where = "STATE_NAME in ('Wisconsin', 'Nebraska')"
>>> geom_select_sql_where = "POPULATION > 1500"

geom_select_uid
~~~~~~~~~~~~~~~

Select specific geometries from the target shapefile chosen using :ref:`geom`. The integer sequence selects matching UGID values from the shapefiles. For more information on adding new shapefiles or the requirements of input shapefiles, please see the section titled `Shapefile Data`_.

>>> geom_select_uid = [1, 2, 3]
>>> geom_select_uid = [4, 55]
>>> geom_select_uid = [1]

As clarification, suppose there is a shapefile called ``basins.shp`` (this assumes the folder containing the shapefile has been set as the value for :attr:`ocgis.env.DIR_GEOMCABINET`) with the following attribute table:

==== =======
UGID Name
==== =======
1    Basin A
2    Basin B
3    Basin C
==== =======

If the goal is to subset the data by the boundary of "Basin A" and write the resulting data to netCDF, a call to OpenClimateGIS operations looks like:

>>> import ocgis
>>> rd = ocgis.RequestDataset(uri='/path/to/data.nc', variable='tas')
>>> path = ocgis.OcgOperations(dataset=rd, geom='basins', geom_select_uid=[1], output_format='nc').execute()

geom_uid
~~~~~~~~

All subset geometries must have a unique identifier. The unique identifier allows subsetted data to be linked to the selection geometry. Passing a string value to ``geom_uid`` will overload the default unique identifier :attr:`ocgis.env.DEFAULT_GEOM_UID`. If no unique identifier is available, a one-based unique identifier will be generated having a name with value :attr:`ocgis.env.DEFAULT_GEOM_UID`.

interpolate_spatial_bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``True``, attempt to interpolate bounds coordinates if they are absent. This will also extrapolate exterior bounds to avoid losing spatial coverage.

melted
~~~~~~

If ``False``, variable names will be individual column headers (non-melted). If ``True``, variable names will be placed into a single column.

A non-melted format:

==== ==== ======
TIME TAS  TASMAX
==== ==== ======
1    30.3 40.3
2    32.2 41.7
3    31.7 40.9
==== ==== ======

A melted format:

==== ====== =====
TIME NAME   VALUE
==== ====== =====
1    TAS    30.3
2    TAS    32.2
3    TAS    31.7
1    TASMAX 40.3
2    TASMAX 41.7
3    TASMAX 40.9
==== ====== =====

optimized_bbox_subset
~~~~~~~~~~~~~~~~~~~~~

If ``True``, only perform the bounding box subset ignoring other subsetting procedures like masking within the bounding coordinates. Using this option should result in lower memory requirements and shorter processing times for subsets. Note this assumes the bounding box aligns appropriately with the target grid.

output_crs
~~~~~~~~~~

By default, the output coordinate reference system (CRS) is the CRS of the input :class:`~ocgis.RequestDataset` object. If multiple :class:`~ocgis.RequestDataset` objects are part of an :class:`~ocgis.OcgOperations` call, then ``output_crs`` must be provided if the input CRS values of the :class:`~ocgis.RequestDataset` objects differ. The value for ``output_crs`` is an instance of :class:`~ocgis.crs.CRS`.

>>> import ocgis
>>> output_crs = ocgis.crs.Spherical()

General ``PROJ.4`` and ``EPSG`` codes are supported by coordinate systems. For example, to output data on the GCS North American 1983 coordinate system, you can configure the coordinate system like:

>>> from ocgis import crs
>>> output_crs = crs.CRS(epgs=4269)

You could also use a ``PROJ.4`` string:

>>> from ocgis import crs
>>> output_crs = crs.CRS(proj4='+proj=longlat +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +no_defs')

.. _output_format_headline:

output_format
~~~~~~~~~~~~~

===================== ===============================================================================================================================================================
Value                 Description
===================== ===============================================================================================================================================================
``'ocgis'`` (default) Return a :class:`~ocgis.SpatialCollection` with keys matching `ugid` (see `geom`_). Also see `Spatial Collections`_ for more information on this output format.
``'csv'``             A CSV file representation of the data.
``'csv-shp'``         In addition to a CSV representation, shapefiles with primary key links to the CSV are provided.
``'nc'``              A NetCDF4-CF file. See :ref:`netcdf_output_headline` for additional information on the structure of the NetCDF format.
``'geojson'``         A GeoJSON representation of the data.
``'shp'``             A shapefile representation of the data.
===================== ===============================================================================================================================================================

.. _output_format_options_headline:

output_format_options
~~~~~~~~~~~~~~~~~~~~~

A dictionary of converter-specific options. Options for each converter are listed in the table below.

+---------------+------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
| Output Format | Option                 | Description                                                                                                                            |
+===============+========================+========================================================================================================================================+
| ``'nc'``      | data_model             | The netCDF data model: http://unidata.github.io/netcdf4-python/#netCDF4.Dataset.                                                       |
|               +------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
|               | variable_kwargs        | Dictionary of keyword parameters to use for netCDF variable creation. See: http://unidata.github.io/netcdf4-python/#netCDF4.Variable.  |
|               +------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
|               | unlimited_to_fixedsize | If ``True``, convert the unlimited dimension to fixed size. Only applies to time and level dimensions.                                 |
|               +------------------------+----------------------------------------------------------------------------------------------------------------------------------------+
|               | geom_dim               | The name of the dimension storing aggregated (unioned) outputs. Only applies when ``aggregate is True``.                               |
+---------------+------------------------+----------------------------------------------------------------------------------------------------------------------------------------+


>>> output_format_options = {'data_model': 'NETCDF4_CLASSIC'}
>>> options = {'variable_kwargs': {'zlib': True, 'complevel': 4}}

prefix
~~~~~~

The ``prefix`` provides the name of the output folder (if ``add_auxiliary_files=True``) and the filename prefix for any file output created by OpenClimateGIS.

>>> prefix = 'fn_start'

regrid_destination
~~~~~~~~~~~~~~~~~~

Please see :ref:`esmpy-regridding` for an overview and limitations.

If provided, all :class:`~ocgis.RequestDataset` objects in ``dataset`` will be regridded to match the grid provided in the argument’s object. This argument must be a :class:`~ocgis.RequestDataset` or :class:`~ocgis.Field`.

>>> regrid_destination = ocgis.RequestDataset(uri='/path/to/destination.nc')

regrid_options
~~~~~~~~~~~~~~

A dictionary with regridding options. Please see the documentation for :func:`~ocgis.regrid.base.regrid_field`. Dictionary elements of ``regrid_options`` correspond to the keyword arguments of this function.

>>> import ESMF
>>> regrid_options = {'regrid_method': ESMF.RegridMethod.CONSERVE}

.. _search_radius_mult key:

search_radius_mult
~~~~~~~~~~~~~~~~~~

This is a scalar float value multiplied by the target data's resolution to determine the buffer radius for the point. This is only applicable when subsetting against gridded datasets.

.. note:: Prior to ``v2.x``, this was a float value by default. This was changed to ``None`` in current versions. Hence, point geometries will be used for subsetting and not a buffered point.

select_nearest
~~~~~~~~~~~~~~

If ``True``, the nearest geometry to the centroid of the current selection geometry is returned.

slice
~~~~~

This is a list of integers, ``None``, or *lists* of integers. The values composing the list will be converted to slice objects. For example, to return the first ten time steps:

>>> slc = [None, [0, 10], None, None, None]

The index locations in the above list correspond to:

===== =============================
Index Description
===== =============================
0     Realization / Ensemble Member
1     Time
2     Level
3     Row
4     Column
===== =============================

To select the last time step:

>>> slice = [None, -1, None, None, None]

.. _snippet_headline:

snippet
~~~~~~~

.. note:: The entire spatial domain is returned unless :ref:`geom` is specified.

.. note:: Only applies for pure subsetting for limiting computations use ``time_range`` and/or ``time_region``.

=================== ===========================================================================
Value               Description
=================== ===========================================================================
``True``            Return only the first time point and the first level slice (if applicable).
``False`` (default) Return all data.
=================== ===========================================================================

spatial_operation
~~~~~~~~~~~~~~~~~

========================== =============================================================================================================================================
Value                      Description
========================== =============================================================================================================================================
``"intersects"`` (default) Source geometries touching or overlapping selection geometries are returned (see :ref:`appendix-intersects`).
``"clip"``                 A full geometric intersection is performed between source and selection geometries. New geometries may be created. (see :ref:`appendix-clip`)
========================== =============================================================================================================================================

spatial_reorder
~~~~~~~~~~~~~~~

If ``True``, reorder wrapped coordinates such that the longitude values are in ascending order. Reordering assumes the first row of longitude coordinates are representative of the other longitude coordinate rows. Bounds and corners will be removed in the event of a reorder. Only applies to spherical coordinate systems.

If ``False`` (the default), do not attempt to reorder wrapped spherical longitude coordinates.

.. note:: If ``aggregate=True``, spatial reordering is not possible.

spatial_wrapping
~~~~~~~~~~~~~~~~

Allows control of the wrapped state for all input fields. Only field objects with a wrappable coordinate system are affected. Wrapping operations are applied before all other operations.

================== ======================================================================
Value              Description
================== ======================================================================
``None`` (default) Do not attempt a wrap or unwrap operation.
``"wrap"``         Wrap spherical coordinates to the -180 to 180 longitudinal domain.
``"unwrap"``       Unwrap spherical coordinate to the 0 to 360 longitudinal domain.
================== ======================================================================

time_range
~~~~~~~~~~

Upper and lower bounds for the time dimension subset composed of a two-element sequence of :class:`datetime.datetime`-like objects. If ``None``, return all time points. Using this argument will overload all :class:`~ocgis.RequestDataset` ``time_range`` values.

time_region
~~~~~~~~~~~

A dictionary with keys of 'month' and/or 'year' and values as sequences corresponding to target month and/or year values. Empty region selection for a key may be set to ``None``. Using this argument will overload all :class:`~ocgis.RequestDataset` ``time_region`` values.

>>> time_region = {'month': [6, 7], 'year': [2010, 2011]}
>>> time_region = {'year': [2010]}

time_subset_func
~~~~~~~~~~~~~~~~

Subset the time dimension by an arbitrary function. The functions must take one argument and one keyword. The argument is a vector of :class:`datetime.datetime`-like objects. The keyword argument should be called "bounds" and may be ``None``. If the bounds value is not ``None``, it should expect a n-by-2 array of ``datetime`` objects. The function must return an integer sequence suitable for indexing. For example:

>>> def subset_func(value, bounds=None):
>>>     indices = []
>>>     for ii, v in enumerate(value):
>>>         if v.month == 6:
>>>             indices.append(ii)
>>>     return indices

.. note:: The subset function is applied following ``time_region`` and ``time_range``.

vector_wrap
~~~~~~~~~~~

.. note:: Only applicable for spherical, geographic coordinate systems.

================== =============================================================================================
Value              Description
================== =============================================================================================
``True`` (default) For vector geometry outputs (e.g. ``shp``), ensure output longitudinal domain is -180 to 180.
``False``          Maintain the :class:`~ocgis.RequestDataset`'s longitudinal domain.
================== =============================================================================================

Environment
===========

These are global parameters used by OpenClimateGIS. For those familiar with :mod:`arcpy` programming, this behaves similarly to the :mod:`arcpy.env` module. Any :mod:`ocgis.env` variable be overloaded with system environment variables by setting `OCGIS_<variable-name>`.

:attr:`env.DEFAULT_GEOM_UID` = ``'UGID'``
 The default unique geometry identifier to search for in geometry datasets. This is also the name of the created unique identifier if none exists in the target.

:attr:`env.DIR_DATA` = ``None``
 Directory(s) to search through to find data. If specified, this should be a sequence of directories. It may also be a single directory location. Note that the search may take considerable time if a very high level directory is chosen. If this variable is set, it is only necessary to specify the filename(s) when creating a :class:`~ocgis.RequestDataset`.

:attr:`env.DIR_OUTPUT` = ``None`` (defaults to current working directory)
 The directory where output data is written. OpenClimateGIS creates directories inside which output data is stored unless :attr:`~ocgis.OcgOperations.add_auxiliary_files` is ``False``. If ``None``, it defaults to the current working directory.

.. _env.DIR_GEOMCABINET:

:attr:`env.DIR_GEOMCABINET` = <path-to-directory>
 Location of the geometry directory (e.g. a directory containing shapefiles) for use by :class:`~ocgis.GeomCabinet`. Formerly called ``DIR_SHPCABINET``.

:attr:`env.MELTED` = ``False``
 If ``True``, use a melted tabular format with all variable values collected in a single column.

:attr:`env.OVERWRITE` = ``False``
 .. warning:: Use with caution.

 Set to ``True`` to overwrite existing output folders. This will remove the folder if it exists!

:attr:`env.PREFIX` = ``'ocgis_output'``
 The default prefix to apply to output files. This is also the output folder name.

:attr:`env.SUPPRESS_WARNINGS` = ``True``
 If ``True``, suppress all OpenClimateGIS warning messages to standard out. Warning messages will still be logged.

:attr:`env.USE_CFUNITS` = ``True``
 If ``True``, use :mod:`cfunits` for any unit transformations. This will be automatically set to ``False`` if :mod:`cfunits` is not available for import.

:attr:`env.USE_MEMORY_OPTIMIZATIONS` = ``False``
 If ``True``, some methods will attempt to minimize their memory usage at the expense of computational time.

:attr:`env.USE_SPATIAL_INDEX` = ``True``
 If ``True``, use :mod:`rtree` to create spatial indices for spatial operations. This will be automatically set to ``False`` if :mod:`rtree` is not available for import.

:attr:`env.VERBOSE` = ``False``
 Indicate if additional output information should be printed to terminal.

.. _env.DEFAULT_COORDSYS:

:attr:`env.DEFAULT_COORDSYS` = :class:`ocgis.crs.Spherical`
 The default coordinate system used by OpenClimateGIS.

:attr:`env.USE_NETCDF4_MPI` = ``None``
 If ``None``, detect if it is possible to use ``netCDF4-python``'s MPI asynchronous write capability. Use it if available. If ``True``, do asynchronous writes with ``netCDF4-python``. Set to ``False`` to use synchronous writes always.

Inspecting Data
===============

See :ref:`inspection` for guidance on inspecting datasets.

Spatial Collections
===================

See the :ref:`advanced-subsetting-example` example for :class:`~ocgis.SpatialCollection` usage. Spatial collections are returned by default from :class:`~ocgis.OcgOperations`.
    
Shapefile Data
==============

Shapefiles may be added to the directory mapped by the environment variable :ref:`env.DIR_GEOMCABINET <env.DIR_GEOMCABINET>`.

The shapefile's `geom key`_ is the name of the shapefile. It must have an alphanumeric name with no spaces with the only allowable special character being underscores "_".