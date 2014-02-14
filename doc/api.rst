.. _python_api:

==========
Python API
==========

Additional information on keyword arguments can be found below the initial documentation in the `Detailed Argument Information`_ section.

:mod:`ocgis.env`
================

These are global parameters used by OpenClimateGIS. For those familiar with :mod:`arcpy` programming, this behaves similarly to the :mod:`arcpy.env` module. Any :mod:`ocgis.env` variable be overloaded with system environment variables by setting `OCGIS_<variable-name>`.

:attr:`env.DIR_OUTPUT` = `None` (defaults to current working directory)
 The directory where output data is written. OpenClimateGIS always creates directories inside which output data is stored. Also, many of the output formats have multiple output files making a single directory location potentially troubling in terms of file quantity. If `None`, it defaults to the current working directory.

:attr:`env.OVERWRITE` = `False`
 .. warning:: Use with caution.
 
 Set to `True` to overwrite existing output folders. This will remove the folder if it exists!

:attr:`env.PREFIX` = 'ocgis_output'
 The default prefix to apply to output files. This is also the output folder name.

.. _env.DIR_SHPCABINET:

:attr:`env.DIR_SHPCABINET` = <path-to-directory>
 Location of the shapefile directory for use by :class:`~ocgis.ShpCabinet`.

:attr:`env.DIR_DATA` = `None`
 Directory(s) to search through to find data. If specified, this should be a sequence of directories. It may also be a single directory location. Note that the search may take considerable time if a very high level directory is chosen. If this variable is set, it is only necessary to specify the filename(s) when creating a :class:`~ocgis.RequestDataset`.

..
   :attr:`env.SERIAL` = `True`
    If `True`, execute in serial. Only set to `False` if you are confident in your grasp of the software and its internal operation.

   :attr:`env.CORES` = 6
    If operating in parallel (i.e. :attr:`env.SERIAL` = `False`), specify the number of cores to use.

:attr:`env.VERBOSE` = `False`
 Indicate if additional output information should be printed to terminal. (Currently not very useful.)

:class:`ocgis.OcgOperations`
============================

.. autoclass:: ocgis.OcgOperations
   :members: execute

Detailed Argument Information
-----------------------------

Additional information on arguments are found in their respective sections.

dataset
~~~~~~~

A `dataset` is the target file(s) where data is stored. A `dataset` may be on the local machine or network location accessible by the software. Unsecured OpenDAP datasets may also be accessed.

.. autoclass:: ocgis.RequestDataset
   :members: inspect, inspect_as_dct

.. autoclass:: ocgis.RequestDatasetCollection
   :members: update

dir_output
~~~~~~~~~~

This sets the output folder for any disk formats. If this is `None` and `env.DIR_OUTPUT` is `None`, then output will be written to the current working directory.

spatial_operation
~~~~~~~~~~~~~~~~~

======================== =============================================================================================================================================
Value                    Description
======================== =============================================================================================================================================
``intersects`` (default) Source geometries touching or overlapping selection geometries are returned (see :ref:`appendix-intersects`).
``clip``                 A full geometric intersection is performed between source and selection geometries. New geometries may be created. (see :ref:`appendix-clip`)
======================== =============================================================================================================================================

.. _geom:

geom
~~~~

If a geometry(s) is provided, it is used to subset `every` :class:`ocgis.RequestDataset` object. Again, supplying a value of `None` (the default) results in the return of the entire spatial domain. Any shapefiles used for subsetting must include a unique integer attribute called `UGID` and have a WGS84 latitude/longitude geographic coordinate system.

There are a number of ways to parameterize the `geom` keyword argument:

1. Bounding Box

This is a list of floats corresponding to: `[min x, min y, max x, max y]`. The coordinates should be WGS84 geographic.

>>> geom = [-120.4, 30.0, -110.3, 41.4]

2. Point

This is a list of floats corresponding to: `[longitude,latitude]`. The coordinates should be WGS84 geographic. For point geometries, the geometry is actually buffered by `search_radius_mult` * (data resolution). Hence, output geometries are in fact polygons.

>>> geom = [-120.4,36.5]

3. Using :class:`ocgis.ShpCabinetIterator`

>>> from ocgis import ShpCabinetIterator
>>> geom = ShpCabinetIterator('state_boundaries',select_ugid=[16])

.. _geom key:

4. Using a :class:`ocgis.ShpCabinet` Key

>>> geom = 'state_boundaries'

5. Custom Sequence of Shapely Geometry Dictionaries

The `crs` key is optional. If it is not included, WGS84 is assumed. The `properties` key is also optional. If not 'UGID' property is provided, defaults will be inserted.

>>> geom = [{'geom':Point(x,y),'properties':{'UGID':23,'NAME':'geometry23'},'crs':CoordinateReferenceSystem(epsg=4326)},...]

6. Path to a Shapefile

>>> geom = '/path/to/shapefile.shp'

.. _search_radius_mult key:

search_radius_mult
~~~~~~~~~~~~~~~~~~

This is a scalar float value multiplied by the target data's resolution to determine the buffer radius for the point. The default is ``0.75``.

output_crs
~~~~~~~~~~

By default, the coordinate reference system (CRS) is the CRS of the input :class:`ocgis.RequestDataset` object. If multiple :class:`ocgis.RequestDataset` objects are part of an :class:`ocgis.OcgOperations` call, then `output_crs` must be provided if the input CRS values of the :class:`ocgis.RequestDataset` objects differ. The value for `output_crs` is an instance of :class:`ocgis.crs.CoordinateReferenceSystem`.

aggregate
~~~~~~~~~

=================== ========================================================================================
Value               Description
=================== ========================================================================================
``True``            Selected geometries are combined into a single geometry (see :ref:`appendix-aggregate`).
``False`` (default) Selected geometries are not combined.
=================== ========================================================================================

.. _calc_headline:

calc
~~~~

See the :ref:`computation_headline` page for more details.

.. _calc_grouping_headline:

calc_grouping
~~~~~~~~~~~~~

There are three forms for this argument:

1. **Date Part Grouping**: Any combination of 'day', 'month', and 'year'.

>>> calc_grouping = ['day']
>>> calc_grouping = ['month','year']
>>> calc_grouping = ['day','year']

Temporal aggregation splits date/time coordinates into parts and groups them according to `unique combinations` of those parts. If data is grouped by month, then all of the January times would be in one group with all of the August times in another. If a grouping of month and year are applied, then all of the January 2000 times would be in a group with all of the January 2001 times and so on.

Any temporal aggregation applied to a dataset should be consistent with the input data's temporal resolution. For example, aggregating by day, month, and year on daily input dataset is not a reasonable aggregation as the data selected for aggregation will have a sample size of one (i.e. one day per aggregation group).

2. **Summarize Over All**: The string ``'all'`` indicates the entire time domain should be summarized.

>>> calc_grouping = 'all'

3. **Seasonal Groups**: A sequence of integer sequences. Element sequences must be mutually exclusive (i.e. no repeated integers). Representatative times for the climatology are chosen as the center month in a sequence (i.e. January in the sequence [12,1,2]).

Month integers map as expected (1=January, 2=February, etc.). The example below constructs a single season composed of March, April, and May. Note the nested lists.

>>> calc_grouping = [[3,4,5]]

The next example consumes all the months in a year.

>>> calc_grouping = [[12,1,2],[3,4,5],[6,7,8],[9,10,11]]

It is also possible to group the seasons by year.

>>> calc_grouping = [[12,1,2],[3,4,5],[6,7,8],[9,10,11],'year']

.. _calc_raw_headline:

calc_raw
~~~~~~~~

====================== =====================================================================================================
Value                  Description
====================== =====================================================================================================
`True`                 If :attr:`ocgis.OcgOperations.aggregate` is `True`, perform computations on raw, unaggregated values.
`False` (default)      Use aggregated values during computation.
====================== =====================================================================================================

abstraction
~~~~~~~~~~~

.. note:: OpenClimateGIS uses the `bounds` attribute of NetCDF file to construct polygon representations of datasets. If no `bounds` attribute is found, the software defaults to the `point` geometry abstraction.

====================== =============================================================
Value                  Description
====================== =============================================================
`polygon` (default)    Represent cells as :class:`shapely.geometry.Polygon` objects.
`point`                Represent cells as :class:`shapely.geometry.Point` objects.
====================== =============================================================

.. _snippet_headline:

snippet
~~~~~~~

.. note:: The entire spatial domain is returned unless `geom` is specified.

.. note:: Only applies for pure subsetting for limiting computations use `time_range` and/or `time_region`.

====================== ===========================================================================
Value                  Description
====================== ===========================================================================
`True`                 Return only the first time point and the first level slice (if applicable).
`False` (default)      Return all data.
====================== ===========================================================================

.. _output_format_headline:

output_format
~~~~~~~~~~~~~

====================== ===============================================================================================
Value                  Description
====================== ===============================================================================================
`numpy` (default)      Return a `ocgis.SpatialCollection` with keys matching `ugid` (see `geom`_).
`shp`                  A shapefile representation of the data.
`csv`                  A CSV file representation of the data.
`csv+`                 In addition to a CSV representation, shapefiles with primary key links to the CSV are provided.
`nc`                   A NetCDF4 file.
`geojson`              A GeoJSON representation of the data.
====================== ===============================================================================================

agg_selection
~~~~~~~~~~~~~

================= ===============================================
Value             Description
================= ===============================================
`True`            Aggregate (union) `geom`_ to a single geometry.
`False` (default) Leave `geom`_ as is.
================= ===============================================

The purpose of this data manipulation is to ease the method required to aggregate (union) geometries into arbitrary regions. A simple example would be unioning the U.S. state boundaries of Utah, Nevada, Arizona, and New Mexico into a single polygon representing a "Southwestern Region".

.. _select_ugid:

select_ugid
~~~~~~~~~~~

Select specific geometries from the target shapefile chosen using "`geom`_". The integer sequence selects matching UGID values from the shapefiles. For more information on adding new shapefiles or the requirements of input shapefiles, please see the section titled `Adding Additional Shapefile Data`_.

>>> select_ugid = [1, 2, 3]
>>> select_ugid = [4, 55]
>>> select_ugid = [1]

As clarification, suppose there is a shapefile called "basins.shp" (this assumes the folder containing the shapefile has been set as the value for `env.DIR_SHPCABINET`_) with the following attribute table:

==== =======
UGID Name
==== =======
1    Basin A
2    Basin B
3    Basin C
==== =======

If the goal is to subset the data by the boundary of "Basin A" and write the resulting data to netCDF, a call to OCGIS looks like:

>>> import ocgis
>>> rd = ocgis.RequestDataset(uri='/path/to/data.nc',variable='tas')
>>> path = ocgis.OcgOperations(dataset=rd,geom='basins',select_ugid=[1],output_format='nc').execute()

vector_wrap
~~~~~~~~~~~

.. note:: Only applicable for WGS84 spatial references.

================= ====================================================================================================
Value             Description
================= ====================================================================================================
`True` (default)  For vector geometry outputs (e.g. `shp`,`keyed`) , ensure output longitudinal domain is -180 to 180.
`False`           Maintain the :class:`~ocgis.RequestDataset`'s longitudinal domain.
================= ====================================================================================================

allow_empty
~~~~~~~~~~~

================= ====================================================================================================
Value             Description
================= ====================================================================================================
`True`            Allow the empty set for geometries not geographically coincident with a source geometry.
`False` (default) Raise :class:`~ocgis.exc.EmptyDataNotAllowed` if the empty set is encountered.
================= ====================================================================================================

headers
~~~~~~~

Useful to limit the number of attributes included in an output file.

>>> headers = ['did','time','value']

interpolate_spatial_bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~

If `True`, attempt to interpolate bounds coordinates if they are absent. This will also extrapolate exterior bounds to avoid losing spatial coverage.

:class:`ocgis.ShpCabinet`
=========================

.. autoclass:: ocgis.ShpCabinet
   :members: keys, iter_geoms

.. autoclass:: ocgis.ShpCabinetIterator
   :members: __iter__

Adding Additional Shapefile Data
--------------------------------

.. warning:: Only add data WGS84 geographic data, ESPS=4326.

Shapefiles may be added to the directory mapped by the environment variable :attr:`ocgis.env.DIR_SHPCABINET`. Shapefiles must have a unique integer attribute called 'UGID'. This attribute is required for the "`select_ugid`_" argument to find specific geometries inside the shapefile.

The shapefile's "`geom key`_" is the name of the shapefile. It must have an alphanumeric name with no spaces with the only allowable special character being underscores "_".

:class:`ocgis.Inspect`
=========================

.. autoclass:: ocgis.Inspect
   :members:

Data Collections
================

When the default output format (i.e. `numpy`) is returned by OpenClimateGIS, the data comes back as a dict with keys mapping to the `ugids` of the selection geometries from `geom`_. If `None` is used for `geom`_, then the `ugid` defaults to 1.

>>> from ocgis import OcgOperations, RequestDataset
...
>>> rd = RequestDataset('/path/to/data','tasmax')
>>> ## default return type is NumPy
>>> ops = OcgOperations(rd,snippet=True)
>>> ret = ops.execute()
>>> ## this the dictionary object mapping a Field to its alias
>>> ret[1]
{'tasmax':<Field object>}
>>> ## this is the field object
>>> tasmax_field = ret[1]['tasmax']
>>> # Dimension and data values are accessed via instance attributes.
>>> tasmax.temporal...
>>> tasmax.level...
>>> tasmax.spatial...
>>> # This is the actual data.
>>> tasmax_field.variables['tasmax']

:mod:`ocgis.constants`
======================

.. automodule:: ocgis.constants
   :members:
   :undoc-members:
   :show-inheritance:

:mod:`ocgis.exc`
================

.. automodule:: ocgis.exc
   :members:
   :undoc-members:
   :show-inheritance:

.. _PROJ4 string: http://trac.osgeo.org/proj/wiki/FAQ
.. _shapely documentation: http://toblerity.github.com/shapely/manual.html

    
