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

====================== ===================================================================================================================
Value                  Description
====================== ===================================================================================================================
`intersects` (default) Source geometries touching or overlapping selection geometries are returned.
`clip`                 A full geometric intersection is performed between source and selection geometries. New geometries may be created.
====================== ===================================================================================================================

geom
~~~~

If a geometry(s) is provided, it is used to subset `every` :class:`ocgis.RequestDataset` object. Again, supplying a value of `None` (the default) results in the return of the entire spatial domain. There are four ways to parameterize the `geom` keyword argument:

1. Bounding Box

This is a list of floats corresponding to: `[min x, min y, max x, max y]`. The coordinates should be WGS84 geographic.

>>> geom = [-120.4, 30.0, -110.3, 41.4]

2. Point

This is a list of floats corresponding to: `[longitude,latitude]`. The coordinates should be WGS84 geographic. For point geometries, the geometry is actually buffered by `search_radius_mult` * (data resolution). Hence, output geometries are in fact polygons.

>>> geom = [-120.4,36.5]

3. Using :class:`ocgis.ShpCabinetIterator`

>>> from ocgis import ShpCabinetIterator
>>> geom = ShpCabinetIterator('state_boundaries',select_ugid=[16])

4. Using a :class:`ocgis.ShpCabinet` Key

>>> geom = 'state_boundaries'

5. Custom Sequence of Shapely Geometry Dictionaries

The `crs` key is optional. If it is not included, WGS84 is assumed. The `properties` key is also optional. If not 'UGID' property is provided, defaults will be inserted.

>>> geom = [{'geom':Point(x,y),'properties':{'UGID':23,'NAME':'geometry23'},'crs':CoordinateReferenceSystem(epsg=4326)},...]

search_radius_mult
~~~~~~~~~~~~~~~~~~

This is a scalar float value multiplied by the target data's resolution to determine the buffer radius for the point. 

output_crs
~~~~~~~~~~

By default, the coordinate reference system (CRS) is the CRS of the input :class:`ocgis.RequestDataset` object. If multiple :class:`ocgis.RequestDataset` objects are part of an :class:`ocgis.OcgOperations` call, then `output_crs` must be provided if the input CRS values of the :class:`ocgis.RequestDataset` objects differ. The value for `output_crs` is an instance of :class:`ocgis.crs.CoordinateReferenceSystem`.

aggregate
~~~~~~~~~

====================== ========================================================
Value                  Description
====================== ========================================================
`True`                 Selected geometries are combined into a single geometry.
`False` (default)      Selected geometries are not combined.
====================== ========================================================

.. _calc_headline:

calc
~~~~

See the :ref:`computation_headline` page for more details.

.. _calc_grouping_headline:

calc_grouping
~~~~~~~~~~~~~

Any combination of 'day', 'month', and 'year'.

>>> calc_grouping = ['day']
>>> calc_grouping = ['month','year']
>>> calc_grouping = ['day','year']

Any temporal aggregation applied to a dataset should be consistent with the input data's temporal resolution. For example, aggregating by day, month, and year on daily input dataset is not a reasonable aggregation as the data selected for aggregation will have a sample size of one.

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

====================== ====================================================================================================================================================================
Value                  Description
====================== ====================================================================================================================================================================
`numpy` (default)      Return a dict with keys matching `ugid` (see `geom`_) and values of :class:`ocgis.api.collection.AbstractCollection`. The collection type depends on the operations.
`shp`                  A shapefile representation of the data.
`csv`                  A CSV file representation of the data.
`csv+`                 In addition to a CSV representation, shapefiles with primary key links to the CSV are provided.
`nc`                   A NetCDF4 file.
`geojson`              A GeoJSON representation of the data.
====================== ====================================================================================================================================================================

agg_selection
~~~~~~~~~~~~~

================= ==========================================
Value             Description
================= ==========================================
`True`            Aggregate `geom`_ to a single geometry.
`False` (default) Leave `geom`_ as is.
================= ==========================================

select_ugid
~~~~~~~~~~~

Select specific geometries from `geom`.

>>> select_ugid = [1, 2, 3]
>>> select_ugid = [4, 55]
>>> select_ugid = [1]

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

:class:`ocgis.ShpCabinet`
=========================

.. autoclass:: ocgis.ShpCabinet
   :members: keys, iter_geoms

.. autoclass:: ocgis.ShpCabinetIterator
   :members: __iter__

Adding Additional Shapefile Data
--------------------------------

.. warning:: Only add data WGS84 geographic data, ESPS=4326.

In the directory specified by :attr:`env.DIR_SHPCABINET`, create a folder with the name you would like to use for the geometry's key. Copy all the shapefile component files to the new directory. Inside the directory, create a `.cfg` file with the same name as the containing folder. The `.cfg` file will containing the header `[mapping]` and two key-value pairs: `ugid` and `attributes`. The `ugid` key value is the name of the attribute for the shapefiles unique identifier. If the file does not contain a unique identifier, setting the value to `none` will cause OpenClimateGIS to generate unique identifiers. The `attributes` key value are the attributes you want OpenClimateGIS to read from the file. Setting this to `none` will result in no additional attributes being inserted into the geometry dictionary. Setting the `attributes` value to `all` results in all attributes being read.

.. code-block:: ini

   [mapping]
   ugid=id
   attributes=state_name,population

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

    
