.. _python_api:

==========
Python API
==========

Additional information on keyword arguments can be found below the initial documentation in the `Detailed Argument Information`_ section.

:mod:`ocgis.env`
================

These are global parameters used by OpenClimateGIS. For those familiar with :mod:`arcpy` programming, this behaves similarly to the :mod:`arcpy.env` module.

.. automodule:: ocgis.env
   :members:

:class:`ocgis.OcgOperations`
============================

.. autoclass:: ocgis.OcgOperations
   :members: execute, as_dict, as_qs

Detailed Argument Information
-----------------------------

Additional information on arguments are found in their respective sections.

dataset
~~~~~~~

A `dataset` is the target file(s) where data is stored. A `dataset` may be on the local machine or network location accessible by the software. Unsecured OpenDAP datasets may also be accessed.

.. autoclass:: ocgis.RequestDataset

.. autoclass:: ocgis.RequestDatasetCollection
   :members: update

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

>>> bounds = [-120.4, 30.0, -110.3, 41.4]

2. Using :class:`ocgis.ShpCabinet`

>>> from ocgis import ShpCabinet
>>> sc = ShpCabinet()
>>> geoms = sc.get_geoms('state_boundaries')

3. Using a :class:`ocgis.ShpCabinet` Key

>>> geom_key = 'state_boundaries'

4. Direct Geometry Construction

A geometry in OpenClimateGIS is a dictionary containing `ugid` (user geometry identifier) and `geom` keys. The `ugid` is a unique integer identifier -- unique within the list of geometries -- with the `geom` key value being a :class:`shapely.geometry.Polygon` or :class:`shapely.geometry.MultiPolygon` object. See `shapely documentation`_.

.. note:: Remember to always keep coordinates in WGS84 geographic. The software will handle projecting to matching coordinate systems.

>>> from shapely.geometry import Polygon
>>> from ocgis.util.helpers import make_poly
...
>>> geom1 = {'ugid':1, 'geom':make_poly((30,40),(-110,-120))}
>>> geom2 = {'ugid':2, 'geom':make_poly((40,50),(-110,-120))}
>>> geoms = [geom1, geom2]

aggregate
~~~~~~~~~

====================== ========================================================
Value                  Description
====================== ========================================================
`True`                 Selected geometries are combined into a single geometry.
`False` (default)      Selected geometries are not combined.
====================== ========================================================

calc
~~~~

tdk: a lot of work to be done here

calc_grouping
~~~~~~~~~~~~~

Any combination of :class:`datetime.datetime` attribute strings.

>>> calc_grouping = ['year']
>>> calc_grouping = ['month','year']
>>> calc_grouping = ['day','second']

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

====================== ========================================================================================
Value                  Description
====================== ========================================================================================
`True`                 Return only the first time point / time group and the first level slice (if applicable).
`False` (default)      Return all data.
====================== ========================================================================================

.. _output_format_headline:

output_format
~~~~~~~~~~~~~

====================== =========================================================================================================================================================
Value                  Description
====================== =========================================================================================================================================================
`numpy` (default)      Return a dict with keys matching `ugid` (see `geom`_) and values of :class:`ocgis.OcgCollection`.
`keyed`                A reduced data format composed of a shapfile geometry index and a series of CSV files for attributes. Best to think of this as a series of linked tables.
`shp`                  A shapefile representation of the data.
`csv`                  A CSV file representation of the data.
`nc`                   A NetCDF4 file.
====================== =========================================================================================================================================================

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

:class:`ocgis.ShpCabinet`
=========================

.. autoclass:: ocgis.ShpCabinet
   :members: keys, get_geoms, write

Adding Additional Shapefile Data
--------------------------------

In the :attr:`~ocgis.env.SHP_DIR`, create a folder with the name you would like to use for the geometry's key. Copy all the shapefile component files to the new directory. Inside the directory, create a `.cfg` file with the same name as the containing folder. The `.cfg` file will containg the header `[mapping]` and two key-value pairs: `ugid` and `attributes`. The `ugid` key value is the name of the attribute for the shapefiles unique identifier. If the file does not contain a unique identifier, setting the value to `none` will cause OpenClimateGIS to generate unique identifiers. The `attributes` key value are the attributes you want OpenClimateGIS to read from the file. Setting this to `none` will result in no additional attributes being inserted into the geometry dictionary.

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
>>> ops = OcgOperations(rd,snippet=True)
>>> ret = ops.execute()
>>> coll = ret[1]
>>> tasmax = coll.variables['tasmax']
>>> # Dimension and data values are accessed via instance attributes.
>>> tasmax.temporal.value
>>> tasmax.level.value
>>> tasmax.geom.value
>>> # This is the actual data.
>>> tasmax.value 

.. autoclass:: ocgis.OcgCollection
   :members:

   .. attribute:: variables

      An :class:`collections.OrderedDict` holding :class:`ocgis.OcgVariable` s.

.. autoclass:: ocgis.OcgVariable
   :members:
   
   .. attribute:: name

   .. attribute:: value

   .. attribute:: temporal

   .. attribute:: level

   .. attribute:: geom



:mod:`ocgis.exc`
================

.. automodule:: ocgis.exc
   :members:

.. _PROJ4 string: http://trac.osgeo.org/proj/wiki/FAQ
.. _shapely documentation: http://toblerity.github.com/shapely/manual.html
