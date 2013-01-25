.. _python_api:

==========
Python API
==========

Additional information on keyword arguments can be found below the initial documentation in the `Detailed Argument Information`_ section.

:class:`ocgis.OcgOperations`
======================

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

If a geometry(s) is provided, it is used to subset `every` :class:`ocgis.RequestDataset` object. Again, supplying a value of `None` (the default) results in the return of the entire spatial domain. There are three ways to parameterize the `geom` keyword argument:

1. Bounding Box

This is a list of floats corresponding to: `[min x, min y, max x, max y]`. The coordinates should be WGS84 geographic.

>>> bounds = [-120.4, 30.0, -110.3, 41.4]

2. Using :class:`ocgis.ShpCabinet`

See :class:`ocgis.ShpCabinet' for more information.

>>> from ocgis import ShpCabinet
>>> sc = ShpCabinet()
>>> geoms = sc.get_geoms('state_boundaries')

3. Direct Geometry Construction

A geometry in OpenClimateGIS is a dictionary containing `ugid` (user geometry identifier) and `geom` keys. The `ugid` is a unique integer identifier -- unique within the list of geometries -- with the `geom` key value being a :class:`shapely.geometry.Polygon` or :class:`shapely.geometry.MultiPolygon` object. See `shapely documentation`_.

.. note:: Remember to always keep coordinates in WGS84 geographic. The software will handle projecting to matching coordinate systems.

>>> from shapely.geometry import Polygon
>>> from ocgis.util.helpers import make_poly
...
>>> geom1 = {'ugid':1, 'geom':make_poly((30,40),(-110,-120))}
>>> geom2 = {'ugid':2, 'geom':make_poly((40,50),(-110,-120))}
>>> geoms = [geom1, geom2]

.. _shpcabinet:

:class:`ocgis.ShpCabinet`
===================

.. autoclass:: ocgis.ShpCabinet


.. _PROJ4 string: http://trac.osgeo.org/proj/wiki/FAQ
.. _shapely documentation: http://toblerity.github.com/shapely/manual.html
