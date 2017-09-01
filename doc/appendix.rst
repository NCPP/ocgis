Appendix
--------

Output Formats
~~~~~~~~~~~~~~

There are multiple formats in which OCGIS output can be saved on disk: shapefiles (``'shp'``), comma separated values (``'csv'``), a combination of shapefiles and csv (``'csv-shp'``), netCDF (``'nc'``) and GeoJSON (``'geojson'``). The :class:`~ocgis.OcgOperations` :ref:`output_format_headline` parameter determines which format is used. Additional options are available through the :ref:`output_format_options_headline` (not all output formats expose options).

.. _netcdf_output_headline:

NetCDF Output
+++++++++++++

`netCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ is a binary file format widely used in the climate community. It is self-descriptive in the sense that it carries metadata describing the file content. netCDF files created by OCGIS carry the original file's metadata in addition to updating the file's history attribute. Spatial and temporal subsetting or aggregation operations will modify the original dimensions of the file but retain all relevant data and metadata on the temporal and spatial coordinates, time and level dimensions. New variable created by OCGIS are named after the ``calc`` operation that defined it.

There are two modes for netCDF output, one handling gridded data and another handling multiple unioned geometries. For example, if ``geom`` is a list of world countries, ``calc`` is the annual average and ``dataset`` stores globally gridded daily temperature and ``aggregate`` is set to ``True``, then the annual average temperature for each country would be stored in a variable with dimensions ``(time, ocgis_geom_union)``, where the ``ocgis_geom_union`` dimension indexes each individual country. Metadata about the geometries (for example its name), will be stored in variables spanning this spatial dimension. Although the netCDF file does not include each region's geometry (point or polygon), the region's centroid coordinates are stored in the latitude and longitude variables. Instead of ``ocgis_geom_union``, users can change the name of the discrete geometry dimension by setting the ``output_format_options`` called ``geom_dim``.

Spatial Operations
~~~~~~~~~~~~~~~~~~

.. note:: Differences between point and polygon geometry representations are discussed in the respective sections. The greatest differences between how the geometries are handled occurs in the way point aggregations are constructued (see `Aggregate (Union)`_).

Spatial Masking
+++++++++++++++

OpenClimateGIS manages spatial masking independent from data masking. Following a spatial operation, a new variable is added to a grid's variable collection if a any geometries are masked. This variable's default name is :attr:`constants.VariableName.SPATIAL_MASK`.

The spatial mask variable allows masking to be tracked independently from data variable masking. The spatial mask is stilled hardened into the data array following a subset. A spatial mask variable will not be created if no geometries within the subset were masked.

If present on an input dataset, the spatial mask variable will always be respected. It may be ignored by setting it to ``None`` in a request dataset's :class:`~ocgis.DimensionMap`.

.. _appendix-intersects:

Intersects (Select)
+++++++++++++++++++

Returns ``True`` if the boundary of two spatial objects overlap. This differs from the classical set-theoretic definition of `intersects`_ for spatial analysis. In OpenClimateGIS, geometric objects that `touch`_ only are excluded.

.. figure:: images/intersects.png
   :scale: 40%
   :align: center
   
   The `intersects` operation returns only overlapping geometries.

.. _appendix-clip:

Clip (Intersection)
+++++++++++++++++++

A clip operation is synonymous with an `intersection`_. If the source data may only be represented as points, then an intersects operation is executed (see `Intersects (Select)`_).

.. figure:: images/clip.png
   :scale: 40%
   :align: center
   
   The `clip` operation creates new geometries by cutting features given the boundary of another feature.

.. _appendix-aggregate:

Aggregate (Union)
+++++++++++++++++

Aggregation or union is the merging of feature geometries to create a new geometry. For polygons, this will result in a single features. For points, the result is a multi-point collection.

.. figure:: images/aggregate.png
   :scale: 40%
   :align: center
   
   Multiple polygons are combined into a single polygon during spatial aggregation.

Point Selection Geometries
~~~~~~~~~~~~~~~~~~~~~~~~~~

When selecting data with a point, the point is automatically buffered according to :ref:`search_radius_mult key`. This may result in multiple geometries or cells being returned by a request.

.. _intersects: http://toblerity.org/shapely/manual.html#object.intersects
.. _touches: http://toblerity.org/shapely/manual.html#object.touches
.. _intersect: http://toblerity.org/shapely/manual.html#object.intersects
.. _touch: http://toblerity.org/shapely/manual.html#object.touches
.. _intersection: http://toblerity.org/shapely/manual.html#object.intersection
