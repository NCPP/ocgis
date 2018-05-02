.. _examples-label:

========
Examples
========

These examples will introduce basic subsetting, formatting, and computation in OpenClimateGIS.

.. _inspection:

Inspection
----------

First, it is always a good idea to ensure a dataset is readable by OpenClimateGIS. Furthermore, it is also good to check if variables/dimensions are properly identifiable and match expectations.

.. literalinclude:: sphinx_examples/inspect_request_dataset.py

Use of ``snippet``
------------------

The :ref:`snippet_headline` argument is important when testing and developing an OpenClimateGIS call. It should generally be set to ``True`` until the final data request is ready. This is for your benefit (requests are faster) as well as for the benefit of any remote storage server (not transferring excessive data).

Simple Subsetting
-----------------

.. warning:: The ``'csv-shp'`` :ref:`output_format_headline` is highly recommended as writing data to pure ESRI Shapefiles / CSV may result in large file sizes. For each record, an ESRI Shapefile repeats the geometry storage.

This example will introduce simple subsetting by a bounding box with conversion to in-memory NumPy arrays, shapefile, CSV, and keyed formats.

.. literalinclude:: sphinx_examples/simple_subset.py

Now, the directory structure for the temporary directory will look like:

.. code-block:: rest

   $ tree /tmp/tmpO900zS
   /tmp/tmpO900zS
   ├── csv_output
   │   ├── csv_output.csv
   │   ├── csv_output_did.csv
   │   ├── csv_output_metadata.txt
   │   └── csv_output_source_metadata.txt
   ├── csv-shp_output
   │   ├── csv-shp_output.csv
   │   ├── csv-shp_output_did.csv
   │   ├── csv-shp_output_metadata.txt
   │   ├── csv-shp_output_source_metadata.txt
   │   └── shp
   │       ├── csv-shp_output_gid.cpg
   │       ├── csv-shp_output_gid.dbf
   │       ├── csv-shp_output_gid.prj
   │       ├── csv-shp_output_gid.shp
   │       ├── csv-shp_output_gid.shx
   │       ├── csv-shp_output_ugid.cpg
   │       ├── csv-shp_output_ugid.dbf
   │       ├── csv-shp_output_ugid.prj
   │       ├── csv-shp_output_ugid.shp
   │       └── csv-shp_output_ugid.shx
   ├── nc_output
   │   ├── nc_output_did.csv
   │   ├── nc_output_metadata.txt
   │   ├── nc_output.nc
   │   └── nc_output_source_metadata.txt
   ├── ocgis_example_simple_subset.nc
   └── shp_output
       ├── shp_output.cpg
       ├── shp_output.dbf
       ├── shp_output_did.csv
       ├── shp_output_metadata.txt
       ├── shp_output.prj
       ├── shp_output.shp
       ├── shp_output.shx
       ├── shp_output_source_metadata.txt
       ├── shp_output_ugid.cpg
       ├── shp_output_ugid.dbf
       ├── shp_output_ugid.prj
       ├── shp_output_ugid.shp
       └── shp_output_ugid.shx

   5 directories, 36 files

.. _advanced-subsetting-example:

Advanced Subsetting
-------------------

In this example, a U.S. state boundary shapefile will be used to subset and aggregate three example climate datasets. The aggregation will occur on a per-geometry + dataset basis. Hence, we will end up with daily aggregated "statewide" temperatures for the three climate variables. We also want to clip the climate data cells to the boundary of the selection geometry to take advantage of area-weighting and avoid repeated grid cells.

.. literalinclude:: sphinx_examples/advanced_subset.py

Subsetting with a Time/Level Range
----------------------------------

Adding a time or level range subset is done at the :class:`~ocgis.RequestDataset` level.

.. warning:: Datetime ranges are absolute and inclusive.

For example, the code below will return all data from the year 2000 for the first two levels. Level indexing originates at 1.

>>> from ocgis import OcgOperations, RequestDataset
>>> import datetime
>>> # Depending on your data's time resolution, the hour/minute/second/etc. may be important for capturing all the data
>>> # within the range.
>>> tr = [datetime.datetime(2000, 1, 1),datetime.datetime(2000, 12, 31, 23, 59, 59)]
>>> rd = RequestDataset('/my/leveled/data', 'tas', time_range=tr, level_range=[1, 2])
>>> ret = OcgOperations(dataset=rd).execute()

.. _configuring-a-dimension-map:

Configuring a Dimension Map
---------------------------

Dimension maps (:class:`~ocgis.DimensionMap`) are used to overload default metadata interpretations. If provided to a request (:class:`~ocgis.RequestDataset`) or field (:class:`~ocgis.Field`), it will be used to intepret a variable collection when converting the collection to a field.

This example shows how to pass a dimension map to a request when working with non-standard metadata:

.. literalinclude:: sphinx_examples/dimension_map.py

Using the Data Interface in Parallel
------------------------------------

Standard operations scripts may be run in parallel using ``mpirun`` with no special code. See :ref:`parallel-example` for an advanced example using OpenClimateGIS's data interface in parallel.

.. _regridding:

Regridding
----------

.. literalinclude:: sphinx_examples/regridding.py

Calculating TG90p using ``icclim``
----------------------------------

*TG90p* is a European Climate Assessment (ECA) climate indice.

.. literalinclude:: sphinx_examples/tg90p_with_icclim.py

Stack Subsetted Data From Spatial Collections
---------------------------------------------

Stack (concatenate) subsetted data from multiple files using the unlimited time dimension.

.. literalinclude:: sphinx_examples/stacking_subsets.py
