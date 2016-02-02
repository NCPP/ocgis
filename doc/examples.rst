.. _examples-label:

========
Examples
========

These examples will introduce basic subsetting, formatting, and computation in OpenClimateGIS. The datasets used in these examples were downloaded from the Earth System Grid Federation.

.. note:: Shapefile data is currently not distributed with OpenClimateGIS. Please contact the maintainer for information on acquiring shapefile datasets.

Inspection
----------

First, it is always a good idea to ensure a dataset is readable by OpenClimateGIS. Furthermore, it is also good to check if variables/dimensions are properly identifiable and match expectations. The :class:`ocgis.Inspect` object offers a way to perform those initial checks.

.. literalinclude:: sphinx_examples/inspect.py

As an alternative, you may use the :meth:`ocgis.RequestDataset.inspect` method to generate a similar output:

.. literalinclude:: sphinx_examples/inspect_request_dataset.py

Passing a variable to :class:`ocgis.Inspect` or using :meth:`ocgis.RequestDataset.inspect` will prepend variable-level information used by OpenClimateGIS. It is important to look carefully at this descriptive information to identify any inconsistencies especially if the target dataset may not be entirely CF-compliant. For example, this is normal output for variable-level descriptions:

.. code-block:: rest

   <snip>
   URI = /usr/local/climate_data/CanCM4/tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc
   VARIABLE = tas
   
   === Temporal =============
   
          Start Date = 2001-01-01 12:00:00
            End Date = 2010-12-31 12:00:00
            Calendar = 365_day
               Units = days since 1850-1-1
   Resolution (Days) = 1
               Count = 3650
          Has Bounds = True
   
   === Spatial ==============
   
   Spatial Reference = WGS84
        Proj4 String = +proj=longlat +datum=WGS84 +no_defs 
              Extent = (-1.40625, -90.0, 358.59375, 90.0)
      Interface Type = SpatialInterfacePolygon
          Resolution = 2.78932702678
               Count = 8192
   
   === Level ================
   
   No level dimension found.
   
   === Dump =================
   
   dimensions:
       time = ISUNLIMITED ; // 3650 currently
       lat = 64 ;
       lon = 128 ;
       bnds = 2 ;
   
   variables:
       float64 time(time) ;
         time:bounds = "time_bnds" ;
         time:units = "days since 1850-1-1" ;
         time:calendar = "365_day" ;
         time:axis = "T" ;
         time:long_name = "time" ;
   </snip>

Use of `snippet`
----------------

The :ref:`snippet_headline` argument is important when testing and developing an OpenClimateGIS call. It should generally be set to `True` until the final data request is ready. This is for your benefit (requests are faster) as well as for the benefit of any remote storage server (not transferring excessive data).

Simple Subsetting
-----------------

.. warning:: The `keyed` :ref:`output_format_headline` is highly recommended as writing data to shapefiles/CSV may result in large file sizes. For each record, a shapefile repeats the geometry storage.

This example will introduce simple subsetting by a bounding box with conversion to in-memory NumPy arrays, shapefile, CSV, and keyed formats.

.. literalinclude:: sphinx_examples/simple_subset.py

Now, the directory structure for `/tmp/foo` will look like:

.. code-block:: rest

   $ find /tmp/foo
   /tmp/foo/
   /tmp/foo/nc_output
   /tmp/foo/nc_output/nc_output_did.csv
   /tmp/foo/nc_output/nc_output.nc
   /tmp/foo/nc_output/nc_output_meta.txt
   /tmp/foo/csv-shp_output
   /tmp/foo/csv-shp_output/csv-shp_output_meta.txt
   /tmp/foo/csv-shp_output/csv-shp_output_did.csv
   /tmp/foo/csv-shp_output/csv-shp_output.csv
   /tmp/foo/csv-shp_output/shp
   /tmp/foo/csv-shp_output/shp/csv-shp_output_gid.csv
   /tmp/foo/csv-shp_output/shp/csv-shp_output_gid.shp
   /tmp/foo/csv-shp_output/shp/csv-shp_output_ugid.prj
   /tmp/foo/csv-shp_output/shp/csv-shp_output_gid.dbf
   /tmp/foo/csv-shp_output/shp/csv-shp_output_ugid.shp
   /tmp/foo/csv-shp_output/shp/csv-shp_output_gid.prj
   /tmp/foo/csv-shp_output/shp/csv-shp_output_ugid.shx
   /tmp/foo/csv-shp_output/shp/csv-shp_output_gid.shx
   /tmp/foo/csv-shp_output/shp/csv-shp_output_ugid.csv
   /tmp/foo/csv-shp_output/shp/csv-shp_output_ugid.dbf
   /tmp/foo/shp_output
   /tmp/foo/shp_output/shp_output_ugid.shx
   /tmp/foo/shp_output/shp_output_ugid.prj
   /tmp/foo/shp_output/shp_output.shp
   /tmp/foo/shp_output/shp_output_ugid.csv
   /tmp/foo/shp_output/shp_output.dbf
   /tmp/foo/shp_output/shp_output_ugid.shp
   /tmp/foo/shp_output/shp_output.shx
   /tmp/foo/shp_output/shp_output_ugid.dbf
   /tmp/foo/shp_output/shp_output_meta.txt
   /tmp/foo/shp_output/shp_output_did.csv
   /tmp/foo/shp_output/shp_output.prj
   /tmp/foo/csv_output
   /tmp/foo/csv_output/csv_output.csv
   /tmp/foo/csv_output/csv_output_meta.txt
   /tmp/foo/csv_output/csv_output_did.csv

Advanced Subsetting
-------------------

In this example, a U.S. state boundary shapefile will be used to subset and aggregate three climate datasets. The aggregation will occur on a per-geometry + dataset basis. Hence, we will end up with daily aggregated "statewide" temperatures for the three climate variables. We also want to clip the climate data cells to the boundary of the selection geometry to take advantage of area-weighting and avoid data duplication.

.. note:: With no output directory specified, data is written to the current working directory.

.. literalinclude:: sphinx_examples/advanced_subset.py

Subsetting with a Time/Level Range
----------------------------------

Adding a time or level range subset is done at the :class:`~ocgis.RequestDataset` level.

.. warning:: Datetime ranges are absolute and inclusive.

For example, the code below will return all data from the year 2000 for the first two levels. Level indexing originates at 1.

>>> from ocgis import OcgOperations, RequestDataset
>>> import datetime
...
>>> ## Depending on your data's time resolution, the hour/minute/second/etc.
>>> ## may be important for capturing all the data within the range.
>>> tr = [datetime.datetime(2000,1,1),datetime.datetime(2000,12,31,23,59,59)]
>>> rd = RequestDataset('/my/leveled/data','tas',time_range=tr,level_range=[1,2])
>>> ret = OcgOperations(dataset=rd).execute()

.. _regridding:

Regridding
----------

.. literalinclude:: sphinx_examples/regridding.py

.. _2d-flexible-mesh-example-label:

Converting an ESRI Shapefile to UGRID
-------------------------------------

.. literalinclude:: sphinx_examples/to_ugrid_2d_flexible_mesh.py