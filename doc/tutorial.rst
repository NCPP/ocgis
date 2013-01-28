========
Tutorial
========

These tutorials will introduce basic subsetting, formatting, and computation in OpenClimateGIS. The datasets used in this tutorial were downloaded from the Earth System Grid Federation.

.. note:: Shapefile data is currently not distributed with OpenClimateGIS. Please contact the maintainer for information on acquiring shapefile datasets.

.. .. literalinclude:: examples.py

Inspection
----------

First, it is always a good idea to ensure a dataset is readable by OpenClimateGIS. Furthermore, it is also good to check if variables/dimensions are properly identifiable and match expectations. The :class:`ocgis.Inspect` object offers a way to perform those initial checks.

.. literalinclude:: sphinx_examples/inspect.py

The second print statement will prepend variable-level information used by OpenClimateGIS. It is important to look carefully at this information to identify any inconsistencies especially if the target dataset may not be entirely CF-compliant. For example, this is normal output for variable-level descriptions:

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

Simple Subset
-------------

Advanced Subsetting
-------------------

Computation
-----------
