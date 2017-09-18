.. _class_reference-label:

===============
Class Reference
===============

.. _request-dataset:

RequestDataset
--------------

.. autoclass:: ocgis.RequestDataset
    :members: inspect

OcgOperations
-------------

.. autoclass:: ocgis.OcgOperations
    :members: execute, get_base_request_size

Data Interface
--------------

Dimension
+++++++++

.. autoclass:: ocgis.Dimension
    :show-inheritance:
    :members:
    :special-members:

Variable
++++++++

.. autoclass:: ocgis.Variable
    :show-inheritance:
    :members:
    :special-members:

.. autoclass:: ocgis.SourcedVariable
    :show-inheritance:
    :members:
    :special-members:

CoordinateReferenceSystem
+++++++++++++++++++++++++

.. autoclass:: ocgis.variable.crs.CoordinateReferenceSystem
    :show-inheritance:
    :members:
    :special-members:

CRS
++++

.. autoclass:: ocgis.variable.crs.CRS

VariableCollection
++++++++++++++++++

.. autoclass:: ocgis.VariableCollection
    :show-inheritance:
    :members:
    :special-members:

SpatialCollection
+++++++++++++++++

.. autoclass:: ocgis.SpatialCollection
    :show-inheritance:
    :members:
    :special-members:

DimensionMap
++++++++++++

.. autoclass:: ocgis.DimensionMap
    :show-inheritance:
    :members:
    :special-members:

Field
+++++

.. autoclass:: ocgis.Field
    :show-inheritance:
    :members:
    :special-members:

Grids
+++++

.. autoclass:: ocgis.Grid
    :show-inheritance:
    :members:
    :special-members:

.. autoclass:: ocgis.GridUnstruct
    :show-inheritance:
    :members:
    :special-members:

GeometryVariable
++++++++++++++++

.. autoclass:: ocgis.GeometryVariable
    :show-inheritance:
    :members:
    :special-members:

Geometry Coordinate Variables
+++++++++++++++++++++++++++++

.. autoclass:: ocgis.PolygonGC
    :show-inheritance:
    :members:
    :special-members:

.. autoclass:: ocgis.LineGC
    :show-inheritance:
    :members:
    :special-members:

.. autoclass:: ocgis.PointGC
    :show-inheritance:
    :members:
    :special-members:

TemporalVariable
++++++++++++++++

.. autoclass:: ocgis.TemporalVariable
    :show-inheritance:
    :members:
    :special-members:

.. autoclass:: ocgis.variable.temporal.TemporalGroupVariable
    :show-inheritance:
    :members:
    :special-members:

Parallelism
-----------

OcgVM
+++++

.. autoclass:: ocgis.OcgVM
    :show-inheritance:

OcgDist
+++++++

.. autoclass:: ocgis.vmachine.mpi.OcgDist
    :show-inheritance:

GIS File Access
---------------

GeomCabinet
+++++++++++

.. autoclass:: ocgis.GeomCabinet
    :show-inheritance:
    :members: keys, iter_geoms

GeomCabinetIterator
+++++++++++++++++++

.. autoclass:: ocgis.GeomCabinetIterator
    :show-inheritance:
    :members: __iter__

Operation Wrappers
------------------

CalculationEngine
+++++++++++++++++

.. autoclass:: ocgis.calc.engine.CalculationEngine
    :show-inheritance:

OperationsEngine
++++++++++++++++

.. autoclass:: ocgis.ops.engine.OperationsEngine
    :show-inheritance:

RegridOperation
+++++++++++++++

.. autoclass:: ocgis.regrid.base.RegridOperation
    :show-inheritance:
    :members: execute

SpatialSubsetOperation
++++++++++++++++++++++

.. autoclass:: ocgis.spatial.spatial_subset.SpatialSubsetOperation
    :show-inheritance:
    :members: get_spatial_subset

Drivers
-------

.. autoclass:: ocgis.driver.nc.DriverNetcdf
    :show-inheritance:

.. autoclass:: ocgis.driver.nc.DriverNetcdfCF
    :show-inheritance:

.. autoclass:: ocgis.driver.nc_ugrid.DriverNetcdfUGRID
    :show-inheritance:

.. autoclass:: ocgis.driver.vector.DriverVector
    :show-inheritance:

.. autoclass:: ocgis.driver.csv_.DriverCSV
    :show-inheritance:

Grid Splitter
-------------

.. autoclass:: ocgis.spatial.grid_splitter.GridSplitter
    :members:

Base Classes
------------

.. autoclass:: ocgis.base.AbstractOcgisObject
    :show-inheritance:

.. autoclass:: ocgis.base.AbstractInterfaceObject
    :show-inheritance:
    :members:

.. autoclass:: ocgis.base.AbstractNamedObject
    :show-inheritance:
    :members:

.. autoclass:: ocgis.variable.base.AbstractContainer
    :show-inheritance:
    :members:

.. autoclass:: ocgis.spatial.grid.AbstractGrid
    :show-inheritance:
    :members:

.. autoclass:: ocgis.variable.attributes.Attributes
    :show-inheritance:
    :members:

Spatial Objects
+++++++++++++++

.. autoclass:: ocgis.spatial.base.AbstractSpatialContainer
    :show-inheritance:
    :members:

.. autoclass:: ocgis.spatial.base.AbstractXYZSpatialContainer
    :show-inheritance:
    :members:

.. autoclass:: ocgis.spatial.geomc.AbstractGeometryCoordinates
    :show-inheritance:
    :members:

Coordinate Systems
++++++++++++++++++

.. autoclass:: ocgis.variable.crs.AbstractCRS
    :show-inheritance:
    :members:

.. autoclass:: ocgis.variable.crs.AbstractProj4CRS
    :show-inheritance:
    :members:

Abstract Drivers
++++++++++++++++

.. autoclass:: ocgis.driver.base.AbstractDriver
    :show-inheritance:

.. autoclass:: ocgis.driver.nc.AbstractDriverNetcdfCF
    :show-inheritance:

.. autoclass:: ocgis.driver.base.AbstractTabularDriver
    :show-inheritance:
