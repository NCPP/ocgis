.. _class_reference-label:

===============
Class Reference
===============

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

Grid
++++

.. autoclass:: ocgis.Grid
    :show-inheritance:
    :members:
    :special-members:

GeometryVariable
++++++++++++++++

.. autoclass:: ocgis.GeometryVariable
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

DriverNetcdf
++++++++++++

.. autoclass:: ocgis.driver.nc.DriverNetcdf
    :show-inheritance:

DriverNetcdfCF
++++++++++++++

.. autoclass:: ocgis.driver.nc.DriverNetcdfCF
    :show-inheritance:

DriverVector
++++++++++++

.. autoclass:: ocgis.driver.vector.DriverVector
    :show-inheritance:

DriverCSV
+++++++++

.. autoclass:: ocgis.driver.csv_.DriverCSV
    :show-inheritance:

Base Classes
------------

AbstractOcgisObject
+++++++++++++++++++

.. autoclass:: ocgis.base.AbstractOcgisObject
    :show-inheritance:

AbstractInterfaceObject
+++++++++++++++++++++++

.. autoclass:: ocgis.base.AbstractInterfaceObject
    :show-inheritance:
    :members:

AbstractNamedObject
+++++++++++++++++++

.. autoclass:: ocgis.base.AbstractNamedObject
    :show-inheritance:
    :members:

AbstractContainer
+++++++++++++++++

.. autoclass:: ocgis.variable.base.AbstractContainer
    :show-inheritance:
    :members:

Attributes
++++++++++

.. autoclass:: ocgis.variable.attributes.Attributes
    :show-inheritance:
    :members:

AbstractCRS
+++++++++++

.. autoclass:: ocgis.variable.crs.AbstractCRS
    :show-inheritance:
    :members:

.. autoclass:: ocgis.variable.crs.AbstractProj4CRS
    :show-inheritance:
    :members:

AbstractDriver
++++++++++++++

.. autoclass:: ocgis.driver.base.AbstractDriver
    :show-inheritance:

AbstractTabularDriver
+++++++++++++++++++++

.. autoclass:: ocgis.driver.base.AbstractTabularDriver
    :show-inheritance:
