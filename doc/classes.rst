.. _class_reference-label:

===============
Class Reference
===============

Dimensions, Variables, and Collections
--------------------------------------

Dimension
=========

.. autoclass:: ocgis.Dimension
    :show-inheritance:
    :members:
    :special-members:

Variable
========

.. autoclass:: ocgis.Variable
    :show-inheritance:
    :members:
    :special-members:

Base Classes
------------

AbstractOcgisObject
===================

.. autoclass:: ocgis.base.AbstractOcgisObject
    :show-inheritance:

AbstractInterfaceObject
=======================

.. autoclass:: ocgis.base.AbstractInterfaceObject
    :show-inheritance:
    :members:

AbstractNamedObject
===================

.. autoclass:: ocgis.base.AbstractNamedObject
    :show-inheritance:
    :members:

Attributes
==========

.. autoclass:: ocgis.variable.attributes.Attributes
    :show-inheritance:
    :members:

GIS File Access
---------------

GeomCabinet
===========

.. autoclass:: ocgis.GeomCabinet
    :members: keys, iter_geoms

GeomCabinetIterator
===================

.. autoclass:: ocgis.GeomCabinetIterator
    :members: __iter__

OcgOperations
=============

.. autoclass:: ocgis.OcgOperations
    :members: execute, get_base_request_size

RegridOperation
===============

.. autoclass:: ocgis.regrid.base.RegridOperation
    :members: execute

RequestDataset
==============

.. autoclass:: ocgis.RequestDataset
    :members: inspect
