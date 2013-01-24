.. _python_api:

==========
Python API
==========

Additional information on keyword arguments can be found below the initial documentation in the `Detailed Argument Information`_ section.

The :class:`OcgOperations` Object
=================================

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

.. _PROJ4 string:: http://trac.osgeo.org/proj/wiki/FAQ
