.. _esmpy-regridding:

================
ESMPy Regridding
================

If you are unfamiliar with regridding and associated terminology, please consider perusing the website for `ESMPy <https://www.earthsystemcog.org/projects/esmpy/>`_.

Example code is found in the :ref:`regridding` section on the code examples page.

The :class:`~ocgis.regrid.base.RegridOperation` class forms the backbone of OCGIS regridding operations.

++++++++++++++++++++++++
Overview and Limitations
++++++++++++++++++++++++

----------------------------
Supported Regridding Methods
----------------------------

If bounds are present on the source and destination grids (i.e. corners may be constructed), then conservative regridding is performed. Without bounds/corners, a bilinear interpolation method is used. Bilinear is the method used when the regrid option ``’with_corners’`` is ``False``.

The ESMPy documentation provides an overview of regridding methods:
 * http://www.earthsystemmodeling.org/esmf_releases/last_built/esmpy_doc/html/api.html#regridding

------------------
Coordinate Systems
------------------

Currently, only spherical coordinate systems are supported. The default input/output coordinate system in OCGIS is WGS84 geographic. The WGS84 datum was chosen to maintain compatibility with GIS software and data where the WGS84 is used widely.

All input data will be projected to the standard sphere used by PROJ.4 with a semi-major axis of 6,370,997 meters (6371 kilometers). However, it is assumed data _not_ assigned a coordinate system in the file metadata _or_ when initializing a :class:`~ocgis.RequestDataset` has a spherical coordinate system equal to the standard sphere used by PROJ.4. If the data is known to have a WGS84 datum, it may be directly assigned via:

>>> import ocgis
>>> rd = ocgis.RequestDataset(uri=..., crs=ocgis.crs.CFWGS84())

This will cause the software to attempt a coordinate system remap to the standard sphere. Future releases will handle more coordinate systems (i.e. planar).

------------------------------------------
Where does regridding occur in operations?
------------------------------------------

Regridding occurs following a spatial subset and before any calculations.

------------------
Spatial Operations
------------------

Currently, only the ``‘intersects’`` spatial operation is supported. If data needs to be clipped, consider first regridding the data to be clipped using an intersects spatial operation. Then feed this output data into another operation to clip.
