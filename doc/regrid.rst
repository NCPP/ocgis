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

All spatial data is projected to the standard sphere before regridding.

------------------------------------------
Where does regridding occur in operations?
------------------------------------------

Regridding occurs following a spatial subset and before any calculations.

------------------
Spatial Operations
------------------

Currently, only the ``‘intersects’`` spatial operation is supported. If data needs to be clipped, consider first regridding the data to be clipped using an intersects spatial operation. Then feed this output data into another operation to clip.
