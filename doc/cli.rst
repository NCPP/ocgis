:tocdepth: 4

======================
Command Line Interface
======================

The OpenClimateGIS command line interface provides access to Python capabilities using command line syntax. Supported subcommands are called using:

.. code-block:: sh

   ocli <subcommand> <arguments>

Current subcommands:

=============== ========================== =======================================================================================================================================================================================
Subcommand      Long Name                  Description
=============== ========================== =======================================================================================================================================================================================
``chunked_rwg`` :ref:`chunked_rwg_section` Chunked regrid weight generation using OCGIS spatial decompositions and ESMF weight generation. Allows weight generation for very high resolution grids in memory-limited environments.
=============== ========================== =======================================================================================================================================================================================

.. _chunked_rwg_section:

++++++++++++++++++++++++++++++++
Chunked Regrid Weight Generation
++++++++++++++++++++++++++++++++

Chunked regrid weight generation uses a spatial decomposition to calculate regridding weights by breaking source and destination grids into smaller pieces (chunks). This allows very high resolution grids to participate in regridding without depleting machine memory. The destination grid is chunked using a spatial decomposition (unstructured grids) or index-based slicing (structured, logically rectangular grids). The source grid is then spatially subset by the spatial extent of the destination chunk increased by a spatial buffer to ensure the destination chunk is fully mapped by the source chunk. Weights are calculated using `ESMPy <http://www.earthsystemmodeling.org/esmf_releases/public/last/esmpy_doc/html/index.html>`_, the Python interface for the `Earth System Modeling Framework (ESMF) <https://www.earthsystemcog.org/projects/esmf/>`_, for each chunked source-destination combination. A global weight file merge is performed by default on the weight chunks to create a global weights file.

In addition to chunked weight generation, the interface also offers spatial subsetting of the source grid using the `global` spatial extent of the destination grid. This is useful in situations where the destination grid spatial extent is very small compared to the spatial extent of the source grid.

-----
Usage
-----

.. literalinclude:: sphinx_examples/chunked_rwg_help.sh

--------------------
ESMF Cross-Reference
--------------------

* `Supported File Formats <http://www.earthsystemmodeling.org/esmf_releases/public/last/ESMF_refdoc/node3.html#SECTION03028000000000000000>`_
* `Regrid Methods <http://www.earthsystemmodeling.org/esmf_releases/public/last/ESMF_refdoc/node3.html#SECTION03023000000000000000>`_
* `Weight File Format Description <http://www.earthsystemmodeling.org/esmf_releases/public/last/ESMF_refdoc/node3.html#SECTION03029000000000000000>`_
* `ESMPy Documentation <http://www.earthsystemmodeling.org/esmf_releases/public/last/esmpy_doc/html/index.html>`_

-----------
Limitations
-----------

* Reducing memory overhead leverages IO heavily. Best performance is attained when `netCDF4-python <http://unidata.github.io/netcdf4-python/>`_ is built with parallel support to allow concurrent IO writes with OpenClimateGIS. A warning will be emitted by OpenClimateGIS if a serial only `netCDF4-python <http://unidata.github.io/netcdf4-python/>`_ installation is detected.
* Supports `weight generation only` without weight application (sparse matrix multiplication).
* Works for spherical latitude/longitude grids only.
* When a spatial decomposition is used on the destination grid, there may be duplicate entries in the merged, global weight file. These may be ignored as it results in only minor performance hits for sparse matrix multiplications.

--------
Examples
--------

__________________________________________________
Weight Generation with Logically Rectangular Grids
__________________________________________________

This example creates two global, spherical, latitude/longitude grids with differing spatial resolutions. First, we write the grids to NetCDF files. We then call the command line chunked regrid weight generation in parallel. The destination grid is decomposed into 25 chunks - five chunks along the y-axis and five chunks along the x-axis.

.. literalinclude:: sphinx_examples/chunked_rwg_rect.py

_____________________________________
Weight Generation with Spatial Subset
_____________________________________

This example creates a global, spherical, latitude/longitude grid. It also creates a grid with a single cell. The spatial extent of the single cell grid is much smaller than the global grid. Spatially subsetting the source grid prior to weight generation decreases the amount of source grid information required in the weight calculation.

.. literalinclude:: sphinx_examples/chunked_rwg_ss.py
