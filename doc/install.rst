============
Installation
============

.. note:: First, make sure `Python 2.7`_ is installed and available on your system path.

Anaconda Package
----------------

An `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ Python package is available through `IOOS's Anaconda Channel <https://anaconda.org/IOOS>`_ repository (linux-64 and osx-64). If you would like to subscribe to the low-volume, OpenClimateGIS mailing list, please fill out the :ref:`download form <download-form>` linked below. Filling out the form also helps us collect useful usage statistics.

Base installation with spatial index and unit conversion support (``nose`` is used for testing purposes only):

.. code-block:: sh

   conda install -c ioos ocgis nose

Installation with all optional dependencies:

.. code-block:: sh

   conda install -c ioos -c nesii/channel/icclim ocgis esmpy icclim nose

Building from Source
--------------------

.. _download-form:

1. Download the current release:

 * http://www.earthsystemmodeling.org/ocgis_releases/public/ocgis-1.2.0/reg/OCGIS_Framework_Reg.html

2. Extract the file using your favorite extraction utility.
3. Navigate into extracted directory.
4. Run the command:

.. code-block:: sh

   [sudo] python setup.py install

Testing the Installation
------------------------

It is recommended that a simple suite of tests are run to verify the new installation. Testing requires the Python ``nose`` library (https://nose.readthedocs.org/en/latest/):

.. code-block:: sh

    [sudo] pip install nose
        # OR
    conda install nose

Run tests:

.. code-block:: sh

    python -c "from ocgis.test import run_simple; run_simple(verbose=False)"

Optional dependencies may also be tested. If an optional dependency is not installed, a test failure will occur:

.. code-block:: sh

    python -c "from ocgis.test import run_simple; run_simple(attrs=['simple', 'optional'], verbose=False)"

Please report any errors to the support email address.

Configuring the :class:`~ocgis.GeomCabinet`
-------------------------------------------

Set the path to the directory containing the shapefiles or shapefile folders in :ref:`env.DIR_GEOMCABINET <env.DIR_GEOMCABINET>`. You may also set the system environment variable ``OCGIS_DIR_GEOMCABINET``.

Dependencies
------------

OpenClimateGIS is tested against the library versions listed below.

Required
~~~~~~~~

============== ======= =======================================================================
Package Name   Version URL
============== ======= =======================================================================
Python         2.7.10  https://www.python.org/downloads/
``osgeo``      1.11.3  https://pypi.python.org/pypi/GDAL/
``setuptools`` 19.6.2  https://pypi.python.org/pypi/setuptools
``shapely``    1.5.13  https://pypi.python.org/pypi/Shapely
``fiona``      1.6.3   https://pypi.python.org/pypi/Fiona
``numpy``      1.10.4  http://sourceforge.net/projects/numpy/files/NumPy/1.9.2/
``netCDF4``    1.2.2   http://unidata.github.io/netcdf4-python/
============== ======= =======================================================================

Optional
--------

Optional dependencies are listed below. OpenClimateGIS will still operate without these libraries installed but functionality and performance may change.

============= ======== ====================================================== =================================================================================================================================
Package Name  Version  URL                                                    Usage
============= ======== ====================================================== =================================================================================================================================
``rtree``     0.8.2    https://pypi.python.org/pypi/Rtree/                    Constructs spatial indexes at runtime. Useful for complicated GIS operations (i.e. large or complex polygons for subsetting)
``cf_units``  1.1      https://github.com/SciTools/cf_units                   Allows unit transformations.
``ESMPy``     7.0.0    https://www.earthsystemcog.org/projects/esmpy/releases Supports regridding operations.
``icclim``    4.1.1    http://icclim.readthedocs.org/en/latest/               Calculation of the full suite of European Climate Assessment (ECA) indices with optimized code implementation.
``nose``      1.3.7    https://nose.readthedocs.org/en/latest/                Run unit tests.
============= ======== ====================================================== =================================================================================================================================

Building from Source
~~~~~~~~~~~~~~~~~~~~

Dependencies may be built entirely from source. A `bash script`_ is available on GitHub.

Platform-Specific Notes
-----------------------

Windows
~~~~~~~

OpenClimateGIS has not been tested on Windows platforms. All libraries are theoretically supported. There are a number of unofficial Windows binaries for Python extensions available here: http://www.lfd.uci.edu/~gohlke/pythonlibs/

Ubuntu Linux
~~~~~~~~~~~~

This method installs all dependencies using hosted packages. This script is available at: https://github.com/NCPP/ocgis/blob/master/doc/sphinx_examples/install_dependencies_ubuntu.sh.

=================== =====================================
Apt-Package         Why?
=================== =====================================
libgdal-dev         ``shapely``, ``osgeo``, and ``fiona``
libgeos-dev         ``shapely`` speedups
libhdf5-dev         ``netCDF4``
libnetcdf-dev       ``netCDF4``
libproj-dev         ``osgeo`` and ``fiona``
libspatialindex-dev ``rtree``
libudunits2-0       ``cfunits``
python-dev          needed at least for ``numpy``
python-pip          all ``pip`` installed Python packages
wget                ``cfunits`` installation
=================== =====================================

Uninstalling
------------

The ``uninstall`` command will simply provide you with the directory location of the OpenClimateGIS package. This must be manually removed.

.. code-block:: sh

    python setup.py uninstall

.. _Python 2.7: http://www.python.org/download/releases/2.7/
.. _bash script: https://github.com/NCPP/ocgis/blob/master/sh/install_geospatial.sh
.. _source: https://github.com/NCPP/ocgis
