============
Installation
============

.. note:: First, make sure `Python 2.7`_ is installed and available on your system path.

Required Dependencies
---------------------

============== ======= =======================================================================
Package Name   Version URL
============== ======= =======================================================================
Python         2.7.6   http://www.python.org/download/releases/2.7.6/
``osgeo``      1.11.1  https://pypi.python.org/pypi/GDAL/
``setuptools`` 12.0.5  https://pypi.python.org/pypi/setuptools
``shapely``    1.4.3   https://pypi.python.org/pypi/Shapely
``fiona``      1.4.5   https://pypi.python.org/pypi/Fiona
``numpy``      1.8.2   http://sourceforge.net/projects/numpy/files/NumPy/1.8.2/
``netCDF4``    1.1.1   http://unidata.github.io/netcdf4-python/
============== ======= =======================================================================

Optional Dependencies
---------------------

Optional dependencies are listed below. OpenClimateGIS will still operate without these libraries installed but functionality and performance may change.

============= ======== ====================================================== =================================================================================================================================
Package Name  Version  URL                                                    Usage
============= ======== ====================================================== =================================================================================================================================
``rtree``     0.8.0    https://pypi.python.org/pypi/Rtree/                    Constructs spatial indexes at runtime. Useful for complicated GIS operations (i.e. large or complex polygons for subsetting)
``cfunits``   0.9.6    https://code.google.com/p/cfunits-python/              Allows unit transformations for ``conform_units_to`` argument to :class:`~ocgis.RequestDataset` or :class:`~ocgis.OcgOperations`.
``ESMPy``     6.3.0rp1 https://www.earthsystemcog.org/projects/esmpy/releases Supports regridding operations.
``icclim``    3.0      http://icclim.readthedocs.org/en/latest/               Calculation of the full suite of European Climate Assessment (ECA) indices with optimized code implementation.
============= ======== ====================================================== =================================================================================================================================

Ubuntu Linux
------------

The recommended install method uses hosted packages. These steps also install optional packages. This script is available at: https://github.com/NCPP/ocgis/blob/master/doc/sphinx_examples/install_dependencies_ubuntu.sh.

.. literalinclude:: sphinx_examples/install_dependencies_ubuntu.sh
   :language: sh

Package Notes
~~~~~~~~~~~~~

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

Building from Source
~~~~~~~~~~~~~~~~~~~~

Dependencies may also be built entirely from source. A `bash script`_ is available containing a command structure for installing most of the OpenClimateGIS dependencies.

Mac OS X Notes
--------------

Download and install GDAL from a pre-packaged DMG installer: http://www.kyngchaos.com/files/software/frameworks/GDAL_Complete-1.9.dmg

Windows Notes
-------------

OpenClimateGIS has not been tested on Windows platforms. All libraries are theoretically supported.

There are a number of unofficial Windows binaries for Python extensions available here: http://www.lfd.uci.edu/~gohlke/pythonlibs/

Installing OpenClimateGIS
-------------------------

1. Download the current release:

 * http://www.earthsystemmodeling.org/ocgis_releases/public/ocgis-1.1.0/reg/OCGIS_Framework_Reg.html

2. Extract the file using your favorite extraction utility.
3. Navigate into extracted directory.
4. Run the command:

.. code-block:: sh

   [sudo] python setup.py install

5. Check that the package may be imported:

>>> import ocgis

or

.. code-block:: sh

   python -c 'import ocgis'

Testing the Installation
~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: There are tests for the optional dependencies. These will fail if the optional dependencies are not installed!

It is recommended that a simple suite of tests are run to verify the new installation:

>>> python setup.py test

Please report any errors to the support email address.

Configuring the :class:`~ocgis.ShpCabinet`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the path to the directory containing the shapefile folders in :attr:`ocgis.env.DIR_SHPCABINET`. You may also set the system environment variable ``OCGIS_DIR_SHPCABINET``.

Uninstalling OpenClimateGIS
---------------------------

The ``uninstall`` command will simply provide you with the directory location of the OpenClimateGIS package. This must be manually removed.

.. code-block:: sh

    python setup.py uninstall

.. _Python 2.7: http://www.python.org/download/releases/2.7/
.. _bash script: https://github.com/NCPP/ocgis/blob/master/sh/install_geospatial.sh
.. _source: https://github.com/NCPP/ocgis
