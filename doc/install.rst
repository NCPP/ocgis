============
Installation
============

.. note:: First, make sure `Python 2.7`_ is installed and available on your system path.

Required Dependencies
---------------------

============== ======= =======================================================================
Package Name   Version URL
============== ======= =======================================================================
Python         2.7.2   http://www.python.org/download/releases/2.7.2/
``osgeo``      1.9.1   http://pypi.python.org/pypi/GDAL/
``shapely``    1.2     http://pypi.python.org/pypi/Shapely
``fiona``      1.0.2   https://pypi.python.org/pypi/Fiona
``numpy``      1.6.2   http://sourceforge.net/projects/numpy/files/NumPy/1.6.2/
``netCDF4``    1.2     http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html
============== ======= =======================================================================

Optional Dependencies
---------------------

There are two optional dependencies. OpenClimateGIS will still operate without these libraries installed but functionality and performance may change.

============== ======= ========================================= =================================================================================================================================
Package Name   Version URL                                       Usage
============== ======= ========================================= =================================================================================================================================
``rtree``      2.7.2   https://pypi.python.org/pypi/Rtree/       Constructs spatial indexes at runtime. Useful for complicated GIS operations (i.e. large or complex polygons for subsetting)
``cfunits``    0.9.6   https://code.google.com/p/cfunits-python/ Allows unit transformations for ``conform_units_to`` argument to :class:`~ocgis.RequestDataset` or :class:`~ocgis.OcgOperations`.
============== ======= ========================================= =================================================================================================================================

Ubuntu Linux
------------

The recommended install method uses hosted packages. These steps also install optional packages.

.. code-block:: sh

   [sudo] apt-get update
   [sudo] apt-get install wget libnetcdf-dev libgeos-dev libgdal-dev libspatialindex-dev libudunits2-0 python-pip python-dev
   [sudo] pip install numpy
   [sudo] pip install netCDF4
   [sudo] pip install shapely
   [sudo] pip install fiona
   [sudo] pip install rtree

   ###########
   ## osgeo ##
   ###########

   ## http://stackoverflow.com/questions/11336153/python-gdal-package-missing-header-file-when-installing-via-pip
   pip install --no-install GDAL
   cd /tmp/pip_build_ubuntu/GDAL
   python setup.py build_ext --include-dirs=/usr/include/gdal
   [sudo] python setup.py install

   #############
   ## cfunits ##
   #############

   SRCDIR=/tmp/build_cfunits
   CFUNITS_VER=0.9.6
   CFUNITS_SRC=$SRCDIR/cfunits-python/v$CFUNITS_VER
   CFUNITS_TARBALL=cfunits-$CFUNITS_VER.tar.gz
   CFUNITS_URL=https://cfunits-python.googlecode.com/files/$CFUNITS_TARBALL

   mkdir -p $CFUNITS_SRC
   cd $CFUNITS_SRC
   wget $CFUNITS_URL
   tar -xzvf $CFUNITS_TARBALL
   cd cfunits-$CFUNITS_VER
   [sudo] python setup.py install
   ## installation does not copy UDUNITS database
   CFUNITS_SETUP_DIR=`pwd`
   cd
   python -c 'import cfunits,os,subprocess;cfunits_path=os.path.split(cfunits.__file__)[0];subprocess.call(["export","CFUNITS_INSTALL_DIR={0}".format(cfunits_path)],shell=True)'
   [sudo] cp -r $CFUNITS_SETUP_DIR/cfunits/etc $CFUNITS_INSTALL_DIR

Package Notes
~~~~~~~~~~~~~

=================== =====================================
Apt-Package         Why?
=================== =====================================
libgdal-dev         ``shapely``, ``osgeo``, and ``fiona``
libgeos-dev         ``shapely`` speedups
libnetcdf-dev       ``netCDF4``
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

1. Download the current release: http://www.earthsystemmodeling.org/ocgis_releases/beta_releases/ocgis-0.07.1b/reg/OCGIS_Framework_Reg.html.
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
