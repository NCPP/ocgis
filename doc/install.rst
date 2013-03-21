============
Installation
============

.. note:: First, make sure `Python 2.7`_ is installed and available on your system path.

Dependencies
------------

Additional information on software dependencies may be found on the project's `CoG dependency page`_.

Linux (Debian/Ubuntu)
~~~~~~~~~~~~~~~~~~~~~

A number of dependencies may be installed from the package repository and using Python's `easy_install`:

.. code-block:: sh

   [sudo] apt-get update
   [sudo] apt-get install g++ libz-dev curl wget python-dev python-setuptools python-gdal
   [sudo] easy_install shapely

.. _netCDF4-python-install:

Installing netcdf4-python_ is slightly more complex:
 * The netcdf4-python_ tarball may be downloaded here: http://code.google.com/p/netcdf4-python/downloads/list.
 * Good instructions describing what to do after you have it are available here: http://code.google.com/p/netcdf4-python/wiki/UbuntuInstall.

You may attempt to use this set of terminal commands (file names and URLs may need to be updated):

.. code-block:: sh
   
   SRC=<path-to-source-file-storage>
   PREFIX=/usr/local
   HDF5=hdf5-1.8.1.10-patch1
   NETCDF4=netcdf-4.2.1
   NETCDF4_PYTHON=netCDF4-1.0.4

   ## HDF5 ##
   cd $SRC
   wget http://www.hdfgroup.org/ftp/HDF5/current/src/$HDF5.tar.gz
   tar -xzvf $HDF5.tar.gz
   cd $HDF5
   ./configure --prefix=$PREFIX --enable-shared --enable-hl
   make 
   [sudo] make install

   ## NetCDF4 ##
   cd $SRC
   wget ftp://ftp.unidata.ucar.edu/pub/netcdf/$NETCDF4.tar.gz
   tar -xzvf $NETCDF4.tar.gz
   cd $NETCDF4
   LDFLAGS=-L$PREFIX/lib
   CPPFLAGS=-I$PREFIX/include
   ./configure --enable-netcdf-4 --enable-dap --enable-shared --prefix=$PREFIX
   make 
   [sudo] make install
   
   ## netCDF4-python ##
   cd $SRC
   # the "ldconfig" is not necessary for Mac OS X
   [sudo] ldconfig
   wget http://netcdf4-python.googlecode.com/files/$NETCDF4_PYTHON.tar.gz
   tar -xzvf $NETCDF4_PYTHON.tar.gz
   cd $NETCDF4_PYTHON
   [sudo] python setup.py install

Dependencies may also be built entirely from source. A `bash script`_ is available containing a command structure for installing most of the OpenClimateGIS dependencies (no need to install PostGIS).

Mac OS X
~~~~~~~~

1. Download and install GDAL from a pre-packaged DMG installer: http://www.kyngchaos.com/files/software/frameworks/GDAL_Complete-1.9.dmg.
2. Download and extract Shapely from: https://pypi.python.org/pypi/Shapely. Navigate into extracted folder. Then run:

   .. code-block:: sh

      [sudo] python setup.py install

3. Follow the instructions on installing :ref:`netCDF4 Python <netCDF4-python-install>`.

Windows
~~~~~~~

OpenClimateGIS has not been tested on Windows platforms. All libraries are theoretically supported.

Installing OpenClimateGIS
-------------------------

1. Download the current release: https://github.com/NCPP/ocgis/tags.
2. Extract the file using your favorite extraction utility.
3. Navigate into extracted directory.
4. Run the system command:

.. code-block:: sh

   [sudo] python setup.py install

5. Check that the package may be imported:

>>> import ocgis

or

.. code-block:: sh

   python -c 'import ocgis'

Configuring the :class:`~ocgis.ShpCabinet`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the path to the directory containing the shapefile folders in :attr:`ocgis.env.DIR_SHPCABINET`.

Uninstalling OpenClimateGIS
---------------------------

The `uninstall` command will simply provide you with the directory location of the OpenClimateGIS package. This must be manually removed.

.. code-block:: sh

    python setup.py uninstall

.. _Python 2.7: http://www.python.org/download/releases/2.7/
.. _netcdf4-python: http://code.google.com/p/netcdf4-python/
.. _bash script: https://github.com/NCPP/ocgis/blob/master/sh/install_geospatial.sh
.. _source: https://github.com/NCPP/ocgis
.. _CoG dependency page: http://www.earthsystemcog.org/projects/openclimategis/dependencies
