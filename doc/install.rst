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
   [sudo] apt-get install g++ libz-dev curl wget python-setuptools python-gdal
   [sudo] easy_install shapely

Installing netcdf4-python_ is slightly more complex:
 * The netcdf4-python_ tarball may be downloaded here: http://code.google.com/p/netcdf4-python/downloads/list.
 * Good instructions describing what to do after you have it are available here: http://code.google.com/p/netcdf4-python/wiki/UbuntuInstall.

You may attempt to use this set of terminal commands (file names and URLs may need to be updated):

.. code-block:: sh
   
   ## HDF5 ##
   cd /tmp
   wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.1.0-patch1.tar.gz
   tar -xzvf hdf5-1.8.1.0-patch1.tar.gz
   cd hdf5-1.8.1.0-patch1
   ./configure --prefix=/usr/local --enable-shared --enable-hl
   make 
   [sudo] make install

   ## NetCDF4 ##
   cd /tmp
   wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.2.1.tar.gz
   tar -xzvf netcdf-4.2.1.tar.gz
   cd netcdf-4.2.1
   LDFLAGS=-L/usr/local/lib CPPFLAGS=-I/usr/local/include
   ./configure --enable-netcdf-4 --enable-dap --enable-shared --prefix=/usr/local
   make 
   [sudo] make install
   
   ## netCDF4-python ##
   cd /tmp
   [sudo] ldconfig
   wget http://netcdf4-python.googlecode.com/files/netCDF4-1.0.2.tar.gz
   tar -xzvf netCDF4-1.0.2.tar.gz
   cd netCDF4-1.0.2
   [sudo] python setup.py install

Dependencies may also be built from source. A `bash script`_ is available containing a command structure for installing most of the OpenClimateGIS dependencies (no need to install PostGIS).

Other Platforms (Mac/Windows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenClimateGIS has not been tested on other platforms. All libraries are theoretically supported on Mac and Windows.

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

Configuring the :class:`~ocgis.ShpCabinet`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the path to the folder containing the shapefile folders in :attr:`ocgis.env.DIR_SHPCABINET`.

Uninstalling OpenClimateGIS
---------------------------

.. warning:: The `uninstall` command is currently not supported.

Uninstalling will remove every OpenClimateGIS package exposed on the Python path.

.. code-block:: sh

    [sudo] python setup.py uninstall

.. _Python 2.7: http://www.python.org/download/releases/2.7/
.. _netcdf4-python: http://code.google.com/p/netcdf4-python/
.. _bash script: https://github.com/NCPP/ocgis/blob/master/sh/install_geospatial.sh
.. _source: https://github.com/NCPP/ocgis
.. _CoG dependency page: http://www.earthsystemcog.org/projects/openclimategis/dependencies
