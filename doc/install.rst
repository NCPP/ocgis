============
Installation
============

.. note:: First, make sure `Python 2.7`_ is installed and available on your system path.

Dependencies
------------

Additional information on software dependencies may be found on the project's `CoG dependency page`_.

Linux (Debian/Ubuntu)
~~~~~~~~~~~~~~~~~~~~~

A number of dependencies may be installed from the package repository:

.. code-block:: sh

   [sudo] apt-get install g++ libz-dev python-dev curl wget python-setuptools gdal-bin python-gdal
   [sudo] easy_install pip
   [sudo] pip install shapely

Installing netcdf4-python_ is slightly more complex:
 * The netcdf4-python_ tarball may be downloaded here: http://code.google.com/p/netcdf4-python/downloads/list.
 * Good instructions describing what to do after you have it are available here: http://code.google.com/p/netcdf4-python/wiki/UbuntuInstall.

Dependencies may also be built from source. A `bash script`_ is available containing a command structure for installing most of the OpenClimateGIS dependencies (no need to install PostGIS).

Other Platforms (Mac/Windows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenClimateGIS has not been tested on other platforms. All libraries are theoretically supported on Mac and Windows.

OpenClimateGIS Installation
---------------------------

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

Uninstalling will remove every OpenClimateGIS package exposed on the Python path.

.. code-block:: sh

   [sudo] python setup.py uninstall

.. _Python 2.7: http://www.python.org/download/releases/2.7/
.. _netcdf4-python: http://code.google.com/p/netcdf4-python/
.. _bash script: https://github.com/NCPP/ocgis/blob/master/sh/install_geospatial.sh
.. _source: https://github.com/NCPP/ocgis
.. _CoG dependency page: http://www.earthsystemcog.org/projects/openclimategis/dependencies
