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

   sudo apt-get install curl wget python-setuptools gdal-bin python-gdal
   sudo easy_install pip
   sudo pip install numpy
   sudo pip install shapely

Installing netcdf4-python_ is slightly more complex. Good instructions are available here: http://code.google.com/p/netcdf4-python/wiki/UbuntuInstall. The netcdf4-python_ tarball may be downloaded here: http://code.google.com/p/netcdf4-python/downloads/list.

Dependencies may also be built from source. A `bash script`_ is available containing a command structure for installing most of the OpenClimateGIS dependencies (no need to install PostGIS).

OpenClimateGIS Installation
---------------------------

There is currently no installer for OpenClimateGIS, and it must be installed from source_.

.. todo:: setup.py explanation

.. _Python 2.7: http://www.python.org/download/releases/2.7/
.. _netcdf4-python: http://code.google.com/p/netcdf4-python/
.. _bash script: https://github.com/NCPP/ocgis/blob/master/sh/install_geospatial.sh
.. _source: https://github.com/NCPP/ocgis
.. _CoG dependency page: http://www.earthsystemcog.org/projects/openclimategis/dependencies

Other Platforms
---------------

OpenClimateGIS has not been tested on other platforms. All libraries are theoretically supported on Mac and Windows.
