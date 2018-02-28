============
Installation
============

If you would like to subscribe to the low-volume, OpenClimateGIS mailing list, please fill out the :ref:`download form <download-form>` linked below. Filling out the form also helps us collect useful usage statistics.

Anaconda Package
----------------

An `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ package is available through `conda-forge <https://conda-forge.github.io/>`_.

Using the Package Manager
+++++++++++++++++++++++++

Installation with all optional dependencies:

.. code-block:: sh

   conda install -c conda-forge -c nesii ocgis esmpy mpi4py cf_units rtree icclim nose

Installation without optional dependencies:

.. code-block:: sh

   conda install -c conda-forge ocgis

Alternatively, NESII provides linux-64 and osx-64 builds for both OpenClimateGIS and ESMPy:

.. code-block:: sh

   conda install -c nesii -c conda-forge ocgis esmpy

Using an Environment File
+++++++++++++++++++++++++

An Anaconda `environment file <https://conda.io/docs/using/envs.html#use-environment-from-file>`_ installing all OpenClimateGIS dependencies is available in the repository root.

.. code-block:: sh

    cd <OpenClimateGIS source directory>
    conda env create -f environment.yml

Building from Source
--------------------

.. _download-form:

1. Download the current release:

 * http://www.earthsystemmodeling.org/ocgis_releases/public/ocgis-2.0.0/reg/OCGIS_Framework_Reg.html

2. Extract the file using your favorite extraction utility.
3. Navigate into extracted directory.
4. Run the command:

.. code-block:: sh

   [sudo] python setup.py install

Testing the Installation
------------------------

It is recommended that a simple suite of tests are run to verify the new installation. Testing requires the Python ``nose`` library (https://nose.readthedocs.io/en/latest/) and ``mock`` (included by default with Python 3+):

.. code-block:: sh

    [sudo] pip install nose mock
        # OR
    conda install nose mock

Run tests:

.. code-block:: sh

    python -c "from ocgis.test import run_simple; run_simple(verbose=False)"

Optional dependencies may also be tested. If an optional dependency is not installed, a test failure will occur:

.. code-block:: sh

    python -c "from ocgis.test import run_simple; run_simple(attrs=['simple', 'optional'], verbose=False)"

Tests may also be ran with a setup command:

.. code-block:: sh

    python setup.py test

Please report any errors to the support email address.

Configuring the :class:`~ocgis.GeomCabinet`
-------------------------------------------

Set the path to the directory containing the shapefiles or shapefile folders using :ref:`env.DIR_GEOMCABINET <env.DIR_GEOMCABINET>`. You may also set the system environment variable ``OCGIS_DIR_GEOMCABINET``.

Supported Python Versions
-------------------------

Python versions 2.7, 3.5, and 3.6 are tested and supported. Versions 2.7 or 3.6 are recommended. Python 3.5 demonstrates undefined behavior with parallel operations using ``mpi4py``.

Dependencies
------------

OpenClimateGIS is tested against the library versions listed below.

Required
++++++++

============== ======= ========================================
Package Name   Version URL
============== ======= ========================================
``numpy``      1.14.1  http://www.numpy.org/
``netCDF4``    1.3.1   http://unidata.github.io/netcdf4-python/
``gdal``       2.2.3   https://pypi.python.org/pypi/GDAL/
``pyproj``     1.9.5.1 https://github.com/jswhit/pyproj
``shapely``    1.6.4   https://pypi.python.org/pypi/Shapely
``fiona``      1.7.11  https://pypi.python.org/pypi/Fiona
``six``        1.11.0  https://pypi.python.org/pypi/six
``setuptools`` 38.5.1  https://pypi.python.org/pypi/setuptools
============== ======= ========================================

Optional
++++++++

Optional dependencies are listed below. OpenClimateGIS will still operate without these libraries installed but functionality and performance may change.

============= ======= ====================================================== =================================================================================================================================
Package Name  Version  URL                                                    Usage
============= ======= ====================================================== =================================================================================================================================
``ESMF``      7.1.0   https://www.earthsystemcog.org/projects/esmpy/releases Supports regridding operations.
``mpi4py``    3.0.0   http://mpi4py.readthedocs.io/en/stable/                Required for parallel execution.
``rtree``     0.8.3   https://pypi.python.org/pypi/Rtree/                    Constructs spatial indexes at runtime. Useful for complicated GIS operations (i.e. large or complex polygons for subsetting)
``cf_units``  1.2.0   https://github.com/SciTools/cf_units                   Allows unit transformations.
``icclim``    4.2.9   http://icclim.readthedocs.io/en/latest/                Calculation of the full suite of European Climate Assessment (ECA) indices with optimized code implementation.
``nose``      1.3.7   https://nose.readthedocs.io/en/latest/                 Run unit tests.
``mock``      2.0.0   https://pypi.python.org/pypi/mock/                     Run unit tests (required for Python 2.7 but part of Python 3 standard library).
============= ======= ====================================================== =================================================================================================================================

Building from Source
~~~~~~~~~~~~~~~~~~~~

Dependencies may be built entirely from source. An (outdated) `bash script`_ is available on GitHub.

Uninstalling
------------

The ``uninstall`` command will simply provide you with the directory location of the OpenClimateGIS package. This must be manually removed.

.. code-block:: sh

    python setup.py uninstall

.. _bash script: https://github.com/NCPP/ocgis/blob/master/misc/sh/install_geospatial.sh
.. _source: https://github.com/NCPP/ocgis
