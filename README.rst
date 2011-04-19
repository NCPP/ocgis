==============
OpenClimateGIS
==============

OpenClimateGIS is a Python web service for distributing climate model data
in geospatial (vector) formats.

------------
Dependencies
------------

* PostgreSQL database
* psycopg2
* numpy
* netcdf4-python
* pykml

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Detailed Instructions for installing Dependencies (Ubuntu 10.04)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a directory for downloading source code::

    SRCDIR=~/src
    if ! [ -e $SRCDIR ]; then
        mkdir $SRCDIR
    fi

~~~~~~~~~~~~~~~~~~~~~
Installing Proj
~~~~~~~~~~~~~~~~~~~~~

Install Proj from source::

    PROJ_VER=4.7.0
    PROJ_SRC=$SRCDIR/proj/$PROJ_VER
    PROJ_DIR=/usr/local/proj/$PROJ_VER
    if [ -e $PROJ_DIR ]; then
        echo "The directory $PROJ_DIR already exists; skipping installation..."
    else
        mkdir -p $PROJ_SRC
        cd $PROJ_SRC
        wget http://download.osgeo.org/proj/proj-$PROJ_VER.tar.gz
        wget http://download.osgeo.org/proj/proj-datumgrid-1.5.zip
        tar xzf proj-$PROJ_VER.tar.gz
        cd proj-$PROJ_VER/nad
        unzip ../../proj-datumgrid-1.5.zip
        cd ..
        ./configure --prefix=$PROJ_DIR > log_proj_configure.out
        make > log_proj_make.out
        make install > log_proj_make_install.out
        sudo sh -c "echo $PROJ_DIR'/lib' > /etc/ld.so.conf.d/proj.conf"
        sudo ldconfig
        # QUICK TEST: search the shared libraries
        #    ldconfig -v | grep proj
        # This should return a path such as:
        #    /usr/local/proj/4.7.0/lib
    fi

~~~~~~~~~~~~~~~~~~~~~
Installing GEOS
~~~~~~~~~~~~~~~~~~~~~

Install GEOS from source::

    GEOS_VER=3.2.2
    GEOS_SRC=$SRCDIR/geos/$GEOS_VER
    GEOS_DIR=/usr/local/geos/$GEOS_VER
    if [ -e $GEOS_DIR ]; then
        echo "The directory $GEOS_DIR already exists; skipping installation..."
    else
        mkdir -p $GEOS_SRC
        cd $GEOS_SRC
        wget http://download.osgeo.org/geos/geos-$GEOS_VER.tar.bz2
        tar xjf geos-$GEOS_VER.tar.bz2
        cd geos-$GEOS_VER
        ./configure --prefix=$GEOS_DIR > log_geos_configure.out
        make > log_geos_make.out
        sudo make install > log_geos_make_install.out
        #NOTE: PostGIS install fails during the make install step if the GEOS library
        #path has not bee added to ldconfig 
        sudo sh -c "echo $GEOS_DIR'/lib' > /etc/ld.so.conf.d/geos.conf" 
        sudo ldconfig
        # QUICK TEST: search the shared libraries
        #    ldconfig -v | grep geos
        # This should return a path such as:
        #    /usr/local/geos/3.2.2/lib
    fi

~~~~~~~~~~~~~~~~~~~~~
Installing PostgreSQL
~~~~~~~~~~~~~~~~~~~~~

Install the required PostgreSQL libraries::

    sudo apt-get install postgresql-server-dev-8.4 libpq-dev

~~~~~~~~~~~~~~~~~~~~~
Installing PostGIS
~~~~~~~~~~~~~~~~~~~~~

Install the PostGIS spatial extension for PostgreSQL::

    POSTGIS_VER=1.5.2
    POSTGIS_DIR=$SRCDIR/postgis/$POSTGIS_VER
    POSTGIS_TEMPLATE=postgis-$POSTGIS_VER-template
    if [ -e $POSTGIS_DIR ]; then
        echo "PostGIS already exists; skipping installation..."
    else
        mkdir -p $POSTGIS_DIR
        cd $POSTGIS_DIR
        wget http://postgis.refractions.net/download/postgis-$POSTGIS_VER.tar.gz
        tar xzf postgis-$POSTGIS_VER.tar.gz
        cd postgis-$POSTGIS_VER
        ./configure --prefix=$POSTGIS_DIR \
            --with-geosconfig=$GEOS_DIR/bin/geos-config \
            --with-projdir=$PROJ_DIR \
            > log_postgis_configure.out
        make > log_postgis_make.out
        # PostGIS tries to install files in:
        #     /usr/share/postgresql/8.4/contrib and 
        #     /usr/lib/postgresql/8.4/lib
        #     /usr/lib/postgresql/8.4/bin
        sudo make install > log_postgis_make_install.out
    fi

~~~~~~~~~~~~~~~~
Installing numpy
~~~~~~~~~~~~~~~~

Install numpy from the Python Package Index::

    pip install numpy

~~~~~~~~~~~~~~~~~~~~~~~~~
Installing netcdf4-python
~~~~~~~~~~~~~~~~~~~~~~~~~

The following scripts install netcdf4-python (including its dependencies). If
the script fails, check to see if the URLs for HDF5 and netcdf4 are still 
accessible, since the project maintainers may have removed the source files
as they become outdated.

Install libcurl (necessary for OPeNDAP functionality)::

    sudo apt-get install libcurl3 libcurl4-openssl-dev

Install HDF5::

    HDF5_VER=1.8.6
    HDF5_DIR=/usr/local/hdf5/$HDF5_VER
    HDF5_SRC=$SRCDIR/hdf5/$HDF5_VER
    if [ -e $HDF5_DIR ]; then
        echo "HDF5 already exists; skipping installation..."
    else
        mkdir -p $HDF5_SRC
        cd $HDF5_SRC
        HDF5_TAR=hdf5-$HDF5_VER.tar.gz
        wget http://www.hdfgroup.org/ftp/HDF5/current/src/$HDF5_TAR
        tar -xzvf $HDF5_TAR
        cd hdf5-$HDF5_VER
        ./configure --prefix=$HDF5_DIR --enable-shared --enable-hl > log_hdf5_configure.log
        make > log_hdf5_make.log 
        sudo make install > log_hdf5_make_install.log 
        sudo sh -c "echo $HDF5_DIR'/lib' > /etc/ld.so.conf.d/hdf.conf" 
        sudo ldconfig
    fi

Install netcdf4::

    NETCDF4_VER=4.1.1
    NETCDF4_DIR=/usr/local/netCDF4/$NETCDF4_VER
    NETCDF4_PYTHON_VER=0.9.2
    NETCDF4_PYTHON_SRC=$SRCDIR/netcdf4-python/$NETCDF4_PYTHON_VER
    NETCDF4_PYTHON_PATH=/usr/local/netcdf4-python/$NETCDF4_PYTHON_VER
    NETCDF4_SRC=$SRCDIR/hdf5/$NETCDF4_VER
    if [ -e $NETCDF4_DIR ]; then
        echo "NETCDF4 already exists; skipping installation..."
    else
        mkdir -p $NETCDF4_SRC
        cd $NETCDF4_SRC
        NETCDF4_TAR=netcdf-$NETCDF4_VER.tar.gz
        wget ftp://ftp.unidata.ucar.edu/pub/netcdf/$NETCDF4_TAR
        tar -xzvf $NETCDF4_TAR
        cd netcdf-$NETCDF4_VER
        ./configure --enable-netcdf-4 --with-hdf5=$HDF5_DIR --enable-shared --enable-dap --prefix=$NETCDF4_DIR > log_netcdf_configure.log
        make > log_netcdf_make.log 
        sudo make install > log_netcdf_make_install.log 
        sudo sh -c "echo $NETCDF4_DIR'/lib' > /etc/ld.so.conf.d/netcdf.conf" 
        sudo ldconfig
    fi

Install netcdf4-python::

    export HDF5_DIR
    export NETCDF4_DIR
    pip install svn+http://netcdf4-python.googlecode.com/svn/trunk

Test the install::

    $ ipython
    Python 2.6.5 (r265:79063, Apr 16 2010, 13:57:41) 
    Type "copyright", "credits" or "license" for more information.

    IPython 0.10.2 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object'. ?object also works, ?? prints more.

    In [1]: import netCDF4

    In [2]: 


------------
Source Code
------------

The source code for OpenClimateGIS is available at::

    https://github.com/tylere/OpenClimateGIS

