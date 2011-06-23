#!/bin/bash

#=============================================================================
# Update and install build tools
#=============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Update the operating system
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sudo apt-get -y update
sudo apt-get -y upgrade

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install dependencies for building libraries from source
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sudo apt-get install -y wget
sudo apt-get install -y unzip
sudo apt-get install -y gcc
sudo apt-get install -y g++
sudo apt-get install -y swig

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install version control tools
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sudo apt-get install -y git-core

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Creating a folder for building from source code
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
export SRCDIR=~/src

echo ""
echo "Creating a directory for source files..."
if ! [ -e $SRCDIR ]; then
    mkdir $SRCDIR
    echo "... source file directory has been created."
else
    echo "... source file directory already exists."
fi

echo "... finished creating a directory for source files"
echo ""

#=============================================================================
# Install geospatial dependencies
#=============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install Proj.4
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "starting to install Proj..."

export PROJ_VER=4.7.0
export PROJ_SRC=$SRCDIR/proj/$PROJ_VER
export PROJ_DIR=/usr/local/proj/$PROJ_VER

if [ -e $PROJ_DIR ]; then
    echo "    The install directory $PROJ_DIR already exists; skipping installation..."
else
    echo "    building in $PROJ_SRC"
    mkdir -p $PROJ_SRC
    cd $PROJ_SRC
    wget http://download.osgeo.org/proj/proj-datumgrid-1.5.zip
    wget http://download.osgeo.org/proj/proj-$PROJ_VER.tar.gz
    tar xzf proj-$PROJ_VER.tar.gz
    unzip proj-datumgrid-1.5.zip -d proj-$PROJ_VER/nad/
    cd proj-$PROJ_VER
    ./configure --prefix=$PROJ_DIR > log_proj_configure.out
    make -j 4 > log_proj_make.out
    
    echo "    installing in $PROJ_DIR"
    sudo make install > log_proj_make_install.out
    sudo sh -c "echo '$PROJ_DIR/lib' > /etc/ld.so.conf.d/proj.conf" 
    sudo ldconfig
fi

echo "...finished installing Proj"
echo ""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install GEOS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "starting to install GEOS..."

export GEOS_VER=3.2.2
export GEOS_SRC=$SRCDIR/geos/$GEOS_VER
export GEOS_DIR=/usr/local/geos/$GEOS_VER

if [ -e $GEOS_DIR ]; then
    echo "    The install directory $GEOS_DIR already exists; skipping installation..."
else
    echo "    building in $GEOS_SRC"
    mkdir -p $GEOS_SRC
    cd $GEOS_SRC
    wget http://download.osgeo.org/geos/geos-$GEOS_VER.tar.bz2
    tar xjf geos-$GEOS_VER.tar.bz2
    cd geos-$GEOS_VER
    ./configure --prefix=$GEOS_DIR > log_geos_configure.out
    make -j 4 > log_geos_make.out
    
    echo "    installing in $GEOS_DIR"
    sudo make install > log_geos_make_install.out
    sudo sh -c "echo '$GEOS_DIR/lib' > /etc/ld.so.conf.d/geos.conf" 
    sudo ldconfig
fi

echo "...finished installing GEOS"
echo ""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Instal GDAL
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "starting to install GDAL..."

export GDAL_VER=1.8.0
export GDAL_SRC=$SRCDIR/gdal/$GDAL_VER
export GDAL_DIR=/usr/local/gdal/$GDAL_VER

if [ -e $GDAL_DIR ]; then
    echo "The install directory $GDAL_DIR already exists; skipping installation..."
else
    echo "    building in $GDAL_SRC"
    mkdir -p $GDAL_SRC
    cd $GDAL_SRC
    wget http://download.osgeo.org/gdal/gdal-$GDAL_VER.tar.gz
    tar xzf gdal-$GDAL_VER.tar.gz
    cd gdal-$GDAL_VER
    ./configure --prefix=$GDAL_DIR --with-geos=$GEOS_DIR/bin/geos-config > log_gdal_configure.out
    make -j 4 > log_gdal_make.out
    echo "    installing in $GDAL_DIR"
    sudo make install > log_gdal_make_install.out
    sudo sh -c "echo '$GDAL_DIR/lib' > /etc/ld.so.conf.d/gdal.conf" 
    sudo ldconfig
fi

echo "...finished installing GDAL"
echo ""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install HDF5
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "starting to install HDF5..."

#Install libcurl (necessary for OPeNDAP functionality)::
sudo apt-get install -y libcurl3 libcurl4-openssl-dev

export HDF5_VER=1.8.7
export HDF5_SRC=$SRCDIR/hdf5/$HDF5_VER
export HDF5_DIR=/usr/local/hdf5/$HDF5_VER
export HDF5_DIR  # used by netcdf4-python

if [ -e $HDF5_DIR ]; then
    echo "The install directory $HDF5_DIR already exists; skipping installation..."
else
    echo "    building in $HDF5_SRC"
    mkdir -p $HDF5_SRC
    cd $HDF5_SRC
    HDF5_TAR=hdf5-$HDF5_VER.tar.gz
    wget http://www.hdfgroup.org/ftp/HDF5/current/src/$HDF5_TAR
    tar -xzvf $HDF5_TAR
    cd hdf5-$HDF5_VER
    ./configure --prefix=$HDF5_DIR --enable-shared --enable-hl > log_hdf5_configure.log
    make -j 4 > log_hdf5_make.log
    
    echo "    installing in $HDF5_DIR"
    sudo make install > log_hdf5_make_install.log 
    sudo sh -c "echo $HDF5_DIR'/lib' > /etc/ld.so.conf.d/hdf.conf" 
    sudo ldconfig
fi

echo "...finished installing HDF5"
echo ""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install netCDF4
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "starting to install netCDF4..."

export NETCDF4_VER=4.1.1
export NETCDF4_SRC=$SRCDIR/hdf5/$NETCDF4_VER
export NETCDF4_DIR=/usr/local/netCDF4/$NETCDF4_VER
export NETCDF4_DIR

if [ -e $NETCDF4_DIR ]; then
    echo "The install directory $NETCDF4_DIR already exists; skipping installation..."
else
    echo "    building in $NETCDF4_SRC"
    mkdir -p $NETCDF4_SRC
    cd $NETCDF4_SRC
    NETCDF4_TAR=netcdf-$NETCDF4_VER.tar.gz
    wget ftp://ftp.unidata.ucar.edu/pub/netcdf/$NETCDF4_TAR
    tar -xzvf $NETCDF4_TAR
    cd netcdf-$NETCDF4_VER
    ./configure --enable-netcdf-4 --with-hdf5=$HDF5_DIR --enable-shared --enable-dap --prefix=$NETCDF4_DIR > log_netcdf_configure.log
    make -j 4 > log_netcdf_make.log 
    
    echo "    installing in $NETCDF4_DIR"
    sudo make install > log_netcdf_make_install.log
    sudo sh -c "echo $NETCDF4_DIR'/lib' > /etc/ld.so.conf.d/netcdf.conf" 
    sudo ldconfig
fi

echo "...finished installing netCDF4"
echo ""

#=============================================================================
# Install database dependencies
#=============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install PostgreSQL
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "starting to install PostgreSQL..."

export POSTGRESQL_VER=8.4
export DBUSER=`logname`

sudo apt-get install -y postgresql-8.4
sudo apt-get install -y postgresql-server-dev-$POSTGRESQL_VER libpq-dev
sudo apt-get install -y postgresql-client-8.4

# Create a PostgreSQL user matching the current user's name so that the build
# can be tested using "make check"
echo "Creating database user $DBUSER"
sudo -u postgres createuser $DBUSER --superuser --pwprompt

echo "...finished installing PostgreSQL"
echo ""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install PostGIS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "starting to install PostGIS..."

sudo apt-get install -y libxml2-dev

export POSTGIS_VER=1.5.2
export POSTGIS_SRC=$SRCDIR/postgis/$POSTGIS_VER
export POSTGIS_DIR=/usr/share/postgresql/8.4/contrib/postgis-1.5

if [ -e $POSTGIS_DIR ]; then
    echo "The destination directory $POSTGIS_DIR already exists; skipping installation..."
else
    mkdir -p $POSTGIS_SRC
    cd $POSTGIS_SRC
    wget http://postgis.refractions.net/download/postgis-$POSTGIS_VER.tar.gz
    tar xzf postgis-$POSTGIS_VER.tar.gz
    cd postgis-$POSTGIS_VER
    ./configure \
        --with-geosconfig=$GEOS_DIR/bin/geos-config \
        --with-projdir=$PROJ_DIR \
        > log_postgis_configure.out
    make > log_postgis_make.out
    make check > log_postgis_make_check.out
    sudo make install > log_postgis_make_install.out
fi

echo "...finished installing PostGIS"
echo ""

#-----------------------------------------------------------------------------
# Install a PostGIS template database
#-----------------------------------------------------------------------------
echo "starting to create a PostGIS template database..."

export POSTGIS_TEMPLATE=postgis-$POSTGIS_VER-template

# create a PostGIS template database
sudo su -c "createdb $POSTGIS_TEMPLATE" - postgres
sudo su -c "createlang plpgsql $POSTGIS_TEMPLATE" - postgres
sudo -u postgres psql -d postgres -c "UPDATE pg_database SET datistemplate='true' WHERE datname='$POSTGIS_TEMPLATE';"
sudo -u postgres psql -d $POSTGIS_TEMPLATE -f /usr/share/postgresql/$POSTGRESQL_VER/contrib/postgis-1.5/postgis.sql
sudo -u postgres psql -d $POSTGIS_TEMPLATE -f /usr/share/postgresql/$POSTGRESQL_VER/contrib/postgis-1.5/spatial_ref_sys.sql
sudo -u postgres psql -d $POSTGIS_TEMPLATE -c "GRANT ALL ON geometry_columns TO PUBLIC;"
sudo -u postgres psql -d $POSTGIS_TEMPLATE -c "GRANT SELECT ON spatial_ref_sys TO PUBLIC;"

echo "...finished creating a PostGIS template database"
echo ""

#=============================================================================
# install Python packages
#=============================================================================

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Install Python dependencies
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo "starting to install Python packages..."

export VIRTUALENVNAME=openclimategis

sudo apt-get install python2.6-dev

# create a python virtual environment
curl -O https://raw.github.com/pypa/virtualenv/master/virtualenv.py
python virtualenv.py --no-site-packages $VIRTUALENVNAME

# activate the virtual environment
. $VIRTUALENVNAME/bin/activate

# install the OpenClimateGIS dependencies
pip install yolk
pip install Django==1.3
pip install psycopg2==2.4
pip install numpy==1.5.1
pip install netCDF4==0.9.4

echo "...finished installing Python packages"
echo ""

#=============================================================================
# Install OpenClimateGIS
#=============================================================================
pip install git+http://github.com/tylere/OpenClimateGIS

#=============================================================================
# configure Django
#=============================================================================
echo ""
echo "starting to create a PostGIS database for Django..."

# create a PostGIS database that will be used by the Django project
DBNAME=openclimategis_sql
DBOWNER=openclimategis_user
sudo -u postgres createuser $DBOWNER --pwprompt
sudo su -c "createdb $DBNAME -T $POSTGIS_TEMPLATE" - postgres
sudo -u postgres psql -d postgres -c "ALTER DATABASE $DBNAME OWNER TO $DBOWNER;"

echo "... finished creating a PostGIS database for Django"
echo ""

