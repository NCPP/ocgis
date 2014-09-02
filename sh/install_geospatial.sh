#!/bin/bash

SRCDIR=/usr/local/src
INSTALLDIR=/usr/local
J=4

##############
# INSTALL HDF5
##############

HDF_VER=1.8.13
HDF_SRC=$SRCDIR/hdf5/v$HDF_VER
HDF_DIR=$INSTALLDIR/hdf5/v$HDF_VER

mkdir -p $HDF_SRC
cd $HDF_SRC
wget http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-$HDF_VER.tar.gz
tar xzf hdf5-$HDF_VER.tar.gz
cd hdf5-$HDF_VER
./configure --prefix=$HDF_DIR --enable-shared --enable-hl
make -j $J
sudo make install
sudo ldconfig

#################
# INSTALL netCDF4
#################

NC_VER=4.2.1
NC_SRC=$SRCDIR/netcdf/v$NC_VER
NC_DIR=$INSTALLDIR/netcdf/v$NC_VER

sudo apt-get install libcurl4-openssl-dev
mkdir -p $NC_SRC
cd $NC_SRC
wget ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-$NC_VER.tar.gz
tar xzf netcdf-$NC_VER.tar.gz
cd netcdf-$NC_VER
export LDFLAGS=-L$HDF_DIR/lib
export CPPFLAGS=-I$HDF_DIR/include
export LD_LIBRARY_PATH=$HDF_DIR/lib
./configure --enable-netcdf-4 --enable-dap --enable-shared --prefix=$NC_DIR
make -j $J
sudo make install
sudo ldconfig

##############
# INSTALL PROJ
##############

## Note: If the error below is received, you may need to access the SVN trunk source code.
##       jniproj.c:52:26: fatal error: org_proj4_PJ.h: No such file or directory
## The SVN repo is here: http://svn.osgeo.org/metacrs/proj/trunk/proj
## See this ticket: http://trac.osgeo.org/proj/ticket/153

PROJ_VER=4.8.0
PROJ_SRC=$SRCDIR/proj/v$PROJ_VER
PROJ_DIR=$INSTALLDIR/proj/v$PROJ_VER

mkdir -p $PROJ_SRC
cd $PROJ_SRC
wget http://download.osgeo.org/proj/proj-datumgrid-1.5.zip
wget http://download.osgeo.org/proj/proj-$PROJ_VER.tar.gz
tar xzf proj-$PROJ_VER.tar.gz
unzip proj-datumgrid-1.5.zip -d proj-$PROJ_VER/nad/
cd proj-$PROJ_VER
./configure --prefix=$PROJ_DIR
make -j $J
sudo make install
sudo sh -c "echo '$PROJ_DIR/lib' > /etc/ld.so.conf.d/proj.conf" 
sudo ldconfig

##############
# INSTALL GEOS
##############

GEOS_VER=3.3.5
GEOS_SRC=$SRCDIR/geos/v$GEOS_VER
GEOS_DIR=$INSTALLDIR/geos/v$GEOS_VER

mkdir -p $GEOS_SRC
cd $GEOS_SRC
wget http://download.osgeo.org/geos/geos-$GEOS_VER.tar.bz2
tar xjf geos-$GEOS_VER.tar.bz2
cd geos-$GEOS_VER
./configure --prefix=$GEOS_DIR
make -j $J
sudo make install
sudo sh -c "echo '$GEOS_DIR/lib' > /etc/ld.so.conf.d/geos.conf" 
sudo ldconfig

##############
# INSTALL GDAL
##############

GDAL_VER=1.9.1
GDAL_SRC=$SRCDIR/gdal/v$GDAL_VER
GDAL_DIR=$INSTALLDIR/gdal/v$GDAL_VER
GEOS_CONFIG=$GEOS_DIR/bin/geos-config

mkdir -p $GDAL_SRC
cd $GDAL_SRC
wget http://download.osgeo.org/gdal/gdal-$GDAL_VER.tar.gz
tar xzf gdal-$GDAL_VER.tar.gz
cd gdal-$GDAL_VER
./configure --prefix=$GDAL_DIR --with-geos=$GEOS_CONFIG --with-python
make -j $J
sudo make install
sudo sh -c "echo '$GDAL_DIR/lib' > /etc/ld.so.conf.d/gdal.conf" 
sudo ldconfig

##############################
# INSTALL GDAL PYTHON BINDINGS
##############################

#GDAL_PYTHON_PREFIX=gdal-python
#GDAL_PYTHON_VER=1.9.1
#GDAL_PYTHON_SRC=$SRCDIR/$GDAL_PYTHON_PREFIX/v$GDAL_PYTHON_VER
#GDAL_PYTHON_DIR=$INSTALLDIR/$GDAL_PYTHON_PREFIX/v$GDAL_PYTHON_VER
#GDAL_PYTHON_URL=http://pypi.python.org/packages/source/G/GDAL/GDAL-$GDAL_PYTHON_VER.tar.gz
#
#mkdir -p $GDAL_PYTHON_SRC
#cd $GDAL_PYTHON_SRC
#wget $GDAL_PYTHON_URL
#tar xzf GDAL-$GDAL_PYTHON_VER.tar.gz
#cd GDAL-$GDAL_PYTHON_VER
### EDIT PATH TO GDAL-CONFIG (should be: '$GDAL_DIR/bin/gdal-config')
#emacs setup.cfg
#
#sudo python setup.py install

########################
# INSTALL NETCDF4-PYTHON
########################

NC4_PYTHON_PREFIX=netCDF4-python
NC4_PYTHON_VER=1.0.4
NC4_PYTHON_SRC=$SRCDIR/$NC4_PYTHON_PREFIX/v$NC4_PYTHON_VER
NC4_PYTHON_DIR=$INSTALLDIR/$NC4_PYTHON_PREFIX/v$NC4_PYTHON_VER
NC4_PYTHON_URL=https://netcdf4-python.googlecode.com/files/netCDF4-1.0.4.tar.gz

mkdir -p $NC4_PYTHON_SRC
cd $NC4_PYTHON_SRC
wget $NC4_PYTHON_URL
tar xzf netCDF4-1.0.4.tar.gz
cd netCDF4-$NC4_PYTHON_VER

#### make this the setup.cfg file ####
echo "[directories]
HDF5_dir=$HDF_DIR
HDF5_incdir=$HDF_DIR/include
HDF5_libdir=$HDF_DIR/lib
netCDF4_dir=$NC_DIR
netCDF4_incdir=$NC_DIR/include
netCDF4_libdir=$NC_DIR/lib" > setup.cfg

sudo python setup.py install

#################
# INSTALL SHAPELY
#################

sudo pip install shapely

#SHAPELY_PREFIX=Shapely
#SHAPELY_VER=1.2.15
#SHAPELY_SRC=$SRCDIR/$SHAPELY_PREFIX/v$SHAPELY_VER
#SHAPELY_DIR=$INSTALLDIR/$SHAPELY_PREFIX/v$SHAPELY_VER
#SHAPELY_URL=http://pypi.python.org/packages/source/S/Shapely/Shapely-1.2.15.tar.gz
#
#mkdir -p $SHAPELY_SRC
#cd $SHAPELY_SRC
#wget $SHAPELY_URL
#tar xzf $SHAPELY_PREFIX-$SHAPELY_VER.tar.gz
#cd $SHAPELY_PREFIX-$SHAPELY_VER
#
## encode uri of geos_c.h, search for line:
##  #include "geos_c.h"
#
## may need to downgrade geos to get speedups to work.
## libgeos_c is now libgeos_c.a in 3.x something
#
#sudo ln -s /usr/local/geos/v3.3.5/include/geos geos
#sudo ln -s /usr/local/geos/v3.3.5/include/geos_c.h geos_c.h
#
#sudo /home/local/WX/ben.koziol/Dropbox/.virtualenvs/ocg/bin/python setup.py install

###############
# INSTALL FIONA
###############

## Website: https://pypi.python.org/pypi/Fiona

## try this first!
sudo pip install fiona

## Pluto installation
module load gdal/1.9.1 python/2.7.2 gfortran/4.7.2/gcc/4.7.2/gcc geos/3.3.5 
cd /home/local/WX/ben.koziol/src/fiona/v0.16.1
wget --no-check-certificate https://pypi.python.org/packages/source/F/Fiona/Fiona-0.16.1.tar.gz
tar -xzvf Fiona-0.16.1.tar.gz
cd Fiona-0.16.1
[sudo] python setup.py install

#########################
# INSTALL CF-UNITS-PYTHON
#########################

# Website for UDUNITS: http://www.unidata.ucar.edu/software/udunits/
# Website for cfunits-python: https://code.google.com/p/cfunits-python/

CFUNITS_VER=0.9.6
CFUNITS_SRC=$SRCDIR/cfunits-python/v$CFUNITS_VER
CFUNITS_TARBALL=cfunits-$CFUNITS_VER.tar.gz
CFUNITS_URL=https://cfunits-python.googlecode.com/files/$CFUNITS_TARBALL
CFUNITS_INSTALL_DIR=/usr/local/lib/python2.7/dist-packages/cfunits

## install udunits C library
[sudo] apt-get install libudunits2-0

## install cfunits-python
mkdir -p $CFUNITS_SRC
cd $CFUNITS_SRC
wget $CFUNITS_URL
tar -xzvf $CFUNITS_TARBALL
cd cfunits-$CFUNITS_VER
[sudo] python setup.py install
# installation does not copy UDUNITS database
[sudo] cp -r cfunits/etc $CFUNITS_INSTALL_DIR

#################
# INSTALL BASEMAP
#################

cd /usr/local/src/basemap
wget https://github.com/matplotlib/basemap/archive/v1.0.7rel.tar.gz
tar -xzvf v1.0.7rel.tar.gz
cd v1.0.7rel
export GEOS_DIR=/usr/local/geos/v3.3.5
sudo -E python setup.py install

######################
# INSTALL PYTHON RTREE
######################

## http://libspatialindex.github.io/
## https://pypi.python.org/pypi/Rtree/

SRC_LIBSPATIALINDEX=/usr/local/src/libspatialindex/v1.8.1
INSTALL_LIBSPATIALINDEX=/usr/local/libspatialindex/v1.8.1
mkdir -p $SRC_LIBSPATIALINDEX
cd $SRC_LIBSPATIALINDEX
wget http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.1.tar.gz
tar -xzvf spatialindex-src-1.8.1.tar.gz
cd spatialindex-src-1.8.1
./configure prefix=$INSTALL_LIBSPATIALINDEX
make
sudo make install
sudo sh -c "echo '/usr/local/libspatialindex/v1.8.1/lib' > /etc/ld.so.conf.d/libspatialindex.conf" 
sudo ldconfig
sudo pip install rtree

#################
# INSTALL POSTGIS
#################

#POSTGIS_VER=2.0.1
#POSTGIS_SRC=$SRCDIR/postgis/v$POSTGIS_VER
#GEOS_CONFIG=/usr/local/geos/v3.3.5/bin/geos-config
#PROJ_DIR=/usr/local/proj/v4.8.0
#GDAL_CONFIG=/usr/local/gdal/v1.9.1/bin/gdal-config
#PG_TEMPLATE=postgis-$POSTGIS_VER-template
#PG_CONFIG=/usr/bin/pg_config
#
## install dependencies
#sudo apt-get install libpq-dev
#sudo apt-get install postgresql-server-dev-9.1
#sudo apt-get install postgresql-contrib-9.1
#
## install postgis
#TAR=postgis-$POSTGIS_VER.tar.gz
#mkdir -p $POSTGIS_SRC
#cd $POSTGIS_SRC
#wget http://postgis.refractions.net/download/$TAR
#tar -xzvf $TAR
#cd postgis-$POSTGIS_VER
#./configure --with-geosconfig=$GEOS_CONFIG --with-projdir=$PROJ_DIR --with-gdalconfig=$GDAL_CONFIG --with-raster --with-gui
#make -j $J
#sudo make install
#
## create template database
#sudo su -c "createdb '$PG_TEMPLATE'" - postgres
#sudo su -c "createlang plpgsql $PG_TEMPLATE" - postgres
#sudo su -c "psql -c \"UPDATE pg_database SET datistemplate='true' WHERE datname='$PG_TEMPLATE';\"" - postgres
#sudo -u postgres psql -f postgis/postgis.sql -d $PG_TEMPLATE
#sudo -u postgres psql -f spatial_ref_sys.sql -d $PG_TEMPLATE
#sudo -u postgres psql -f raster/rt_pg/rtpostgis.sql -d $PG_TEMPLATE
##sudo -u postgres psql -d $PG_TEMPLATE -c "GRANT ALL ON geometry_columns TO PUBLIC;"
##sudo -u postgres psql -d $PG_TEMPLATE -c "GRANT SELECT ON spatial_ref_sys TO PUBLIC;"
