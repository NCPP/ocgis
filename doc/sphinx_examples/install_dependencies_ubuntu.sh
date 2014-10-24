#!/bin/bash

sudo apt-get update
sudo apt-get install wget libhdf5-dev libnetcdf-dev libgeos-dev libgdal-dev libspatialindex-dev libudunits2-0 libproj-dev python-pip python-dev
sudo pip install numpy netCDF4 shapely fiona rtree

###########
## osgeo ##
###########

## If this fails, try the apt-get install below...

## http://stackoverflow.com/questions/11336153/python-gdal-package-missing-header-file-when-installing-via-pip
pip install --no-install GDAL
cd /tmp/pip_build_ubuntu/GDAL
python setup.py build_ext --include-dirs=/usr/include/gdal
sudo python setup.py install

## If the above install fails, you can install from apt-get...
#sudo apt-get install python-gdal

#############
## cfunits ##
#############

CFUNITS_SRCDIR=/tmp/build_cfunits
CFUNITS_VER=0.9.6
CFUNITS_SRC=$CFUNITS_SRCDIR/cfunits-python/v$CFUNITS_VER
CFUNITS_TARBALL=cfunits-$CFUNITS_VER.tar.gz
CFUNITS_URL=https://cfunits-python.googlecode.com/files/$CFUNITS_TARBALL

mkdir -p $CFUNITS_SRC
cd $CFUNITS_SRC
wget $CFUNITS_URL
tar -xzvf $CFUNITS_TARBALL
cd cfunits-$CFUNITS_VER
sudo python setup.py install
## installation does not copy UDUNITS database
CFUNITS_SETUP_DIR=`pwd`
## assumes a standard location. the installation directory may be retrieved by running the command:
##  python -c "import cfunits, os;print(os.path.split(cfunits.__file__)[0])"
CFUNITS_INSTALL_DIR=/usr/local/lib/python2.7/dist-packages/cfunits
sudo cp -r $CFUNITS_SETUP_DIR/cfunits/etc $CFUNITS_INSTALL_DIR

#########
# ESMPy #
#########

# download the ESMF tarball via registration page:
#  http://www.earthsystemmodeling.org/download/releases.shtml

# link for building ESMF:
#  http://www.earthsystemmodeling.org/esmf_releases/public/ESMF_6_3_0rp1/ESMF_usrdoc/

# notes:
#  * will need fortran and c compiler

ESMF_TAR=esmf_6_3_0rp1_src.tar.gz
ESMF_VER='v6.3.0rp1'
ESMF_SRCDIR=/tmp/esmf/$ESMF_VER
ESMF_INSTALL_PREFIX=<path to esmf install directory>

## ESMF framework install ##

sudo apt-get install gfortran g++
mkdir -p $ESMF_SRCDIR
cd $ESMF_SRCDIR
<make sure ESMF tarball is available in the source directory>
tar -xzvf $ESMF_TAR
cd esmf
export ESMF_DIR=`pwd`
make
export ESMF_INSTALL_PREFIX=$ESMF_INSTALL_PREFIX
export ESMF_INSTALL_LIBDIR=$ESMF_INSTALL_PREFIX/lib
sudo -E make install

## ESMPy install ##

cd $ESMF_SRCDIR/esmf/src/addon/ESMPy
python setup.py build --ESMFMKFILE=$ESMF_INSTALL_PREFIX/lib/esmf.mk
sudo python setup.py install
