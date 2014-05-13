#!/bin/bash

sudo apt-get update
sudo apt-get install wget libnetcdf-dev libgeos-dev libgdal-dev libspatialindex-dev libudunits2-0 python-pip python-dev
sudo pip install numpy
sudo pip install netCDF4
sudo pip install shapely
sudo pip install fiona
sudo pip install rtree

###########
## osgeo ##
###########

## http://stackoverflow.com/questions/11336153/python-gdal-package-missing-header-file-when-installing-via-pip
pip install --no-install GDAL
cd /tmp/pip_build_ubuntu/GDAL
python setup.py build_ext --include-dirs=/usr/include/gdal
sudo python setup.py install

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
cd
python -c 'import cfunits,os,subprocess;cfunits_path=os.path.split(cfunits.__file__)[0];subprocess.call(["export","CFUNITS_INSTALL_DIR={0}".format(cfunits_path)],shell=True)'
sudo cp -r $CFUNITS_SETUP_DIR/cfunits/etc $CFUNITS_INSTALL_DIR