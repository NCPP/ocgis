## getting the development code
## following instructions from: https://github.com/UV-CDAT/uvcdat/wiki/Development

SRC_DIR=/home/local/WX/ben.koziol/src/uvcdat
SRC_GIT_MASTER=https://github.com/NCPP/uvcdat.git
SRC_GIT_DEVEL=https://github.com/NCPP/uvcdat-devel.git

cd $SRC_DIR
git clone $SRC_GIT_MASTER source
cd source
git remote add devel $SRC_GIT_DEVEL
git fetch --all
git submodule update --init

## building...

################
## INSTALL CMAKE
################

URL=http://www.cmake.org/files/v2.8/cmake-2.8.12.2.tar.gz
SRC_DIR_CMAKE=/usr/local/src/cmake/v2.8.12.2
TAR=cmake-2.8.12.2.tar.gz
CMAKE_DIR=cmake-2.8.12.2

mkdir -p $SRC_DIR_CMAKE
cd $SRC_DIR_CMAKE
wget $URL
tar -xzvf $TAR
cd $CMAKE_DIR
./configure
sudo make install

#############
## INSTALL QT
#############

QT_VER=4.8.5
QT_URL=http://download.qt-project.org/official_releases/qt/4.8/4.8.5/qt-everywhere-opensource-src-$QT_VER.tar.gz
QT_DIR=/usr/local/src/qt/v$QT_VER

mkdir -p $QT_DIR
cd $QT_DIR
wget $QT_URL
tar -xzvf qt-everywhere-opensource-src-$QT_VER.tar.gz
cd qt-everywhere-opensource-src-$QT_VER
./configure
make
sudo make install
sudo apt-get install qt4-qmake
## place a qt.conf file in the same director as qmake (i.e. whereis qmake)
## [Paths]
## Prefix=/usr/local/Trolltech/Qt-4.8.5/

##################
## INSTALL UV-CDAT
##################

mkdir $SRC_DIR/build-uvcdat
cd $SRC_DIR/build-uvcdat
ccmake ../source
## press "c" to get through the configure steps. may need to do this more than once.
## press "e" if there are no errors...
## for super lightweight build use (note OCGIS will not be installed with this option):
##   * CDAT_BUILD_ESGF = ON
## to avoid build issues with graphics and pax archive turn off:
##   * CDAT_BUILD_GRAPHICS
##   * CDAT_BUILD_GUI
##   * CDAT_BUILD_GUI_SUPPORT
## press "g" to generate...
make
## Path to UV-CDAT Python executable:
##   * $SRC_DIR/build-uvcdat/install/bin/python
## To run the UV-CDAT Python:
##   * source $SRC_DIR/build-uvcdat/install/bin/setup_runtime.sh
##   * $SRC_DIR/build-uvcdat/install/bin/python

##############################
## BUILDING OCGIS DEPENDENCIES
##############################

SRC_DIR=/home/local/WX/ben.koziol/src/uvcdat
source $SRC_DIR/build-uvcdat/install/bin/setup_runtime.sh
PYTHON=$SRC_DIR/build-uvcdat/install/bin/python

## shapely
sudo $PYTHON setup.py install

## fiona
sudo $PYTHON setup.py build_ext -I/usr/local/gdal/v1.9.1/include -L/usr/local/gdal/v1.9.1/lib -lgdal install

## netcdf4-python
echo "[directories]
HDF5_dir=/usr/local/hdf5/v1.8.9
HDF5_incdir=/usr/local/hdf5/v1.8.9/include
HDF5_libdir=/usr/local/hdf5/v1.8.9/lib
netCDF4_dir=/usr/local/netcdf/v4.2.1
netCDF4_incdir=/usr/local/netcdf/v4.2.1/include
netCDF4_libdir=/usr/local/netcdf/v4.2.1/lib" > setup.cfg
sudo $PYTHON setup.py install

## osgeo (gdal-python)
## edit path to gdal-config in setup.cfg (should be: '$GDAL_DIR/bin/gdal-config')
## recommend re-extracing the tarball, otherwise build errors against UV-CDAT's
##  python distribution may occur.
sudo $PYTHON setup.py install

## nose
sudo $SRC_DIR/build-uvcdat/install/bin/pip install nose