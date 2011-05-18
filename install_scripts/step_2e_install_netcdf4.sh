#!/bin/bash

echo "starting to install netCDF4..."

NETCDF4_VER=4.1.1
NETCDF4_SRC=$SRCDIR/hdf5/$NETCDF4_VER
NETCDF4_DIR=/usr/local/netCDF4/$NETCDF4_VER
export NETCDF4_DIR  # used by netcdf4-python

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

#NETCDF4_PYTHON_VER=0.9.2
#NETCDF4_PYTHON_SRC=$SRCDIR/netcdf4-python/$NETCDF4_PYTHON_VER
#NETCDF4_PYTHON_PATH=/usr/local/netcdf4-python/$NETCDF4_PYTHON_VER

echo "...finished installing netCDF4"
echo ""

cd $INSTALL_SCRIPT_DIR

