#!/bin/bash

# save the starting directory, so it can be restored after the script has finished
STARTDIR=`pwd`

echo "starting to install HDF5..."

#Install libcurl (necessary for OPeNDAP functionality)::
sudo apt-get install -y libcurl3 libcurl4-openssl-dev

HDF5_VER=1.8.6
HDF5_SRC=$SRCDIR/hdf5/$HDF5_VER
HDF5_DIR=/usr/local/hdf5/$HDF5_VER
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

# switch back to the starting directory
cd $STARTDIR
