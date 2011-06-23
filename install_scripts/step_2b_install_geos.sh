#!/bin/bash

# save the starting directory, so it can be restored after the script has finished
STARTDIR=`pwd`

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

# switch back to the starting directory
cd $STARTDIR

#NOTE:
#set the GEOS_LIBRARY_PATH variable in the Django settings file to specify
#a specific version of GEOS to use.
#GEOS_LIBRARY_PATH = '$VIRTUALENV/lib/libgeos_c.so'
