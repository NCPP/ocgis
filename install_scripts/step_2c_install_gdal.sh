#!/bin/bash

# save the starting directory, so it can be restored after the script has finished
STARTDIR=`pwd`

echo "starting to install GDAL..."

GDAL_VER=1.8.0
GDAL_SRC=$SRCDIR/gdal/$GDAL_VER
GDAL_DIR=/usr/local/gdal/$GDAL_VER

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
    make > log_gdal_make.out
    
    echo "    installing in $GDAL_DIR"
    sudo make install > log_gdal_make_install.out
    sudo sh -c "echo '$GDAL_DIR/lib' > /etc/ld.so.conf.d/gdal.conf" 
    sudo ldconfig
fi

echo "...finished installing GDAL"
echo ""

# switch back to the starting directory
cd $STARTDIR
