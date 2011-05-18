#!/bin/bash

# save the starting directory, so it can be restored after the script has finished
STARTDIR=`pwd`

echo "starting to install Proj..."

PROJ_VER=4.7.0
PROJ_SRC=$SRCDIR/proj/$PROJ_VER
PROJ_DIR=/usr/local/proj/$PROJ_VER

if [ -e $PROJ_DIR ]; then
    echo "    The install directory $PROJ_DIR already exists; skipping installation..."
else
    echo "    building in $PROJ_SRC"
    mkdir -p $PROJ_SRC
    cd $PROJ_SRC
    wget http://download.osgeo.org/proj/proj-datumgrid-1.5.zip
    wget http://download.osgeo.org/proj/proj-$PROJ_VER.tar.gz
    unzip proj-datumgrid-1.5.zip -d proj-$PROJ_VER/nad/
    tar xzf proj-$PROJ_VER.tar.gz
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

# switch back to the starting directory
cd $STARTDIR
