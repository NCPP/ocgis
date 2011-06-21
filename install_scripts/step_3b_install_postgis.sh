#!/bin/bash

# save the starting directory, so it can be restored after the script has finished
STARTDIR=`pwd`

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

# switch back to the starting directory
cd $STARTDIR

#-----------------------------------------------
# Troubleshooting PostGIS
#-----------------------------------------------
# If successful, PostGIS installs the following files/directories:
#     /usr/share/postgresql/8.4/contrib/postgis-1.5/* 
#     /usr/lib/postgresql/8.4/lib/pgxs
#     /usr/lib/postgresql/8.4/lib/postgis-1.5.so
#     /usr/lib/postgresql/8.4/bin/pgsql2shp
#     /usr/lib/postgresql/8.4/bin/shp2pgsql
