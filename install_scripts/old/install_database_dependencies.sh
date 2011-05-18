#!/bin/bash

# install database dependencies

SRCDIR=~/src
if ! [ -e $SRCDIR ]; then
    mkdir $SRCDIR
fi

# install PostgreSQL
POSTGRESQL_VER=8.4
sudo apt-get install postgresql-server-dev-$POSTGRESQL_VER libpq-dev

# Create a PostgreSQL user matching the current user's name so that the build
# can be tested using "make check"
DBUSER=`logname`
sudo -u postgres createuser $DBUSER --superuser --pwprompt

POSTGIS_VER=1.5.2
POSTGIS_SRC=$SRCDIR/postgis/$POSTGIS_VER
POSTGIS_DIR=/usr/share/postgresql/8.4/contrib/postgis-1.5
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

# create a PostGIS template database
POSTGIS_TEMPLATE=postgis-$POSTGIS_VER-template
sudo su -c "createdb $POSTGIS_TEMPLATE" - postgres
sudo su -c "createlang plpgsql $POSTGIS_TEMPLATE" - postgres
sudo -u postgres psql -d postgres -c "UPDATE pg_database SET datistemplate='true' WHERE datname='$POSTGIS_TEMPLATE';"
sudo -u postgres psql -d $POSTGIS_TEMPLATE -f /usr/share/postgresql/$POSTGRESQL_VER/contrib/postgis-1.5/postgis.sql
sudo -u postgres psql -d $POSTGIS_TEMPLATE -f /usr/share/postgresql/$POSTGRESQL_VER/contrib/postgis-1.5/spatial_ref_sys.sql
sudo -u postgres psql -d $POSTGIS_TEMPLATE -c "GRANT ALL ON geometry_columns TO PUBLIC;"
sudo -u postgres psql -d $POSTGIS_TEMPLATE -c "GRANT SELECT ON spatial_ref_sys TO PUBLIC;"
