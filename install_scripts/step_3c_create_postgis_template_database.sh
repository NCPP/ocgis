#!/bin/bash

echo "starting to create a PostGIS template database..."

export POSTGIS_TEMPLATE=postgis-$POSTGIS_VER-template

# create a PostGIS template database
sudo su -c "createdb $POSTGIS_TEMPLATE" - postgres
sudo su -c "createlang plpgsql $POSTGIS_TEMPLATE" - postgres
sudo -u postgres psql -d postgres -c "UPDATE pg_database SET datistemplate='true' WHERE datname='$POSTGIS_TEMPLATE';"
sudo -u postgres psql -d $POSTGIS_TEMPLATE -f /usr/share/postgresql/$POSTGRESQL_VER/contrib/postgis-1.5/postgis.sql
sudo -u postgres psql -d $POSTGIS_TEMPLATE -f /usr/share/postgresql/$POSTGRESQL_VER/contrib/postgis-1.5/spatial_ref_sys.sql
sudo -u postgres psql -d $POSTGIS_TEMPLATE -c "GRANT ALL ON geometry_columns TO PUBLIC;"
sudo -u postgres psql -d $POSTGIS_TEMPLATE -c "GRANT SELECT ON spatial_ref_sys TO PUBLIC;"

echo "...finished creating a PostGIS template database"
echo ""

