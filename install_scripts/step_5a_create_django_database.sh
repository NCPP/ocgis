#!/bin/bash

echo ""
echo "starting to create a PostGIS database for Django..."

# create a PostGIS database that will be used by the Django project
DBNAME=openclimategis_sql
sudo su -c "createdb $DBNAME -T $POSTGIS_TEMPLATE" - postgres
sudo -u postgres psql -d postgres -c "ALTER DATABASE $DBNAME OWNER TO $DBOWNER;"

echo "... finished creating a PostGIS database for Django"
echo ""
