#!/bin/bash

# save the starting directory, so it can be restored after the script has finished
STARTDIR=`pwd`

echo ""
echo "starting to create a PostGIS database for Django..."

# create a PostGIS database that will be used by the Django project
DBNAME=openclimategis_sql
DBOWNER=openclimategis_user
sudo -u postgres createuser $DBOWNER --pwprompt
sudo su -c "createdb $DBNAME -T $POSTGIS_TEMPLATE" - postgres
sudo -u postgres psql -d postgres -c "ALTER DATABASE $DBNAME OWNER TO $DBOWNER;"

echo "... finished creating a PostGIS database for Django"
echo ""

# switch back to the starting directory
cd $STARTDIR
