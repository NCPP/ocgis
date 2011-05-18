#!/bin/bash

echo "starting to install PostgreSQL..."

POSTGRESQL_VER=8.4

sudo apt-get install postgresql-8.4
sudo apt-get install postgresql-server-dev-$POSTGRESQL_VER libpq-dev
sudo apt-get install postgresql-client-8.4

# Create a PostgreSQL user matching the current user's name so that the build
# can be tested using "make check"
DBUSER=`logname`
echo "Creating database user $DBUSER"
sudo -u postgres createuser $DBUSER --superuser --pwprompt

echo "...finished installing PostgreSQL"
echo ""

