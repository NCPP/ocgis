#!/bin/bash

# save the starting directory, so it can be restored after the script has finished
STARTDIR=`pwd`

echo "starting to install PostgreSQL..."

export POSTGRESQL_VER=8.4
export DBUSER=`logname`

sudo apt-get install -y postgresql-8.4
sudo apt-get install -y postgresql-server-dev-$POSTGRESQL_VER libpq-dev
sudo apt-get install -y postgresql-client-8.4

# Create a PostgreSQL user matching the current user's name so that the build
# can be tested using "make check"
echo "Creating database user $DBUSER"
sudo -u postgres createuser $DBUSER --superuser --pwprompt

echo "...finished installing PostgreSQL"
echo ""

# switch back to the starting directory
cd $STARTDIR
