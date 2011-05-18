#!/bin/bash

echo "starting to install Python packages..."

VIRTUALENVNAME=openclimategis
mkvirtualenv --no-site-packages $VIRTUALENVNAME

pip -E $VIRTUALENVNAME install Django>=1.3
pip -E $VIRTUALENVNAME install psycopg2>=2.4
pip -E $VIRTUALENVNAME install numpy>=1.5.1
pip -E $VIRTUALENVNAME install netCDF4>=0.9.4
pip -E $VIRTUALENVNAME install git+http://github.com/tylere/OpenClimateGIS

echo "...finished installing Python packages"
echo ""

