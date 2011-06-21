#!/bin/bash

# save the starting directory, so it can be restored after the script has finished
STARTDIR=`pwd`

echo "starting to install Python packages..."

VIRTUALENVNAME=openclimategis

# create a python virtual environment
curl -O https://raw.github.com/pypa/virtualenv/master/virtualenv.py
python virtualenv.py --no-site-packages $VIRTUALENVNAME

# install the OpenClimateGIS dependencies
pip -E $VIRTUALENVNAME install yolk
pip -E $VIRTUALENVNAME install Django>=1.3
pip -E $VIRTUALENVNAME install psycopg2>=2.4
pip -E $VIRTUALENVNAME install numpy>=1.5.1
pip -E $VIRTUALENVNAME install netCDF4>=0.9.4

# finally, install the OpenClimateGIS Django project
pip -E $VIRTUALENVNAME install git+http://github.com/tylere/OpenClimateGIS

echo "...finished installing Python packages"
echo ""

# switch back to the starting directory
cd $STARTDIR
