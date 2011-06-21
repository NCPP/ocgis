#!/bin/bash

# save the starting directory, so it can be restored after the script has finished
STARTDIR=`pwd`

echo "starting to install Python packages..."

export VIRTUALENVNAME=openclimategis

# create a python virtual environment
curl -O https://raw.github.com/pypa/virtualenv/master/virtualenv.py
python virtualenv.py --no-site-packages $VIRTUALENVNAME

# activate the virtual environment
. $VIRTUALENVNAME/bin/activate

# install the OpenClimateGIS dependencies
pip install yolk
pip install Django>=1.3
pip install psycopg2>=2.4
pip install numpy>=1.5.1
pip install netCDF4>=0.9.4

# finally, install the OpenClimateGIS Django project
pip install git+http://github.com/tylere/OpenClimateGIS

echo "...finished installing Python packages"
echo ""

# switch back to the starting directory
cd $STARTDIR
