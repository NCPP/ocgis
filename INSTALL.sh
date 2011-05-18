#!/bin/bash

INSTALL_SCRIPT_DIR=$HOME

# Update and install build tools
. install_scripts/step_1a_install_update_system.sh
. install_scripts/step_1b_install_install_build_dependencies.sh

# Install geospatial dependencies
. install_scripts/step_2a_install_proj4.sh
. install_scripts/step_2b_install_geos.sh
. install_scripts/step_2c_install_gdal.sh
. install_scripts/step_2d_install_hdf5.sh
. install_scripts/step_2e_install_netcdf4.sh

# Install database dependencies
. install_scripts/step_3a_install_postgresql.sh
. install_scripts/step_3b_install_postgis.sh
. install_scripts/step_3c_create_postgis_template_database.sh

# install Python packages
. install_scripts/step_4a_install_python_packages.sh

# configure Django
. install_scripts/step_5a_create_django_database.sh

# switch back to the home directory
cd

