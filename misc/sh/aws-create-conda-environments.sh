#!/usr/bin/env bash


#PYTHON_VERSIONS=( "2.7" )
PYTHON_VERSIONS=( "2.7" "3.6" )
CONDA_TEST_PREFIX="ocgis-py"
CHANNELS="-c nesii/channel/dev-ocgis -c nesii/channel/dev-esmf -c conda-forge"
BASE_DEPENDENCIES="ocgis esmpy mpi4py rtree cf_units icclim nose mock"
OCGIS_DIR=~/l/ocgis


for ii in "${PYTHON_VERSIONS[@]}"; do
    CONDA_ENV_NAME="${CONDA_TEST_PREFIX}${ii}"
    conda create -y -n ${CONDA_ENV_NAME} ${CHANNELS} python=${ii} ${BASE_DEPENDENCIES}
    source activate ${CONDA_ENV_NAME}
    conda remove -y ocgis
    source deactivate
done

# Create a core environment without optional dependencies.
conda create -y -n ocgis-core ${CHANNELS} ocgis nose mock python=3.6
conda remove -y rtree cf_units ocgis
