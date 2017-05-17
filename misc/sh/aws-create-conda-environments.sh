#!/usr/bin/env bash


PYTHON_VERSIONS=( "2.7" )
#PYTHON_VERSIONS=( "2.7" "3.6" )
CONDA_TEST_PREFIX="ocgis-py"
CHANNELS="-c nesii/channel/dev-ocgis -c conda-forge"
BASE_DEPENDENCIES="ocgis esmpy mpi4py rtree cf_units icclim nose"
OCGIS_DIR="/home/ubuntu/project/ocg/git/ocgis"


for ii in "${PYTHON_VERSIONS[@]}"; do
    CONDA_ENV_NAME="${CONDA_TEST_PREFIX}${ii}"
    conda create -y -n ${CONDA_ENV_NAME} ${CHANNELS} python=${ii} ${BASE_DEPENDENCIES}
    source activate ${CONDA_ENV_NAME}
    conda remove -y ocgis
done
