#!/usr/bin/env bash


CONDA_ENVS=( "ocgis-py36" "ocgis-py27" )

export OCGIS_DIR=~/sandbox/ocgis/src/ocgis
export OCGIS_DIR_TEST_DATA=~/storage/ocgis_test_data
export OCGIS_DIR_GEOMCABINET=${OCGIS_DIR_TEST_DATA}/shp

# Run full test suites
for ii in "${CONDA_ENVS[@]}"; do
    echo "running environment: ${ii}"
    cd ${OCGIS_DIR}/misc/sh || exit 1
    source activate ${ii} || exit 1
    export OCGIS_TEST_OUT_FILE=~/htmp/test-${ii}.out
    bash ./test.sh || exit 1
done

# Run tests without big optional dependencies - no mpi4py, ESMF, icclim
#conda create -n ocgis-core -c nesii/channel/dev-ocgis -c conda-forge ocgis nose python=3.6
source activate ocgis-core || exit 1
cd ${OCGIS_DIR} || exit 1
python setup.py test || exit 1
