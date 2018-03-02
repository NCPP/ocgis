#!/usr/bin/env bash

source ./logging.sh

PYTHON_VERSIONS=( "3.6" "2.7" )
CONDA_ROOT=/home/ubuntu/anaconda3

export OCGIS_BASH_LOGNAME="ocgis.aws-run-tests"
export OCGIS_DIR=~/sandbox/ocgis/src/ocgis
export OCGIS_DIR_TEST_DATA=~/storage/ocgis_test_data
export OCGIS_DIR_GEOMCABINET=${OCGIS_DIR_TEST_DATA}/shp

notify "starting"

# Run full test suites
if [ ${OCGIS_TEST_EXHAUSTIVE} != "false" ]; then
    rm ~/htmp/test-ocgis-python*.out
    for ii in "${PYTHON_VERSIONS[@]}"; do
        inf "running test suite for python version: ${ii}"
        cd ${OCGIS_DIR}/misc/sh || exit 1
        CONDA_ENV=ocgis-py${ii}|| exit 1
        source activate ${CONDA_ENV} || exit 1
        export OCGIS_TEST_OUT_FILE=~/htmp/test-ocgis-python${ii}.out
        bash ./test.sh || exit 1
    done

    # Run tests without big optional dependencies - no mpi4py, ESMF, icclim
    inf "running tests without dependencies"
    source activate ocgis-core || exit 1
    cd ${OCGIS_DIR} || exit 1
    python setup.py test || exit 1
else
    inf "skipping exhaustive tests"
fi

# Test installation and testing routines.
INSTALL_ENV=tmp-ocgis-install
rm -rf ${CONDA_ROOT}/envs/${INSTALL_ENV}
for ii in "${PYTHON_VERSIONS[@]}"; do
    inf "running installation test for python version: ${ii}"
    cd ${OCGIS_DIR} || exit 1
    conda create -y -n ${INSTALL_ENV} -c nesii/channel/dev-ocgis -c conda-forge ocgis nose mock python=${ii} || exit 1
    source activate ${INSTALL_ENV} || exit 1
    conda remove -y ocgis || exit 1
    python setup.py install || exit 1
    python setup.py uninstall || exit 1
    rm -rf build dist || exit 1
    cd /tmp || exit 1
    python -c "from ocgis.test.run_tests import run_simple; run_simple(verbose=False)" || exit 1
    ocli --help || exit 1
    source deactivate || exit 1
    rm -rf ${CONDA_ROOT}/envs/${INSTALL_ENV} || exit 1
done

notify "success"
