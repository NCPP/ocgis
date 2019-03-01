#!/usr/bin/env bash

########################################################################################################################

function run_tests_core(){

source ./logging.sh || exit 1

#export OCGIS_USE_NETCDF4_MPI="false"

export RUN_SERIAL_TESTS="true"
#export RUN_SERIAL_TESTS="false"

export RUN_PARALLEL_TESTS="true"
#export RUN_PARALLEL_TESTS="false"

export NOSE_ATTRS="!slow,!remote,!icclim,!mpi"

nps=(2 3 4 5 6 7 8)
#nps=(4)

tests=(../../src/ocgis/test)

for jj in "${tests[@]}"; do

    if [ ${RUN_SERIAL_TESTS} == "true" ]; then
        inf "Running serial tests: ${jj}"

        nosetests -vsx -a ${NOSE_ATTRS} ${jj}
        if [ $? == 1 ]; then
            error "One or more serial tests failed."
            exit 1
        fi
    fi

    if [ ${RUN_PARALLEL_TESTS} == "true" ]; then
        for ii in "${nps[@]}"; do
            inf "Current MPI Test Suite: nproc=${ii}, path=${jj}"

            mpirun -n ${ii} nosetests -vsx -a 'mpi,!slow' ${jj}
        done
    fi
done

inf "grep results:"
inf `grep FAIL ${OCGIS_TEST_OUT_FILE}`

# This error is printed by a GDAL library.
inf `grep -v -E "ERROR 1:.*not recognised as an available field" ${OCGIS_TEST_OUT_FILE} | grep ERROR`

inf "tests skipped because of HDF error:"
inf `grep "HDF sometimes has trouble reading the dataset" ${OCGIS_TEST_OUT_FILE}`

debug "finished run_tests_core()"

}
