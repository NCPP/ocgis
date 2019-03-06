#!/usr/bin/env bash

########################################################################################################################

function run_tests_core(){

[ -z "${TESTS}" ] && echo 'ERROR: TESTS not set' && exit 1
[ -z "${RUN_SERIAL_TESTS}" ] && echo 'ERROR: RUN_SERIAL_TESTS not set' && exit 1
[ -z "${RUN_PARALLEL_TESTS}" ] && echo 'ERROR: RUN_PARALLEL_TESTS not set' && exit 1
[ -z "${NOSE_ATTRS}" ] && echo 'ERROR: NOSE_ATTRS not set' && exit 1
[ -z "${MPI_NOSE_ATTRS}" ] && echo 'ERROR: MPI_NOSE_ATTRS not set' && exit 1

source ./logging.sh || exit 1

nps=(2 3 4 5 6 7 8)
#nps=(4)

for jj in "${TESTS[@]}"; do
    if [ ${RUN_SERIAL_TESTS} == "true" ]; then
        inf "Running serial tests: ${jj}"

        nosetests -vs -a ${NOSE_ATTRS} ${jj}
        if [ $? == 1 ]; then
            error "One or more serial tests failed."
            exit 1
        fi
    fi

    if [ ${RUN_PARALLEL_TESTS} == "true" ]; then
        for ii in "${nps[@]}"; do
            inf "Current MPI Test Suite: nproc=${ii}, path=${jj}"

            mpirun -n ${ii} nosetests -vsx -a ${MPI_NOSE_ATTRS} ${jj}
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
