#!/usr/bin/env bash

########################################################################################################################

function run_tests_core(){

source ./logging.sh || exit 1

export RUN_SERIAL_TESTS="true"
#export RUN_SERIAL_TESTS="false"

export RUN_PARALLEL_TESTS="true"
#export RUN_PARALLEL_TESTS="false"

nps=(2 3 4 5 6 7 8)

tests=(../../src/ocgis/test)

for jj in "${tests[@]}"; do

    if [ ${RUN_SERIAL_TESTS} == "true" ]; then
        inf "Running serial tests: ${jj}"

        nosetests -vsx ${jj}
        if [ $? == 1 ]; then
            error "One or more serial tests failed."
            exit 1
        fi
    fi

    if [ ${RUN_PARALLEL_TESTS} == "true" ]; then
        for ii in "${nps[@]}"; do
            inf "Current MPI Test Suite: nproc=${ii}, path=${jj}"

            mpirun -n ${ii} nosetests -vsx -a 'mpi' ${jj}
        done
    fi
done

inf "grep results:"
inf `grep FAIL ${OCGIS_TEST_OUT_FILE}`

# This error is printed by a GDAL library.
inf `grep -v -E "ERROR 1:.*not recognised as an available field" ${OCGIS_TEST_OUT_FILE} | grep ERROR`

debug "finished run_tests_core()"

}

#function run_tests(){
#
##source ./logging.sh || exit 1
#source ./test-core.sh || exit 1
#
#export RUN_SERIAL_TESTS="true"
##export RUN_SERIAL_TESTS="false"
#
#export RUN_PARALLEL_TESTS="true"
##export RUN_PARALLEL_TESTS="false"
#
## Wall time for the tests (seconds). Equal to 1.5*`time test.sh`.
##WTIME=900
#
#########################################################################################################################
#
#notify "starting test.sh"
#
##cd ${OCGIS_DIR}/misc/sh || { error "Could not cd to OCGIS_DIR: ${OCGIS_DIR}"; exit 1; }
#rm .noseids
#
##$(timeout -k 5 --foreground ${WTIME} bash ./test-core.sh)
#
#run_tests_core
#
##if [ $? == 124 ]; then
##    error "Hit wall time (${WTIME}s) when running test-core.sh"
##    exit 1
##else
#notify "sucess test.sh"
##fi
#
#}