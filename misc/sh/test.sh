#!/usr/bin/env bash


export RUN_SERIAL_TESTS="true"
#export RUN_SERIAL_TESTS="false"

export RUN_PARALLEL_TESTS="true"
#export RUN_PARALLEL_TESTS="false"

# Wall time for the tests (seconds). Equal to 1.5*`time test.sh`.
WTIME=900

########################################################################################################################

cd ${OCGIS_DIR}/misc/sh || { echo "ERROR: Could not cd OCGIS_DIR: ${OCGIS_DIR}"; exit 1; }
rm ${OCGIS_TEST_OUT_FILE}
touch ${OCGIS_TEST_OUT_FILE} || exit 1
rm .noseids

timeout -k 5 --foreground ${WTIME} bash ./test-core.sh
if [ $? == 124 ]; then
    echo -e "\\n\\nFAIL: Hit wall time (${WTIME}s) when running test-core.sh" | tee -a ${OCGIS_TEST_OUT_FILE}
    exit 1
else
    echo -e "\\n\\nSUCCESS: test-core.sh finished" | tee -a ${OCGIS_TEST_OUT_FILE}
fi

exit 0