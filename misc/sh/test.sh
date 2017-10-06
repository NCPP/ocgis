#!/usr/bin/env bash

# File for capturing stderr and stdout
export OUT_FILE=test-ocgis.out

export RUN_SERIAL_TESTS="true"
#export RUN_SERIAL_TESTS="false"

export RUN_PARALLEL_TESTS="true"
#export RUN_PARALLEL_TESTS="false"

# Wall time for the tests (seconds). Equal to 1.5*`time test-core.sh`.
WTIME=900

########################################################################################################################

rm ${OUT_FILE}
rm .noseids

timeout -k 5 --foreground ${WTIME} bash ./test-core.sh
if [ $? == 124 ]; then
    echo -e "\\n\\nFAIL: Hit wall time (${WTIME}s) when running test-core.sh" | tee -a ${OUT_FILE}
    exit 1
else
    echo -e "\\n\\nSUCCESS: test-core.sh finished" | tee -a ${OUT_FILE}
fi
