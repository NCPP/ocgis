#!/usr/bin/env bash

nps=(2 3 4 5 6 7 8)
#nps=(6 7 8)

tests=(../../src/ocgis/test)
#tests=(/home/benkoziol/l/ocgis/src/ocgis/test/test_ocgis/test_spatial/test_grid_splitter.py)
#tests=(/home/benkoziol/l/ocgis/src/ocgis/test/test_ocgis/test_variable)
#tests=(/home/benkoziol/l/ocgis/src/ocgis/test/test_ocgis/test_vm)

########################################################################################################################

function run_tests(){

for jj in "${tests[@]}"; do

    if [ ${RUN_SERIAL_TESTS} == "true" ]; then
        echo "====================="
        echo "Running serial tests: ${jj}"
        echo -e "=====================\\n"

        nosetests -vsx -a '!release' ${jj}
        if [ $? == 1 ]; then
            echo "FAIL: One or more serial tests failed."
            exit 1
        fi
    fi

    if [ ${RUN_PARALLEL_TESTS} == "true" ]; then
        for ii in "${nps[@]}"; do
            echo "======================="
            echo "Current MPI Test Suite: nproc=${ii}, path=${jj}"
            echo -e "=======================\\n"
            echo ""

            mpirun -n ${ii} nosetests -vsx -a 'mpi' ${jj}
        done
    fi
done

echo ""
echo "======================================================================"
echo "grep results:"
echo "======================================================================"
echo ""
grep FAIL ${OUT_FILE}
grep ERROR ${OUT_FILE}
echo ""

echo "Finished run_tests()"

}

run_tests 2>&1 | tee -a ${OUT_FILE}
