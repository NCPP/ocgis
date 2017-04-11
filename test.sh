#!/usr/bin/env bash

OUT_FILE=test-ocgis.out
rm ${OUT_FILE}
rm .noseids

#nps=(1 2 5 8 12)
nps=(2 3 4 5 6 7 8)
#nps=(4 5 6 7 8)
#nps=(2)

#nps=(1)
#nps=(2)
#nps=(3)
#nps=(5)
#nps=(6)
#nps=(8)

tests=(
        src/ocgis/test

#        src/ocgis/test/test_ocgis/test_vm/test_mpi.py
#        /home/benkoziol/l/ocgis/src/ocgis/test/test_ocgis/test_vm/test_core.py
#        /home/benkoziol/l/ocgis/src/ocgis/test/test_ocgis/test_variable/test_dimension.py
#        /home/benkoziol/l/ocgis/src/ocgis/test/test_ocgis/test_variable/test_base.py
#        /home/benkoziol/l/ocgis/src/ocgis/test/test_ocgis/test_spatial/test_grid.py
#        /home/benkoziol/l/ocgis/src/ocgis/test/test_ocgis/test_variable/test_geom.py
#        /home/benkoziol/l/ocgis/src/ocgis/test/test_ocgis/test_collection/test_field.py
#        /home/benkoziol/l/ocgis/src/ocgis/test/test_ocgis/test_driver/test_request/test_core.py
#        /home/benkoziol/l/ocgis/src/ocgis/test/test_simple/test_simple.py
	)

for jj in "${tests[@]}"; do

    echo ${jj}
    nosetests -x -a '!release' ${jj} 2>&1 | tee -a ${OUT_FILE}

    for ii in "${nps[@]}"; do
        echo ${ii} ${jj}
        mpirun -n ${ii} nosetests -vsx -a 'mpi' ${jj} 2>&1 | tee -a ${OUT_FILE}
    done
done

grep FAIL ${OUT_FILE}
grep ERROR ${OUT_FILE}
