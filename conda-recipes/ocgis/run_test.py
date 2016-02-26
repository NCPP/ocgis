import os

from ocgis.test import run_simple, run_no_esmf

test_target = os.environ.get('CBUILD_OCGIS_TEST_TARGET')
if test_target == 'simple' or test_target is None:
    run_simple()
elif test_target == 'all':
    run_no_esmf()
else:
    raise NotImplementedError(test_target)
