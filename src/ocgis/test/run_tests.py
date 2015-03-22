import os
import nose
import sys


def run(exclude='!slow,!remote,!esmpy7', dir_shpcabinet='~/data/ocgis_test_data/shp',
        dir_test_data='~/data/ocgis_test_data', simple=True):
    if simple:
        from ocgis.test import test_simple
        target = test_simple
    else:
        from ocgis import test
        target = test

    path = os.path.split(target.__file__)[0]
    os.environ['OCGIS_DIR_SHPCABINET'] = os.path.expanduser(dir_shpcabinet)
    os.environ['OCGIS_DIR_TEST_DATA'] = os.path.expanduser(dir_test_data)
    result = nose.run(argv=[sys.argv[0], path, '-a', exclude])
    if not result:
        sys.exit(1)
