import os
import nose
import sys


def run(exclude='!slow,!remote,!esmpy7', dir_shpcabinet='~/data/ocgis_test_data/shp',
        dir_test_data='~/data/ocgis_test_data'):
    from ocgis import test

    module_name = os.path.split(test.__file__)[0]
    os.environ['OCGIS_DIR_SHPCABINET'] = os.path.expanduser(dir_shpcabinet)
    os.environ['OCGIS_DIR_TEST_DATA'] = os.path.expanduser(dir_test_data)
    result = nose.run(argv=[sys.argv[0], module_name, '-a', exclude])
    if not result:
        sys.exit(1)
