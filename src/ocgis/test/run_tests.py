import os
import sys

import nose


def run(attrs='simple', dir_shpcabinet='~/data/ocgis_test_data/shp', dir_test_data='~/data/ocgis_test_data',
        verbose=False):

    os.environ['OCGIS_DIR_SHPCABINET'] = os.path.expanduser(dir_shpcabinet)
    os.environ['OCGIS_DIR_TEST_DATA'] = os.path.expanduser(dir_test_data)
    argv = [sys.argv[0], 'ocgis']
    if attrs is not None:
        argv += ['-a', attrs]
    if verbose:
        argv.append('-v')
    result = nose.run(argv=argv)
    if not result:
        sys.exit(1)
