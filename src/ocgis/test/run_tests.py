import os
import sys

import nose

import ocgis


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


def run_simple(verbose=False):
    path = os.path.realpath(ocgis.__file__)
    path = os.path.split(path)[0]
    path = os.path.join(path, 'test', 'test_simple')
    argv = [sys.argv[0], '-a', '!optional', path]
    if verbose:
        argv.append('-v')
    result = nose.run(argv=argv)
    if not result:
        sys.exit(1)
