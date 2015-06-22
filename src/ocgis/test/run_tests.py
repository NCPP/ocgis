import os
import sys
import re

import nose

import ocgis


class RunNoESMF(nose.plugins.Plugin):
    name = 'no-esmf'
    _exclude = 'test_regrid.test_base|test_conv.test_esmpy'

    def wantFile(self, file):
        match = re.search(self._exclude, file)
        if match is None:
            ret = None
        else:
            ret = False
        return ret


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


def run_no_esmf(dir_shpcabinet='~/data/ocgis_test_data/shp', dir_test_data='~/data/ocgis_test_data',
                attrs='!esmf,!esmpy7,!remote,!slow'):
    os.environ['OCGIS_DIR_SHPCABINET'] = os.path.expanduser(dir_shpcabinet)
    os.environ['OCGIS_DIR_TEST_DATA'] = os.path.expanduser(dir_test_data)
    result = nose.run(addplugins=[RunNoESMF()],
                      argv=[__file__, '-vs', '-a', attrs, '--with-no-esmf', 'ocgis'])
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
