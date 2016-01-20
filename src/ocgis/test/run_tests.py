import os
import re
import sys

import nose

import ocgis


class RunOcgis(nose.plugins.Plugin):
    name = 'ocgis'
    target = 'ocgis'
    _attrs_standard = '!esmpy7,!slow,!remote,!data'

    def __init__(self, *args, **kwargs):
        self.attrs = kwargs.pop('attrs', self._attrs_standard)
        self.verbose = kwargs.pop('verbose', True)

        dir_shpcabinet = kwargs.pop('dir_shpcabinet', '~/data/ocgis_test_data/shp')
        dir_test_data = kwargs.pop('dir_test_data', '~/data/ocgis_test_data')
        os.environ['OCGIS_DIR_SHPCABINET'] = os.path.expanduser(dir_shpcabinet)
        os.environ['OCGIS_DIR_TEST_DATA'] = os.path.expanduser(dir_test_data)

        super(RunOcgis, self).__init__(*args, **kwargs)

    def get_argv(self):
        argv = [sys.argv[0], '--with-{0}'.format(self.name), '-s']

        if isinstance(self.attrs, basestring):
            attrs = [self.attrs]
        else:
            attrs = self.attrs
        for a in attrs:
            argv.extend(['-a', a])

        if self.verbose:
            argv.append('-v')
        argv += self._get_argv_adds_()
        argv.append(self.target)
        return argv

    def _get_argv_adds_(self):
        return []


class RunNoESMF(RunOcgis):
    name = 'no-esmf'
    _exclude = 'test_regrid.test_base|test_conv.test_esmpy'
    _attrs_standard = '!esmf,!esmpy7,!remote,!slow,!data'

    def wantFile(self, file):
        match = re.search(self._exclude, file)
        if match is None:
            ret = None
        else:
            ret = False
        return ret


class RunSimple(RunOcgis):
    _attrs_standard = 'simple,!optional'

    @property
    def target(self):
        path = os.path.realpath(ocgis.__file__)
        path = os.path.split(path)[0]
        path = os.path.join(path, 'test', 'test_simple')
        return path


def _run_tests_(nose_plugin):
    result = nose.run(addplugins=[nose_plugin], argv=nose_plugin.get_argv())
    if not result:
        sys.exit(1)


def run_all(**kwargs):
    nose_plugin = RunOcgis(**kwargs)
    _run_tests_(nose_plugin)


def run_no_esmf(**kwargs):
    nose_plugin = RunNoESMF(**kwargs)
    _run_tests_(nose_plugin)


def run_simple(**kwargs):
    nose_plugin = RunSimple(**kwargs)
    _run_tests_(nose_plugin)
