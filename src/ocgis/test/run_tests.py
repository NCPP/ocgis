import sys

import nose
import six


class RunAll(nose.plugins.Plugin):
    name = 'test_all'
    target = 'ocgis'
    _attrs_standard = '!remote'

    def __init__(self, *args, **kwargs):
        self.attrs = kwargs.pop('attrs', self._attrs_standard)
        self.verbose = kwargs.pop('verbose', True)

        super(RunAll, self).__init__(*args, **kwargs)

    def get_argv(self):
        argv = [sys.argv[0], '--with-{0}'.format(self.name), '-s']

        if isinstance(self.attrs, six.string_types):
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


class RunMore(RunAll):
    name = 'test_more'
    _attrs_standard = '!slow,!remote,!data'


class RunNoESMF(RunAll):
    name = 'no-esmf'
    # _exclude = 'test_regrid.test_base|test_conv.test_esmpy'
    _attrs_standard = '!esmf,!esmpy7,!remote,!slow,!data'

    # def wantFile(self, file):
    #     match = re.search(self._exclude, file)
    #     if match is None:
    #         ret = None
    #     else:
    #         ret = False
    #     return ret


class RunSimple(RunAll):
    name = 'test_simple'
    _attrs_standard = 'simple'


class RunMPINoData(RunAll):
    name = 'test_mpi'
    _attrs_standard = 'mpi,!data'


def _run_tests_(nose_plugin):
    result = nose.run(addplugins=[nose_plugin], argv=nose_plugin.get_argv())
    if not result:
        sys.exit(1)


def run_more(**kwargs):
    nose_plugin = RunMore(**kwargs)
    _run_tests_(nose_plugin)


def run_all(**kwargs):
    nose_plugin = RunAll(**kwargs)
    _run_tests_(nose_plugin)


def run_mpi_nodata(**kwargs):
    nose_plugin = RunMPINoData(**kwargs)
    _run_tests_(nose_plugin)


def run_no_esmf(**kwargs):
    nose_plugin = RunNoESMF(**kwargs)
    _run_tests_(nose_plugin)


def run_simple(**kwargs):
    nose_plugin = RunSimple(**kwargs)
    _run_tests_(nose_plugin)
