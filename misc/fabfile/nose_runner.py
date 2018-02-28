import datetime
import os

import nose
from nose.plugins.plugintest import run

PATH_LOG = '/tmp/foo.txt'


class SimpleStream(object):

    def __init__(self, path):
        self.path = path
        self.write('', mode='w')
        self.writeln('OpenClimateGIS Test Results')
        self.writeln('Started {0} UTC'.format(datetime.datetime.utcnow()))
        self.writeln()

    def flush(self):
        pass

    def write(self, msg, mode='a'):
        print msg
        with open(self.path, mode) as f:
            f.write(msg)

    def writeln(self, msg=None, mode='a'):
        if msg is None:
            msg = '\n'
        else:
            msg = '{0}\n'.format(msg)
        print msg
        self.write(msg, mode=mode)


class NESIITestRunner(nose.plugins.Plugin):
    name = 'nesii-remote-tests'
    _days = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}

    def __init__(self, *args, **kwargs):
        self._path_log = kwargs.pop('path_log')
        super(NESIITestRunner, self).__init__(*args, **kwargs)
        self._ss = None

    def finalize(self, result):
        # skipped = len(result.skipped)
        errors = len(result.errors)
        failures = len(result.failures)
        # total = result.testsRun

        self._ss.writeln()
        total_bad = errors + failures
        self._ss.writeln('Test_Failures:{0}'.format(total_bad))
        self._ss.writeln('Day_of_Week:{0}'.format(self._days[datetime.datetime.now().weekday()]))

        if total_bad > 0:
            color = 'yellow'
        elif total_bad == 0:
            color = 'green'
        else:
            raise NotImplementedError(total_bad)

        self._ss.writeln('Test_results:{0}'.format(color))

    def setOutputStream(self, stream):
        self._ss = SimpleStream(self._path_log)
        return self._ss


if __name__ == '__main__':
    nose.main(addplugins=[NESIITestRunner(path_log='/tmp/nesii_test_results.log')],
              argv=[__file__, '-vs', os.getenv('OCGIS_TEST_TARGET'), '--with-nesii-remote-tests'])
