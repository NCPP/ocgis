import itertools
import logging
import os

import ocgis
from ocgis import env
from ocgis.exc import OcgWarning
from ocgis.test.base import TestBase
from ocgis.test.base import attr
from ocgis.util.helpers import get_temp_path
from ocgis.util.logging_ocgis import ocgis_lh, ProgressOcgOperations, OcgisLogging


class TestOcgisLogging(TestBase):
    def tearDown(self):
        ocgis_lh.shutdown()
        TestBase.tearDown(self)

    def test_init(self):
        OcgisLogging()
        self.assertIsNone(logging._warnings_showwarning)

    def test_call(self):
        # test warning is logged to the terminal
        self.assertTrue(ocgis_lh.null)

        def _run_():
            ocgis_lh.configure()
            self.assertTrue(ocgis_lh.null)
            env.SUPPRESS_WARNINGS = False
            ocgis_lh(level=logging.WARNING, exc=RuntimeWarning('show me'))
            env.SUPPRESS_WARNINGS = True

        self.assertWarns(RuntimeWarning, _run_)

        # test warning is logged to the terminal and also logged to file
        ocgis_lh.shutdown()

        def _run_():
            env.SUPPRESS_WARNINGS = False
            logpath = self.get_temporary_file_path('ocgis.log')
            ocgis_lh.configure(to_file=logpath)
            exc = FutureWarning('something is about to happen')
            ocgis_lh(level=logging.WARNING, exc=exc)
            with open(logpath, 'r') as f:
                lines = f.readlines()
                lines = ''.join(lines)
            self.assertIn('FutureWarning', lines)
            self.assertIn('something is about to happen', lines)
            env.SUPPRESS_WARNINGS = True

        self.assertWarns(FutureWarning, _run_)

        # test a warning without an exception
        ocgis_lh.shutdown()

        def _run_():
            env.SUPPRESS_WARNINGS = False
            logpath = self.get_temporary_file_path('foo.log')
            ocgis_lh.configure(to_file=logpath)
            ocgis_lh(msg='hey there', level=logging.WARN)
            env.SUPPRESS_WARNINGS = True

        self.assertWarns(OcgWarning, _run_)

        # test suppressing warnings
        ocgis_lh.shutdown()

        def _run_():
            logpath = self.get_temporary_file_path('foo.log')
            ocgis_lh.configure(to_file=logpath)
            ocgis_lh(msg='oh my', level=logging.WARN)
            with open(logpath, 'r') as f:
                lines = f.readlines()
                lines = ''.join(lines)
            self.assertIn('OcgWarning', lines)
            self.assertIn('oh my', lines)

        with self.assertRaises(AssertionError):
            self.assertWarns(OcgWarning, _run_)

    def test_system_combinations(self):
        _to_stream = [
            True,
            False
        ]
        _to_file = [
            os.path.join(env.DIR_OUTPUT, 'test_ocgis_log.log'),
            None
        ]

        _level = [logging.INFO, logging.DEBUG, logging.WARN]
        for ii, (to_file, to_stream, level) in enumerate(itertools.product(_to_file, _to_stream, _level)):
            ocgis_lh.configure(to_file=to_file, to_stream=to_stream, level=level)
            try:
                ocgis_lh(ii)
                ocgis_lh('a test message')
                subset = ocgis_lh.get_logger('subset')
                interp = ocgis_lh.get_logger('interp')
                ocgis_lh('a subset message', logger=subset)
                ocgis_lh('an interp message', logger=interp)
                ocgis_lh('a general message', alias='foo', ugid=10)
                ocgis_lh('another message', level=level)
                if to_file is not None:
                    self.assertTrue(os.path.exists(to_file))
                    os.remove(to_file)
            finally:
                logging.shutdown()

    def test_system_exc(self):
        to_file = os.path.join(env.DIR_OUTPUT, 'test_ocgis_log.log')
        to_stream = False
        ocgis_lh.configure(to_file=to_file, to_stream=to_stream)
        try:
            raise ValueError
        except Exception as e:
            with self.assertRaises(ValueError):
                ocgis_lh('something happened', exc=e)

    def test_system_simple(self):
        to_file = os.path.join(env.DIR_OUTPUT, 'test_ocgis_log.log')
        to_stream = False

        ocgis_lh.configure(to_file, to_stream)

        ocgis_lh('a test message')
        subset = ocgis_lh.get_logger('subset')
        subset.info('a subset message')

    def test_system_with_callback(self):
        fp = get_temp_path(wd=self.current_dir_output)

        def callback(message, path=fp):
            with open(path, 'a') as sink:
                sink.write(message)
                sink.write('\n')

        class FooError(Exception):
            pass

        ocgis_lh.configure(callback=callback)
        ocgis_lh(msg='this is a test message')
        ocgis_lh()
        ocgis_lh(msg='this is a second test message')
        ocgis_lh(msg='this should not be there', level=logging.DEBUG)
        exc = FooError('foo message for value error')
        try:
            ocgis_lh(exc=exc)
        except FooError:
            pass
        with open(fp, 'r') as source:
            lines = source.readlines()
        self.assertEqual(lines, ['this is a test message\n', 'this is a second test message\n',
                                 'FooError: foo message for value error\n'])

    def test_configure(self):
        # test suppressing warnings in the logger
        self.assertIsNone(logging._warnings_showwarning)
        env.SUPPRESS_WARNINGS = False
        ocgis_lh.configure()
        self.assertFalse(logging._warnings_showwarning)

    def test_shutdown(self):
        env.SUPPRESS_WARNINGS = False
        ocgis_lh.configure(to_stream=True)
        self.assertFalse(logging._warnings_showwarning)
        ocgis_lh.shutdown()
        self.assertIsNone(logging._warnings_showwarning)

    @attr('data')
    def test_writing(self):
        env.ENABLE_FILE_LOGGING = True
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, snippet=True, output_format='csv')
        ret = ops.execute()
        folder = os.path.split(ret)[0]
        log = os.path.join(folder, 'logs', '{}-rank-0.log'.format(ops.prefix))
        with open(log) as f:
            lines = f.readlines()
            self.assertTrue(len(lines) >= 4)


class TestProgressOcgOperations(TestBase):
    def test_constructor(self):
        prog = ProgressOcgOperations(lambda x, y: (x, y))
        self.assertEqual(prog.n_operations, 1)
        self.assertEqual(prog.n_completed_operations, 0)

    def test_simple(self):
        n_geometries = 3
        n_calculations = 3

        def callback(percent, message):
            return percent, message

        for cb in [callback, None]:
            prog = ProgressOcgOperations(callback=cb, n_geometries=n_geometries, n_calculations=n_calculations)
            n_operations = 9
            self.assertEqual(prog.n_operations, n_operations)
            prog.mark()
            if cb is None:
                self.assertEqual(prog.percent_complete, 100 * (1 / float(n_operations)))
                self.assertEqual(prog(), None)
            else:
                self.assertEqual(prog(), (100 * (1 / float(n_operations)), None))
            prog.mark()
            if cb is None:
                self.assertEqual(prog.percent_complete, (100 * (2 / float(n_operations))))
            else:
                self.assertEqual(prog(message='hi'), (100 * (2 / float(n_operations)), 'hi'))

    def test_hypothetical_operations_loop(self):

        def callback(percent, message):
            return percent, message

        n = [0, 1, 2]
        for n_subsettables, n_geometries, n_calculations in itertools.product(n, n, n):
            try:
                prog = ProgressOcgOperations(callback,
                                             n_subsettables=n_subsettables,
                                             n_geometries=n_geometries,
                                             n_calculations=n_calculations)
            except AssertionError:
                if n_geometries == 0 or n_subsettables == 0:
                    continue
                else:
                    raise

            for ns in range(n_subsettables):
                for ng in range(n_geometries):
                    for nc in range(n_calculations):
                        prog.mark()
                    if n_calculations == 0:
                        prog.mark()
            self.assertEqual(prog(), (100.0, None))
