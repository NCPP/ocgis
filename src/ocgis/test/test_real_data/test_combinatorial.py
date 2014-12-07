import os
import shutil

from ocgis import OcgOperations
from ocgis.test.base import TestBase, attr


class TestCombinatorial(TestBase):

    def iter_dataset(self):
        for as_request_dataset in [True, False]:
            for k in self.test_data.iterkeys():
                kwds = {}
                if k == 'cmip3_extraction':
                    dimension_map = {'R': 'projection', 'T': 'time', 'Y': 'latitude', 'X': 'longitude'}
                    kwds['dimension_map'] = dimension_map
                rd = self.test_data.get_rd(k, kwds=kwds)
                if as_request_dataset:
                    yield k, rd
                else:
                    yield k, rd.get()

    @attr('slow')
    def test(self):
        import logbook

        log = logbook.Logger(name='combos', level=logbook.INFO)

        for key, dataset in self.iter_dataset():

            # if key != 'qed_2013_TNn_annual_min': continue

            # these datasets have only one time element
            if key in ('qed_2013_TNn_annual_min',
                       'qed_2013_TasMin_seasonal_max_of_seasonal_means',
                       'qed_2013_climatology_Tas_annual_max_of_annual_means',
                       'qed_2013_maurer02v2_median_txxmmedm_january_1971-2000',
                       'qed_2013_maurer02v2_median_txxmmedm_february_1971-2000',
                       'qed_2013_maurer02v2_median_txxmmedm_march_1971-2000',
                       'snippet_maurer_dtr',
                       'snippet_seasonalbias'):
                slc = None
            else:
                slc = [None, [10, 20], None, None, None]

            # this has different data types on the bounds for the coordinate variables. they currently get casted by the
            # software.
            if key == 'maurer_bcca_1991':
                check_types = False
            else:
                check_types = True

            log.debug('processing: {0} ({1})'.format(key, dataset.__class__.__name__))
            ops = OcgOperations(dataset=dataset, output_format='nc', prefix='nc1', slice=slc)
            try:
                log.debug('initial write...')
                ret1 = ops.execute()
            except ValueError:
                # realization dimensions may not be written to netCDF yet
                if key == 'cmip3_extraction':
                    continue
                else:
                    raise
            else:
                try:
                    ops2 = OcgOperations(dataset={'uri': ret1}, output_format='nc', prefix='nc2')
                    log.debug('second write...')
                    ret2 = ops2.execute()
                    log.debug('comparing...')
                    self.assertNcEqual(ret1, ret2, ignore_attributes={'global': ['history']}, check_types=check_types)
                finally:
                    for path in [ret1, ret2]:
                        folder = os.path.split(path)[0]
                        shutil.rmtree(folder)
        log.debug('success')
