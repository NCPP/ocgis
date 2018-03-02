import itertools
import unittest
from datetime import datetime as dt
from unittest.case import SkipTest

import numpy as np
from ocgis.api.operations import OcgOperations
from ocgis.api.request import RequestDatasetCollection, RequestDataset

import ocgis
from ocgis.calc.library import QEDDynamicPercentileThreshold
from ocgis.test.base import TestBase
from ocgis.util.large_array import compute


class TestQedBase(TestBase):

    def setUp(self):
        raise (SkipTest('dev'))

    pass


class TestSnowfallWaterEquivalent(TestQedBase):

    def setUp(self):
        super(self.__class__, self).setUp()
        ocgis.env.DIR_DATA = '/usr/local/climate_data/maurer/2010-concatenated'

    @property
    def maurer_pr(self):
        ret = {'uri': 'Maurer02new_OBS_pr_daily.1971-2000.nc', 'variable': 'pr'}
        return (ret)

    @property
    def maurer_tas(self):
        ret = {'uri': 'Maurer02new_OBS_tas_daily.1971-2000.nc', 'variable': 'tas'}
        return (ret)

    def test_output_data(self):
        uri = [
            '/home/local/WX/ben.koziol/climate_data/QED-2013/sfwe/maurer02v2/sfwe_p/sfwe_p.nc',
            '/home/local/WX/ben.koziol/climate_data/QED-2013/sfwe/maurer02v2/sfwe/sfwe.nc',
            '/home/local/WX/ben.koziol/climate_data/QED-2013/sfwe/maurer02v2/pr/pr.nc'
        ]
        variable = [
            'sfwe_p',
            'sfwe',
            'p',
        ]
        rds = [RequestDataset(u, v, time_region={'month': [7], 'year': range(1990, 2000)}) for u, v in
               zip(uri, variable)]

        ops = ocgis.OcgOperations(dataset=rds, geom='state_boundaries', select_ugid=[16],
                                  spatial_operation='clip', aggregate=True)
        ret = ops.execute()
        ref = ret[16]

        sfwe_p = ref.variables['sfwe_p'].value
        p = ref.variables['p'].value
        sfwe = ref.variables['sfwe'].value

        idx_bad = sfwe_p > 1
        bad_sfwe_p = sfwe_p[idx_bad]
        self.assertEqual(bad_sfwe_p.compressed().shape[0], 0)

        idx_bad = sfwe_p > 1
        bad_sfwe = sfwe[idx_bad]
        bad_p = p[idx_bad]

        rds = [RequestDataset(u, v) for u, v in zip(uri, variable)]

        ops = ocgis.OcgOperations(dataset=rds, geom='WBDHU8_June2013', select_ugid=[378],
                                  spatial_operation='clip', aggregate=True)
        ret = ops.execute()
        ref = ret[378]
        import ipdb;
        ipdb.set_trace()

    def test_calculate(self):
        #        ocgis.env.VERBOSE = True
        #        ocgis.env.DEBUG = True

        calc = [{'func': 'sfwe', 'name': 'sfwe', 'kwds': {'tas': 'tas', 'pr': 'pr'}}]
        time_range = [dt(1990, 1, 1), dt(1990, 3, 31)]
        rds = []
        for var in [self.maurer_pr, self.maurer_tas]:
            var.update({'time_range': time_range})
            rds.append(var)
        geom = 'state_boundaries'
        select_ugid = [16]
        ops = OcgOperations(dataset=rds, geom=geom, select_ugid=select_ugid,
                            calc=calc, calc_grouping=['month'], output_format='nc')
        ret = ops.execute()

    def test_calculate_compute(self):
        #        ocgis.env.VERBOSE = True
        #        ocgis.env.DEBUG = True
        calc = [{'func': 'sfwe', 'name': 'sfwe', 'kwds': {'tas': 'tas', 'pr': 'pr'}}]
        time_range = None
        rds = []
        for var in [self.maurer_pr, self.maurer_tas]:
            var.update({'time_range': time_range})
            rds.append(var)
        rdc = RequestDatasetCollection(rds)
        sfwe = compute(rdc, calc, ['month', 'year'], 50, verbose=True, prefix='sfwe')
        import ipdb;
        ipdb.set_trace()


class TestDynamicPercentiles(TestQedBase):

    def setUp(self):
        raise (SkipTest('dev'))

    def get_file(self, dir_output):
        ## leap year is 1996. time range will be 1995-1997.
        uri = '/usr/local/climate_data/maurer/2010-concatenated/Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
        rd = ocgis.RequestDataset(uri, variable, time_region={'year': [1995, 1996, 1997]})
        ops = ocgis.OcgOperations(dataset=rd, prefix='subset_1995-1997_Maurer02new_OBS_tasmax_daily.1971-2000',
                                  output_format='nc', dir_output=dir_output)
        ret = ops.execute()
        return (ret)

    def get_request_dataset(self, time_region=None):
        uri = '/home/local/WX/ben.koziol/climate_data/snippets/subset_1995-1997_Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
        rd = ocgis.RequestDataset(uri, variable, time_region=time_region)
        return (rd)

    def make_file(self):
        dir_output = '/tmp'
        print(self.get_file(dir_output))

    def test_get_day_index(self):
        rd = self.get_request_dataset()
        dates = rd.ds.temporal.value
        qdt = QEDDynamicPercentileThreshold()
        di = qdt._get_day_index_(dates)
        self.assertEqual((di['index'] == 366).sum(), 1)
        return (di)

    def test_is_leap_year(self):
        years = np.array([1995, 1996, 1997])
        qdt = QEDDynamicPercentileThreshold()
        is_leap = map(qdt._get_is_leap_year_, years)
        self.assertNumpyAll(is_leap, [False, True, False])

    def test_get_dynamic_index(self):
        di = self.test_get_day_index()
        qdt = QEDDynamicPercentileThreshold()
        dyidx = map(qdt._get_dynamic_index_, di.flat)

    def test_calculate(self):
        ocgis.env.DIR_BIN = '/home/local/WX/ben.koziol/links/ocgis/bin/QED_2013_dynamic_percentiles'
        percentiles = [90, 92.5, 95, 97.5]
        operations = ['gt', 'gte', 'lt', 'lte']
        calc_groupings = [
            ['month'],
            #                          ['month','year'],
            #                          ['year']
        ]
        uris_variables = [[
            '/home/local/WX/ben.koziol/climate_data/maurer/2010-concatenated/Maurer02new_OBS_tasmax_daily.1971-2000.nc',
            'tasmax'],
            [
                '/home/local/WX/ben.koziol/climate_data/maurer/2010-concatenated/Maurer02new_OBS_tasmin_daily.1971-2000.nc',
                'tasmin']]
        geoms_select_ugids = [
            ['qed_city_centroids', None],
            ['state_boundaries', [39]],
            #                              ['us_counties',[2416,1335]]
        ]
        for tup in itertools.product(percentiles, operations, calc_groupings, uris_variables, geoms_select_ugids):
            print(tup)
            percentile, operation, calc_grouping, uri_variable, geom_select_ugid = tup
            ops = OcgOperations(dataset={'uri': uri_variable[0], 'variable': uri_variable[1],
                                         'time_region': {'year': [1990], 'month': [6, 7, 8]}},
                                geom=geom_select_ugid[0], select_ugid=geom_select_ugid[1],
                                calc=[{'func': 'qed_dynamic_percentile_threshold',
                                       'kwds': {'operation': operation, 'percentile': percentile}, 'name': 'dp'}],
                                calc_grouping=calc_grouping, output_format='numpy')
            ret = ops.execute()

    def test_get_geometries_with_percentiles(self):
        bin_directory = '/home/local/WX/ben.koziol/links/project/ocg/bin/QED_2013_dynamic_percentiles'
        qdt = QEDDynamicPercentileThreshold()
        percentiles = [90, 92.5, 95, 97.5]
        shp_keys = ['qed_city_centroids', 'state_boundaries', 'us_counties']
        variables = ['tmin', 'tmax']
        for percentile, shp_key, variable in itertools.product(percentiles, shp_keys, variables):
            ret = qdt._get_geometries_with_percentiles_(variable, shp_key, bin_directory, percentile)
            self.assertTrue(len(ret) >= 1, msg=(percentile, shp_key, len(ret)))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
