import unittest
from ocgis.test.base import TestBase
import ocgis
from ocgis.calc.library import QEDDynamicPercentileThreshold
import numpy as np
import itertools
from unittest.case import SkipTest
from ocgis.api.operations import OcgOperations


class TestDynamicPercentiles(TestBase):
    
    def setUp(self):
        raise(SkipTest('dev'))

    def get_file(self,dir_output):
        ## leap year is 1996. time range will be 1995-1997.
        uri = '/usr/local/climate_data/maurer/2010-concatenated/Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
        rd = ocgis.RequestDataset(uri,variable,time_region={'year':[1995,1996,1997]})
        ops = ocgis.OcgOperations(dataset=rd,prefix='subset_1995-1997_Maurer02new_OBS_tasmax_daily.1971-2000',
                                  output_format='nc',dir_output=dir_output)
        ret = ops.execute()
        return(ret)
    
    def get_request_dataset(self,time_region=None):
        uri = '/home/local/WX/ben.koziol/climate_data/snippets/subset_1995-1997_Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
        rd = ocgis.RequestDataset(uri,variable,time_region=time_region)
        return(rd)
    
    def make_file(self):
        dir_output = '/tmp'
        print(self.get_file(dir_output))
        
    def test_get_day_index(self):
        rd = self.get_request_dataset()
        dates = rd.ds.temporal.value
        qdt = QEDDynamicPercentileThreshold()
        di = qdt._get_day_index_(dates)
        self.assertEqual((di['index'] == 366).sum(),1)
        return(di)
        
    def test_is_leap_year(self):
        years = np.array([1995,1996,1997])
        qdt = QEDDynamicPercentileThreshold()
        is_leap = map(qdt._get_is_leap_year_,years)
        self.assertNumpyAll(is_leap,[False,True,False])
        
    def test_get_dynamic_index(self):
        di = self.test_get_day_index()
        qdt = QEDDynamicPercentileThreshold()
        dyidx = map(qdt._get_dynamic_index_,di.flat)
    
    def test_calculate(self):
        ocgis.env.DIR_BIN = '/home/local/WX/ben.koziol/links/ocgis/bin/QED_2013_dynamic_percentiles'
        percentiles = [90,92.5,95,97.5]
        operations = ['gt','gte','lt','lte']
        calc_groupings = [
                          ['month'],
#                          ['month','year'],
#                          ['year']
                          ]
        uris_variables = [['/home/local/WX/ben.koziol/climate_data/maurer/2010-concatenated/Maurer02new_OBS_tasmax_daily.1971-2000.nc','tasmax'],
                          ['/home/local/WX/ben.koziol/climate_data/maurer/2010-concatenated/Maurer02new_OBS_tasmin_daily.1971-2000.nc','tasmin']]
        geoms_select_ugids = [
                              ['qed_city_centroids',None],
                              ['state_boundaries',[39]],
#                              ['us_counties',[2416,1335]]
                              ]
        for tup in itertools.product(percentiles,operations,calc_groupings,uris_variables,geoms_select_ugids):
            print(tup)
            percentile,operation,calc_grouping,uri_variable,geom_select_ugid = tup
            ops = OcgOperations(dataset={'uri':uri_variable[0],'variable':uri_variable[1],'time_region':{'year':[1990],'month':[6,7,8]}},
                geom=geom_select_ugid[0],select_ugid=geom_select_ugid[1],
                calc=[{'func':'qed_dynamic_percentile_threshold','kwds':{'operation':operation,'percentile':percentile},'name':'dp'}],
                calc_grouping=calc_grouping,output_format='numpy')
            ret = ops.execute()
        
    def test_get_geometries_with_percentiles(self):
        bin_directory = '/home/local/WX/ben.koziol/links/project/ocg/bin/QED_2013_dynamic_percentiles'
        qdt = QEDDynamicPercentileThreshold()
        percentiles = [90,92.5,95,97.5]
        shp_keys = ['qed_city_centroids','state_boundaries','us_counties']
        variables = ['tmin','tmax']
        for percentile,shp_key,variable in itertools.product(percentiles,shp_keys,variables):
            ret = qdt._get_geometries_with_percentiles_(variable,shp_key,bin_directory,percentile)
            self.assertTrue(len(ret) >= 1,msg=(percentile,shp_key,len(ret)))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()