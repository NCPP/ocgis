import unittest
from ocgis.test.base import TestBase
import ocgis
from ocgis.calc.library import QEDDynamicPercentileThreshold
import numpy as np


class TestDynamicPercentiles(TestBase):

    def get_file(self,dir_output):
        ## leap year is 1996. time range will be 1995-1997.
        uri = '/usr/local/climate_data/maurer/2010-concatenated/Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
        rd = ocgis.RequestDataset(uri,variable,time_region={'year':[1995,1996,1997]})
        ops = ocgis.OcgOperations(dataset=rd,prefix='subset_1995-1997_Maurer02new_OBS_tasmax_daily.1971-2000',
                                  output_format='nc',dir_output=dir_output)
        ret = ops.execute()
        return(ret)
    
    def make_file(self):
        dir_output = '/tmp'
        print(self.get_file(dir_output))
        
    def test_get_day_index(self):
        uri = '/home/local/WX/ben.koziol/climate_data/snippets/subset_1995-1997_Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
        rd = ocgis.RequestDataset(uri,variable)
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
        import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()