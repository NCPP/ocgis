import unittest
import numpy as np
from ocgis.calc import library
from ocgis.api.operations import OcgOperations
from nose.plugins.skip import SkipTest
from datetime import datetime as dt


class Test(unittest.TestCase):
    
    @property
    def tasmax(self):
        cancm4 = {'uri':'/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc','variable':'tasmax'}
        return(cancm4.copy())
    @property
    def rhsmax(self):
        cancm4 = {'uri':'/usr/local/climate_data/CanCM4/rhsmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc','variable':'rhsmax'}
        return(cancm4.copy())
    
    def test_HeatIndex(self):
        ds = [self.tasmax,self.rhsmax]
        calc = {'func':'heat_index','name':'heat_index','kwds':{'tas':'tasmax','rhs':'rhsmax','units':'k'}}
        time_range = [dt(2011,1,1),dt(2011,12,31)]
        ops = OcgOperations(dataset=ds,calc=calc,time_range=time_range,output_format='csv')
        self.assertEqual(ops.calc_grouping,None)
        ret = ops.execute()
        import ipdb;ipdb.set_trace()
    
    def test_element_wise(self):
        raise(SkipTest)
        calc = {'func':'max','name':'max'}
        calc_grouping = None
        ops = OcgOperations(dataset=self.cancm4,calc=calc,calc_grouping=calc_grouping)
        self.assertEqual(ops.calc_grouping,None)
        ret = ops.execute()

    def test_Mean(self):
        agg = True
        weights = None
        values = np.ones((36,2,4,4))
        values = np.ma.array(values,mask=False)
        
        on = np.ones(12,dtype=bool)
        off = np.zeros(12,dtype=bool)
        
        groups = []
        base_groups = [[on,off,off],[off,on,off],[off,off,on]]
        for bg in base_groups:
            groups.append(np.concatenate(bg))
        
        mean = library.Mean(values=values,agg=agg,weights=weights,groups=groups)
        ret = mean.calculate()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()