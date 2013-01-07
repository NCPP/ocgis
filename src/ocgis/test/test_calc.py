import unittest
import numpy as np
from ocgis.calc import library
from ocgis.api.operations import OcgOperations
from nose.plugins.skip import SkipTest
from datetime import datetime as dt
from ocgis.api.dataset.collection.iterators import MeltedIterator, KeyedIterator


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
        ops = OcgOperations(dataset=ds,calc=calc,time_range=time_range)
        self.assertEqual(ops.calc_grouping,None)
        ret = ops.execute()
        ref = ret[1]
        self.assertEqual(ref.variables.keys(),['tasmax','rhsmax','heat_index'])
        hi = ref.variables['heat_index']
        self.assertEqual(hi.value.shape,(364,1,64,128))
        it = MeltedIterator(ret[1],mode='calc')
        for ii,row in enumerate(it.iter_rows()):
            if ii == 0:
                self.assertEqual(row['value'],None)
            if ii < 1000:
                for key in ['vid','var_name','did','uri']:
                    self.assertEqual(row[key],None)
            else:
                break
        
        ops = OcgOperations(dataset=ds,calc=calc,output_format='numpy',snippet=True)
        ret = ops.execute()
        
    def test_HeatIndex_keyed_output(self):
        ds = [self.tasmax,self.rhsmax]
        calc = {'func':'heat_index','name':'heat_index','kwds':{'tas':'tasmax','rhs':'rhsmax','units':'k'}}
#        time_range = [dt(2011,1,1),dt(2011,12,31)]
        ops = OcgOperations(dataset=ds,calc=calc,snippet=True,output_format='keyed')
        self.assertEqual(ops.calc_grouping,None)
        ret = ops.execute()
#        it = KeyedIterator(ret[1],mode='calc')
#        for row in it.iter_rows(ret[1]):
#            import ipdb;ipdb.set_trace()

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