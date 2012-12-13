import unittest
import numpy as np
import itertools
from ocgis.dev.collection import *
import datetime


class TestCollection(unittest.TestCase):

    def test_OcgDimension(self):
        bounds = [None,
                  np.array([[0,100],[100,200]])]
        add_bounds = [True,False]
        value = np.array([50,150])
        uid = np.array([1,2])
        
        for bound,add_bounds in itertools.product(bounds,add_bounds):
            dim = OcgDimension('lid',uid,'level',value,bounds=bound)
            self.assertEqual(dim.headers,{'bnds':{0:'bnds0_level',
                                                  1:'bnds1_level'},
                                          'uid':'lid','value':'level'})
            for row in dim.iter_rows(add_bounds=add_bounds):
                if add_bounds and bound is not None:
                    self.assertTrue('bnds' in row)
                else:
                    self.assertFalse('bnds' in row)
                    
    def test_OcgIdentifier(self):
        oid = OcgIdentifier()
        oid.add(55)
        self.assertEqual(oid,{55:1})
        oid.add(55)
        self.assertEqual(oid,{55:1})
        oid.add(56)
        self.assertEqual(oid,{55:1,56:2})
        
        
    def get_TemporalDimension(self):
        start = datetime.datetime(2000,1,1,12)
        end = datetime.datetime(2001,12,31,12)
        delta = datetime.timedelta(1)
        times = []
        check = start
        while check <= end:
            times.append(check)
            check += delta
            
        times = np.array(times)
        time_bounds = []
        delta = datetime.timedelta(hours=12)
        for t in times.flat:
            time_bounds.append([t-delta,t+delta])
        time_bounds = np.array(time_bounds)
        
        uid = np.arange(1,len(times)+1)
        
        tdim = TemporalDimension(uid,times,bounds=time_bounds)
        
        return(tdim)
    
    def test_TemporalGroupDimension(self):
        tdim = self.get_TemporalDimension()
        
#        args = ('month',2)
        args = [['year']]
        tgdim = tdim.group(*args)
        for dgrp in tgdim.dgroups:
            print dgrp.sum()
        for row in tgdim.iter_rows():
            print row
        self.assertEqual(len(tgdim.dgroups),2)
        self.assertEqual(np.sum([dgrp.sum() for dgrp in tgdim.dgroups]),len(tdim.value))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()