import unittest
from ocgis.test.make_test_data import make_simple
from ocgis.api.operations import OcgOperations
from ocgis.api.interpreter import OcgInterpreter
import itertools
import numpy as np
import datetime
from ocgis.util.helpers import make_poly
from ocgis import exc


class TestSimple(unittest.TestCase):
    uri = '/tmp/test_simple_spatial_01.nc'
    var = 'foo'
    base_value = cmp = np.array([[1.0,1.0,2.0,2.0],
                                 [1.0,1.0,2.0,2.0],
                                 [3.0,3.0,4.0,4.0],
                                 [3.0,3.0,4.0,4.0]])
    
    @property
    def dataset(self):
        return({'uri':self.uri,'variable':self.var})
    
    def get_ops(self,kwds={}):
        kwds.update({'dataset':self.dataset,'output_format':'numpy'})
        ops = OcgOperations(**kwds)
        return(ops)
    
    def get_ret(self,ops=None,kwds={},shp=False):
        if ops is None:
            ops = self.get_ops(kwds)
        ret = OcgInterpreter(ops).execute()
        
        if shp:
            kwds2 = kwds.copy()
            kwds2.update({'output_format':'shp'})
            ops2 = OcgOperations(**kwds2)
            OcgInterpreter(ops2).execute()
        
        return(ret)
    
    def make_shp(self):
        ops = OcgOperations(dataset=self.dataset,
                            output_format='shp')
        OcgInterpreter(ops).execute()
    
    def setUp(self):
#        self.make_shp()
        make_simple()

    def test_return_all(self):
        ops = self.get_ops()
        ret = self.get_ret(ops)
        
        ## confirm size of geometry array
        ref = ret[1]
        attrs = ['gid','geom','geom_masked']
        for attr in attrs:
            self.assertEqual(getattr(ref,attr).shape,(4,4))
        
        ## confirm value array
        ref = ret[1].variables[self.var].raw_value
        self.assertEqual(ref.shape,(61,2,4,4))
        for tidx,lidx in itertools.product(range(0,61),range(0,2)):
            slice = ref[tidx,lidx,:,:]
            idx = self.base_value == slice
            self.assertTrue(np.all(idx))
            
    def test_aggregate(self):
        ret = self.get_ret(kwds={'aggregate':True})
        
        ## test area-weighting
        ref = ret[1].variables[self.var].agg_value
        self.assertTrue(np.all(ref.compressed() == np.ma.average(self.base_value)))
        
        ## test geometry reduction
        ref = ret[1]
        self.assertEqual(len(ref.gid),1)
        
    def test_time_level_subset(self):
        ret = self.get_ret(kwds={'time_range':[datetime.datetime(2000,3,1),
                                               datetime.datetime(2000,3,31)],
                                 'level_range':1})
        ref = ret[1].variables[self.var].raw_value
        self.assertEqual(ref.shape,(31,1,4,4))
    
    def test_time_level_subset_aggregate(self):
        ret = self.get_ret(kwds={'time_range':[datetime.datetime(2000,3,1),
                                               datetime.datetime(2000,3,31)],
                                 'level_range':1,
                                 'aggregate':True})
        ref = ret[1].variables[self.var].agg_value
        self.assertTrue(np.all(ref.compressed() == np.ma.average(self.base_value)))

    def test_spatial(self):
        ## intersects
        geom = make_poly((37.5,39.5),(-104.5,-102.5))
        geom = {'id':1,'geom':geom}
        ret = self.get_ret(kwds={'geom':geom})
        ref = ret[1]
        gids = set([6,7,10,11])
        ret_gids = set(ref.gid.compressed())
        intersection = gids.intersection(ret_gids)
        self.assertEqual(len(intersection),4)
        self.assertTrue(np.all(ref.variables['foo'].raw_value[0,0,:,:] == np.array([[1.0,2.0],[3.0,4.0]])))
        
        ## intersection
        geom = make_poly((38,39),(-104,-103))
        geom = {'id':1,'geom':geom}
        ret = self.get_ret(kwds={'geom':geom,'spatial_operation':'clip'})
        self.assertEqual(len(ret[1].gid.compressed()),4)
        self.assertEqual(ret[1].variables[self.var].raw_value.shape,(61,2,2,2))
        ref = ret[1].variables[self.var].raw_value
        self.assertTrue(np.all(ref[0,0,:,:] == np.array([[1,2],[3,4]],dtype=float)))
        ## compare areas to intersects returns
        ref = ret[1]
        intersection_areas = [g.area for g in ref.geom.flat]
        for ii in intersection_areas:
            self.assertEqual(ii,0.25)
            
        ## intersection + aggregation
        geom = make_poly((38,39),(-104,-103))
        geom = {'id':1,'geom':geom}
        ret = self.get_ret(kwds={'geom':geom,'spatial_operation':'clip','aggregate':True})
        ref = ret[1]
        self.assertEqual(len(ref.gid.flatten()),1)
        self.assertEqual(ref.geom.flatten()[0].area,1.0)
        self.assertEqual(ref.variables[self.var].agg_value.flatten().mean(),2.5)
        
    def test_empty_intersection(self):
        geom = make_poly((20,25),(-90,-80))
        geom = {'id':1,'geom':geom}
        with self.assertRaises(exc.ExtentError):
            self.get_ret(kwds={'geom':geom})
        ret = self.get_ret(kwds={'geom':geom,'allow_empty':True})
        self.assertEqual(len(ret[1].gid),0)
        
    def test_calc(self):
        calc = {'func':'mean','name':'my_mean'}


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
