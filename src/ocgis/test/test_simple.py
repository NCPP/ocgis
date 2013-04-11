import unittest
from ocgis.test.make_test_data import SimpleNc, SimpleMaskNc, SimpleNc360
from ocgis.api.operations import OcgOperations
from ocgis.api.interpreter import OcgInterpreter
import itertools
import numpy as np
import datetime
from ocgis.util.helpers import make_poly
from ocgis import exc, env
import os.path
from ocgis.util.inspect import Inspect
import tempfile
import shutil
from abc import ABCMeta, abstractproperty


class TestBase(unittest.TestCase):
    __metaclass__ = ABCMeta
    
    base_value = None
    return_shp = False
    var = 'foo'
    
    @abstractproperty
    def nc_factory(self): pass
    @abstractproperty
    def fn(self): pass
    
    @classmethod
    def setUpClass(cls):
        env.DIR_OUTPUT = tempfile.mkdtemp(prefix='ocgis_test_',dir=env.DIR_OUTPUT)
        env.OVERWRITE = True
        
    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(env.DIR_OUTPUT)
        finally:
            env.reset()
        
    def setUp(self):
        self.nc_factory().write()
    
    def get_dataset(self,time_range=None,level_range=None):
        uri = os.path.join(env.DIR_OUTPUT,self.fn)
        return({'uri':uri,'variable':self.var,
                'time_range':time_range,'level_range':level_range})
    
    def get_ops(self,kwds={},time_range=None,level_range=None):
        dataset = self.get_dataset(time_range,level_range)
        kwds.update({'dataset':dataset,'output_format':'numpy'})
        ops = OcgOperations(**kwds)
        return(ops)
    
    def get_ret(self,ops=None,kwds={},shp=False,time_range=None,level_range=None):
        if ops is None:
            ops = self.get_ops(kwds,time_range=time_range,level_range=level_range)
        self.ops = ops
        ret = OcgInterpreter(ops).execute()
        
        if shp or self.return_shp:
            kwds2 = kwds.copy()
            kwds2.update({'output_format':'shp'})
            ops2 = OcgOperations(**kwds2)
            OcgInterpreter(ops2).execute()
        
        return(ret)
    
    def make_shp(self):
        ops = OcgOperations(dataset=self.dataset,
                            output_format='shp')
        OcgInterpreter(ops).execute()


class TestSimple(TestBase):
    base_value = np.array([[1.0,1.0,2.0,2.0],
                           [1.0,1.0,2.0,2.0],
                           [3.0,3.0,4.0,4.0],
                           [3.0,3.0,4.0,4.0]])
    nc_factory = SimpleNc
    fn = 'test_simple_spatial_01.nc'
    
    def test_slicing(self):
        ops = self.get_ops(kwds={'slice_row':[0,2],'slice_column':[0,2]})
        import ipdb;ipdb.set_trace()

    def test_return_all(self):
        ret = self.get_ret()
        
        ## confirm size of geometry array
        ref = ret[1].variables[self.var].spatial
        attrs = ['value','uid']
        for attr in attrs:
            self.assertEqual(getattr(ref,attr).shape,(4,4))
        
        ## confirm value array
        ref = ret[1].variables[self.var].value
        self.assertEqual(ref.shape,(61,2,4,4))
        for tidx,lidx in itertools.product(range(0,61),range(0,2)):
            slice = ref[tidx,lidx,:,:]
            idx = self.base_value == slice
            self.assertTrue(np.all(idx))
            
    def test_aggregate(self):
        ret = self.get_ret(kwds={'aggregate':True})
        
        ## test area-weighting
        ref = ret[1].variables[self.var].value
        self.assertTrue(np.all(ref.compressed() == np.ma.average(self.base_value)))
        
        ## test geometry reduction
        ref = ret[1].variables[self.var]
        self.assertEqual(ref.spatial.shape,(1,1))
        
    def test_time_level_subset(self):
        ret = self.get_ret(time_range=[datetime.datetime(2000,3,1),
                                       datetime.datetime(2000,3,31,23)],
                           level_range=[1,1])
        ref = ret[1].variables[self.var].value
        self.assertEqual(ref.shape,(31,1,4,4))
    
    def test_time_level_subset_aggregate(self):
        ret = self.get_ret(kwds={'aggregate':True},
                           time_range=[datetime.datetime(2000,3,1),datetime.datetime(2000,3,31)],
                           level_range=[1,1],)
        ref = ret[1].variables[self.var].value
        self.assertTrue(np.all(ref.compressed() == np.ma.average(self.base_value)))
        ref = ret[1].variables[self.var]
        self.assertEqual(ref.level.value.shape,(1,3))
        
    def test_using_ugid(self):
        ## swap names of id variable in geometry dictionary
        ## intersects
        geom = make_poly((37.5,39.5),(-104.5,-102.5))
        geom = [{'ugid':1,'geom':geom}]
        ret = self.get_ret(kwds={'geom':geom})

    def test_spatial(self):
        ## intersects
        geom = make_poly((37.5,39.5),(-104.5,-102.5))
        geom = [{'ugid':1,'geom':geom}]
        ret = self.get_ret(kwds={'geom':geom})
        ref = ret[1]
        gids = set([6,7,10,11])
        ret_gids = set(ref.variables[self.var].spatial.uid.compressed())
        intersection = gids.intersection(ret_gids)
        self.assertEqual(len(intersection),4)
        self.assertTrue(np.all(ref.variables[self.var].value[0,0,:,:] == np.array([[1.0,2.0],[3.0,4.0]])))
        
        ## intersection
        geom = make_poly((38,39),(-104,-103))
        geom = [{'ugid':1,'geom':geom}]
        ret = self.get_ret(kwds={'geom':geom,'spatial_operation':'clip'})
        self.assertEqual(len(ret[1].variables[self.var].spatial.uid.compressed()),4)
        self.assertEqual(ret[1].variables[self.var].value.shape,(61,2,2,2))
        ref = ret[1].variables[self.var].value
        self.assertTrue(np.all(ref[0,0,:,:] == np.array([[1,2],[3,4]],dtype=float)))
        ## compare areas to intersects returns
        ref = ret[1].variables[self.var]
        intersection_areas = [g.area for g in ref.spatial._value.flat]
        for ii in intersection_areas:
            self.assertAlmostEqual(ii,0.25)
            
        ## intersection + aggregation
        geom = make_poly((38,39),(-104,-103))
        geom = [{'ugid':1,'geom':geom}]
        ret = self.get_ret(kwds={'geom':geom,'spatial_operation':'clip','aggregate':True})
        ref = ret[1]
        self.assertEqual(len(ref.variables[self.var].spatial.uid.flatten()),1)
        self.assertEqual(ref.variables[self.var].spatial._value.flatten()[0].area,1.0)
        self.assertEqual(ref.variables[self.var].value.flatten().mean(),2.5)
        
    def test_empty_intersection(self):
        geom = make_poly((20,25),(-90,-80))
        geom = [{'ugid':1,'geom':geom}]
        
        with self.assertRaises(exc.ExtentError):
            self.get_ret(kwds={'geom':geom})
            
        ret = self.get_ret(kwds={'geom':geom,'allow_empty':True})
        ref = ret[1].variables[self.var].spatial.uid
        self.assertEqual(len(ref),0)
        ref = ret[1]
        self.assertTrue(ref.is_empty)
        
    def test_calc(self):
        calc = [{'func':'mean','name':'my_mean'}]
        group = ['month','year']
        
        ## raw
        ret = self.get_ret(kwds={'calc':calc,'calc_grouping':group})
        ref = ret[1].variables[self.var].calc_value
        for value in ref.itervalues():
            self.assertEqual(value.shape,(2,2,4,4))
        n_foo = ref['n']
        self.assertEqual(n_foo[0,:].mean(),31)
        self.assertEqual(n_foo[1,:].mean(),30)
        
        ## aggregated
        for calc_raw in [True,False]:
            ret = self.get_ret(kwds={'calc':calc,'calc_grouping':group,
                                     'aggregate':True,'calc_raw':calc_raw})
            ref = ret[1].variables[self.var].calc_value
            self.assertEqual(ref['n'].shape,(2,2,1,1))
            self.assertEqual(ref['my_mean'].shape,(2,2,1,1))
            self.assertEqual(ref['my_mean'].flatten().mean(),2.5)
            
    def test_inspect(self):
        uri = self.get_dataset()['uri']
        for variable in [self.get_dataset()['variable'],None]:
            ip = Inspect(uri,variable=variable)
            ret = ip.__repr__()
            self.assertTrue(len(ret) > 100)
            
    def test_nc_conversion(self):
        ops = OcgOperations(dataset=self.get_dataset(),output_format='nc')
        ret = self.get_ret(ops)
        
    def test_shp_conversion(self):
        calc = [
                None,
                [{'func':'mean','name':'my_mean'}]
                ]
        group = ['month','year']
        for c in calc:
            ops = OcgOperations(dataset=self.get_dataset(),
                                output_format='shp',
                                calc_grouping=group,
                                calc=c)
            ret = self.get_ret(ops)
        
    def test_csv_conversion(self):
        ops = OcgOperations(dataset=self.get_dataset(),output_format='csv')
        ret = self.get_ret(ops)
        
    def test_meta_conversion(self):
        ops = OcgOperations(dataset=self.get_dataset(),output_format='meta')
        ret = self.get_ret(ops)
            
    def test_keyed_conversion(self):
        calc = [
                None,
                [{'func':'mean','name':'my_mean'}]
                ]
        group = ['month','year']
        for c in calc:
            ops = OcgOperations(dataset=self.get_dataset(),
                                output_format='keyed',
                                calc=c,
                                calc_grouping=group)
            ret = self.get_ret(ops)
        
    def test_shpidx_conversion(self):
        ops = OcgOperations(dataset=self.get_dataset(),output_format='shpidx')
        ret = self.get_ret(ops)


class TestSimpleMask(TestBase):
    base_value = None
    nc_factory = SimpleMaskNc
    fn = 'test_simple_mask_spatial_01.nc'
        
    def test_spatial(self):
        self.return_shp = False
        ret = self.get_ret()
        ref = ret[1].variables[self.var].value.mask
        cmp = np.array([[True,False,False,False],
                        [False,False,False,True],
                        [False,False,False,False],
                        [True,True,False,False]])
        for tidx,lidx in itertools.product(range(0,ref.shape[0]),range(ref.shape[1])):
            self.assertTrue(np.all(cmp == ref[tidx,lidx,:]))
            
        ## aggregation
        ret = self.get_ret(kwds={'aggregate':True})
        ref = ret[1].variables[self.var]
        self.assertAlmostEqual(ref.value.mean(),2.58333333333,5)
        self.assertEqual(ref.spatial.uid.shape,(1,1))
    
    def test_empty_mask(self):
        geom = make_poly((37.762,38.222),(-102.281,-101.754))
        geom = [{'ugid':1,'geom':geom}]
        with self.assertRaises(exc.MaskedDataError):
            ret = self.get_ret(kwds={'geom':geom})
        ret = self.get_ret(kwds={'geom':geom,'allow_empty':True})
        
        
class TestSimple360(TestBase):
#    return_shp = True
    fn = 'test_simple_360_01.nc'
    nc_factory = SimpleNc360
        
    def test_wrap(self):
        
        def _get_longs_(geom):
            ret = np.array([g.centroid.x for g in geom.flat])
            return(ret)
        
        ret = self.get_ret(kwds={'vector_wrap':False})
        longs_unwrap = _get_longs_(ret[1].variables[self.var].spatial._value)
        self.assertTrue(np.all(longs_unwrap > 180))
        
        ret = self.get_ret(kwds={'vector_wrap':True})
        longs_wrap = _get_longs_(ret[1].variables[self.var].spatial._value)
        self.assertTrue(np.all(np.array(longs_wrap) < 180))
        
        self.assertTrue(np.all(longs_unwrap-360 == longs_wrap))
        
    def test_spatial(self):
        geom = make_poly((38,39),(-93,-92))
        geom = [{'ugid':1,'geom':geom}]
        
        for abstraction in ['polygon','point']:
            ret = self.get_ret(kwds={'geom':geom,'abstraction':abstraction})
            self.assertEqual(len(ret[1].variables[self.var].spatial.uid.compressed()),4)
            
            self.get_ret(kwds={'vector_wrap':False})
            ret = self.get_ret(kwds={'geom':geom,'vector_wrap':False,'abstraction':abstraction})
            self.assertEqual(len(ret[1].variables[self.var].spatial.uid.compressed()),4)


if __name__ == "__main__":
#    import sys;sys.argv = ['', 'TestSimple.test_time_level_subset']
    unittest.main()
