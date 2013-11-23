#from ocgis.interface.shp import ShpDataset
import numpy as np
from ocgis.util.helpers import format_bool, iter_array, validate_time_subset,\
    get_formatted_slice, get_is_date_between
import itertools
from ocgis.test.base import TestBase
#from ocgis.util.spatial.wrap import Wrapper
from datetime import datetime as dt


class TestHelpers(TestBase):
    
    def test_get_is_date_between(self):
        lower = dt(1971,1,1)
        upper = dt(2000,2,1)
        self.assertFalse(get_is_date_between(lower,upper,month=6))
        self.assertFalse(get_is_date_between(lower,upper,month=2))
        self.assertTrue(get_is_date_between(lower,upper,month=1))
        
        self.assertFalse(get_is_date_between(lower,upper,year=1968))
        self.assertTrue(get_is_date_between(lower,upper,year=1995))
        
        lower = dt(2013, 1, 1, 0, 0)
        upper = dt(2013, 1, 2, 0, 0)
        self.assertTrue(get_is_date_between(lower,upper,year=2013))
            
    def test_get_formatted_slc(self):
        ret = get_formatted_slice(slice(None,None,None),10)
        self.assertEqual(ret,[slice(None,None,None)]*10)
        
        ret = get_formatted_slice(0,1)
        self.assertEqual(slice(0,1),ret)
        with self.assertRaises(IndexError):
            get_formatted_slice(slice(0,1),2)
            
        ret = get_formatted_slice((slice(0,1),0),2)
        self.assertEqual(ret,[slice(0,1,None),slice(0,1,None)])
        
        ret = get_formatted_slice([(1,2,3),slice(None)],2)
        self.assertNumpyAll(ret[0],np.arange(1,4))
        self.assertEqual(ret[1],slice(None))
        self.assertEqual(len(ret),2)
        
        ret = get_formatted_slice((1,2),1)
        self.assertNumpyAll(ret,np.array([1,2]))
        
        ret = get_formatted_slice((1,),1)
        self.assertEqual(ret,slice(1))
    
    def test_validate_time_subset(self):
        time_range = [dt(2000,1,1),dt(2001,1,1)]
        self.assertTrue(validate_time_subset(time_range,{'year':[2000,2001]}))
        self.assertFalse(validate_time_subset(time_range,{'year':[2000,2001,2002]}))
        self.assertTrue(validate_time_subset(time_range,{'month':[6,7,8]}))
        self.assertTrue(validate_time_subset(time_range,{'month':[6,7,8],'year':[2000]}))
        self.assertFalse(validate_time_subset(time_range,{'month':[6,7,8],'year':[2008]}))
        self.assertFalse(validate_time_subset([dt(2000,1,1),dt(2000,2,1)],{'month':[6,7,8],'year':[2008]}))
        self.assertTrue(validate_time_subset([dt(2000,1,1),dt(2000,2,1)],None))

    def test_iter_array(self):
        arrays = [
                  1,
                  [[1,2],[1,2]],
                  np.array([1,2,3]),
                  np.array(1),
                  np.ma.array([1,2,3],mask=False),
                  np.ma.array([[1,2],[3,4]],mask=[[True,False],[False,True]]),
                  np.ma.array([[1,2],[3,4]],mask=True),
                 ]
        _flag1 = [
                  True,
                  False
                  ]
        _flag2 = [
                  True,
                  False
                  ]
        
        for arr,flag1,flag2 in itertools.product(arrays,_flag1,_flag2):
#            try:
            for ret in iter_array(arr,use_mask=flag1,return_value=flag2):
                pass
#            except Exception as e:
#                print(arr,flag1,flag2)
#                import ipdb;ipdb.set_trace()

        arr = np.ma.array([1,2,3],mask=True)
        ret = list(iter_array(arr))
        self.assertEqual(len(ret),0)
        arr = np.ma.array([1,2,3],mask=False)
        ret = list(iter_array(arr))
        self.assertEqual(len(ret),3)
        
        values = np.random.rand(2,2,4,4)
        mask = np.random.random_integers(0,1,values.shape)
        values = np.ma.array(values,mask=mask)
        for idx in iter_array(values):
            self.assertFalse(values.mask[idx])
        self.assertEqual(len(list(iter_array(values,use_mask=True))),len(values.compressed()))
        self.assertEqual(len(list(iter_array(values,use_mask=False))),len(values.data.flatten()))
        
    def test_format_bool(self):
        mmap = {0:False,1:True,'t':True,'True':True,'f':False,'False':False}
        for key,value in mmap.iteritems():
            ret = format_bool(key)
            self.assertEqual(ret,value)

#class TestSpatial(TestBase):
#    axes = [-10.0,-5.0,0.0,5.0,10]
#
#    def test_unwrap(self):
#        sd = ShpDataset('state_boundaries')        
#        for axis in self.axes:
#            w = Wrapper(axis=axis)
#            for geom in sd.spatial.geom:
#                new_geom = w.unwrap(geom)
#                bounds = np.array(new_geom.bounds)
#                self.assertFalse(np.any(bounds < axis))
#                
#    def test_wrap(self):
#        sd = ShpDataset('state_boundaries')
#        for axis in self.axes:
#            w = Wrapper(axis=axis)
#            unwrapped = [w.unwrap(geom) for geom in sd.spatial.geom]
#            for idx,unwrapped_geom in enumerate(unwrapped):
#                new_geom = w.wrap(unwrapped_geom)
#                self.assertFalse(unwrapped_geom.equals(new_geom))
#                self.assertTrue(sd.spatial.geom[idx].almost_equals(new_geom))
