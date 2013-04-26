from ocgis.interface.shp import ShpDataset
import numpy as np
from ocgis.util.helpers import format_bool, iter_array
import itertools
from ocgis.test.base import TestBase
from ocgis.util.spatial.wrap import Wrapper


class TestHelpers(TestBase):

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
        
        sd = ShpDataset('state_boundaries')
        ret = list(iter_array(sd.spatial.geom))
        self.assertTrue(len(ret),51)
        
    def test_format_bool(self):
        mmap = {0:False,1:True,'t':True,'True':True,'f':False,'False':False}
        for key,value in mmap.iteritems():
            ret = format_bool(key)
            self.assertEqual(ret,value)

class TestSpatial(TestBase):
    axes = [-10.0,-5.0,0.0,5.0,10]

    def test_unwrap(self):
        sd = ShpDataset('state_boundaries')        
        for axis in self.axes:
            w = Wrapper(axis=axis)
            for geom in sd.spatial.geom:
                new_geom = w.unwrap(geom)
                bounds = np.array(new_geom.bounds)
                self.assertFalse(np.any(bounds < axis))
                
    def test_wrap(self):
        sd = ShpDataset('state_boundaries')
        for axis in self.axes:
            w = Wrapper(axis=axis)
            unwrapped = [w.unwrap(geom) for geom in sd.spatial.geom]
            for idx,unwrapped_geom in enumerate(unwrapped):
                new_geom = w.wrap(unwrapped_geom)
                self.assertFalse(unwrapped_geom.equals(new_geom))
                self.assertTrue(sd.spatial.geom[idx].almost_equals(new_geom))
