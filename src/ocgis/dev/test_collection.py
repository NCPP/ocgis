import unittest
import numpy as np
import itertools
from ocgis.dev.collection import *
import datetime
from shapely.geometry.point import Point
from ocgis.dev.collection.dimension.dimension import OcgDimension
from ocgis.dev.collection.collection import Identifier
from ocgis.dev.collection.iterators import KeyedIterator, MeltedIterator
import time


class TestCollection(unittest.TestCase):

    def test_OcgDimension(self):
        bounds = [None,
                  np.array([[0,100],[100,200]])]
        add_bounds = [True,False]
        value = np.array([50,150])
        
        for bound,add_bounds in itertools.product(bounds,add_bounds):
            dim = OcgDimension(value,bounds=bound)
            for row in dim.iter_rows(add_bounds=add_bounds):
                self.assertTrue(OcgDimension._value_name in row)
                if add_bounds and bound is not None:
                    self.assertTrue('bnds' in row)
                else:
                    self.assertFalse('bnds' in row)
                    
    def test_OcgIdentifier(self):
        oid = Identifier()
        oid.add(55)
        self.assertEqual(oid,{55:1})
        oid.add(55)
        self.assertEqual(oid,{55:1})
        oid.add(56)
        self.assertEqual(oid,{55:1,56:2})
        
    def get_TemporalDimension(self,add_bounds=True):
        start = datetime.datetime(2000,1,1,12)
        end = datetime.datetime(2001,12,31,12)
        delta = datetime.timedelta(1)
        times = []
        check = start
        while check <= end:
            times.append(check)
            check += delta
        times = np.array(times)
        
        if add_bounds:
            time_bounds = []
            delta = datetime.timedelta(hours=12)
            for t in times.flat:
                time_bounds.append([t-delta,t+delta])
            time_bounds = np.array(time_bounds)
        else:
            time_bounds = None
        
        tdim = TemporalDimension(times,bounds=time_bounds)
        
        return(tdim)
    
    def get_hourly_TemporalDimension(self,add_bounds=True):
        start = datetime.datetime(2000,1,1,0,30)
        end = datetime.datetime(2001,12,31,23,30)
        delta = datetime.timedelta(hours=1)
        times = []
        check = start
        while check <= end:
            times.append(check)
            check += delta
        times = np.array(times)
        
        if add_bounds:
            time_bounds = []
            delta = datetime.timedelta(minutes=30)
            for t in times.flat:
                time_bounds.append([t-delta,t+delta])
            time_bounds = np.array(time_bounds)
        else:
            time_bounds = None
        
        tdim = TemporalDimension(times,bounds=time_bounds)

        return(tdim)
    
    def get_monthly_TemporalDimension(self,add_bounds=True):
        years = [2000,2001]
        months = range(1,13)
        times = np.empty(len(years)*len(months),dtype=object)
        time_bounds = np.empty((len(times),2),dtype=object)
        for idx,(year,month) in enumerate(itertools.product(years,months)):
            times[idx] = datetime.datetime(year,month,15)
            time_bounds[idx,0] = datetime.datetime(year,month,1)
            try:
                time_bounds[idx,1] = datetime.datetime(year,month+1,1)
            except ValueError:
                time_bounds[idx,1] = datetime.datetime(year+1,1,1)
        
        if not add_bounds:
            time_bounds = None
            
        tdim = TemporalDimension(times,bounds=time_bounds)
        
        return(tdim)
    
    def test_TemporalGroupDimension(self):
        
        tdims = [
#                 self.get_hourly_TemporalDimension,
                 self.get_TemporalDimension,
                 self.get_monthly_TemporalDimension,
                 ]
        perms = ['year','month','day','hour','minute','second','microsecond']
#        perms = [['day','hour']]
        add_bounds_opts = [True,False]
#        add_bounds_opts = [True]
        for ii,add_bounds,tdim_func in itertools.product(range(1,4),add_bounds_opts,tdims):
            for perm in itertools.permutations(perms,ii):
                tdim = tdim_func(add_bounds=add_bounds)
                try:
                    tgdim = tdim.group(perm)
                except TypeError:
                    tgdim = tdim.group(*perm)
                for row in tgdim.iter_rows():
                    if np.random.rand() <= -0.01:
                        print (perm,add_bounds)
                        print row
                        import ipdb;ipdb.set_trace()
                    else:
                        continue
#                try:
                self.assertEqual(np.sum([dgrp.sum() for dgrp in tgdim.dgroups]),len(tdim.value))
#                except:
#                    import ipdb;ipdb.set_trace()

    def get_SpatialDimension(self):
        y = range(40,45)
        x = range(-90,-85)
        x,y = np.meshgrid(x,y)
        geoms = [Point(ix,iy) for ix,iy in zip(x.flat,y.flat)]
        geoms = np.array(geoms,dtype=object).reshape(5,5)
        np.random.seed(1)
        self._mask = np.random.random_integers(0,1,geoms.shape)
#        gid = np.arange(1,26).reshape(5,5)
#        gid = np.ma.array(gid,mask=self._mask)
        sdim = SpatialDimension(geoms,self._mask)
        return(sdim)
    
    def test_SpatialDimension(self):
        sdim = self.get_SpatialDimension()
        masked = sdim.get_masked()
        self.assertTrue(np.all(masked.mask == self._mask))
        
        for row in sdim.iter_rows():
            continue

    def get_LevelDimension(self,add_bounds=True):
        values = np.array([50,150])
        if add_bounds:
            bounds = np.array([[0,100],[100,200]])
        else:
            bounds = None
        ldim = LevelDimension(values,bounds)
        return(ldim)
    
    def get_OcgVariable(self,add_level=True,add_bounds=True,name='foo'):
        temporal = self.get_TemporalDimension(add_bounds)
        spatial = self.get_SpatialDimension()
        if add_level:
            level = self.get_LevelDimension(add_bounds)
            level_len = len(level.value)
        else:
            level = None
            level_len = 1
        
        value = np.random.rand(len(temporal.value),
                               level_len,
                               spatial.value.shape[0],
                               spatial.value.shape[1])
        mask = np.empty(value.shape,dtype=bool)
        for idx in range(mask.shape[0]):
            mask[idx,:,:] = spatial.value_mask
        value = np.ma.array(value,mask=mask)
        
        var = OcgVariable(name,value,temporal,spatial,level)
        return(var)
        
    def test_OcgCollection(self):
        _add_bounds = [
                       True,
                       False
                       ]
        _add_level = [True,False]
#        _group = [None,['month']]
        _group = [['month']]
        args = (_add_bounds,_add_level,_group)
        
        for add_bounds,add_level,group in itertools.product(*args):
#            print add_bounds,add_level
            coll = OcgCollection()
            var1 = self.get_OcgVariable(add_level=add_level,add_bounds=add_bounds)
            coll.add_variable(var1)
            lens_original = [len(getattr(coll,attr)) for attr in ['tid','lid','gid','tbid','lbid']]
            var2 = self.get_OcgVariable(add_level=add_level,add_bounds=add_bounds,name='foo2')
            coll.add_variable(var2)
            lens_new = [len(getattr(coll,attr)) for attr in ['tid','lid','gid','tbid','lbid']]
            self.assertEqual(lens_original,lens_new)
            
            if group is not None:
                for name in ['my_mean','my_median']:
                    for var in [var1,var2]:
                        tgdim = var.group(group)
                        new_values = np.random.rand(tgdim.value.shape[0],
                                                    var.value.shape[1],
                                                    var.spatial.value.shape[0],
                                                    var.spatial.value.shape[1])
                        mask = np.empty(new_values.shape,dtype=bool)
                        mask[:] = var.value.mask[0,0,:]
                        new_values = np.ma.array(new_values,mask=mask)
                        coll.add_calculation(var,name,new_values,tgdim)
            
            if len(coll.lid) == 0:
                m = 1
            else:
                m = len(coll.lid)
            iter_len = len(coll.tid)*len(coll.gid)*m*2
            it = MeltedIterator(coll).iter_rows()
            for ii,row in enumerate(it,start=1):
                if add_level:
                    self.assertTrue(row['level'] is not None)
                else:
                    self.assertTrue(row['level'] is None)
                self.assertEqual(len(row),12)
                if group is None:
                    self.assertEqual(row.keys(),['lid','ugid','vid','level','did','var_name','uri','value','gid','geom','time','tid'])
                else:
                    self.assertTrue(all([v in row.keys() for v in ['cid','calc_name','calc_value']]))
                    self.assertFalse('value' in row.keys())
            self.assertEqual(iter_len,ii)
#                import ipdb;ipdb.set_trace()
                
#            import ipdb;ipdb.set_trace()

if __name__ == "__main__":
#    import sys;sys.argv = ['', 'TestCollection.test_OcgVariable']
    unittest.main()