import unittest
import numpy as np
import itertools
from ocgis.dev.collection import *
import datetime
from ocgis.dev.collection import OcgDimension
from shapely.geometry.point import Point


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
        
        uid = np.arange(1,len(times)+1)
        
        tdim = TemporalDimension(uid,times,bounds=time_bounds)
        
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
        
        uid = np.arange(1,len(times)+1)
        
        tdim = TemporalDimension(uid,times,bounds=time_bounds)

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
        
        uid = np.arange(1,len(times)+1)
        
        tdim = TemporalDimension(uid,times,bounds=time_bounds)
        
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

    def test_SpatialDimension(self):
        y = range(40,45)
        x = range(-90,-85)
        x,y = np.meshgrid(x,y)
        geoms = [Point(ix,iy) for ix,iy in zip(x.flat,y.flat)]
        geoms = np.array(geoms,dtype=object).reshape(5,5)
        np.random.seed(1)
        mask = np.random.random_integers(0,1,geoms.shape)
        gid = np.arange(1,26).reshape(5,5)
        gid = np.ma.array(gid,mask=mask)
        sdim = SpatialDimension(gid,geoms,mask)
        
        masked = sdim.get_masked()
        self.assertTrue(np.all(masked.mask == mask))
        import ipdb;ipdb.set_trace()

if __name__ == "__main__":
    import sys;sys.argv = ['', 'TestCollection.test_SpatialDimension']
    unittest.main()