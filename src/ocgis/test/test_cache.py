import unittest
from ocgis.test.base import TestBase
import datetime
from tempfile import mkstemp, mkdtemp
from ocgis.util.cache import CachedObject, CacheCabinet
import numpy as np
import os.path
import webbrowser
import subprocess
import time
from ocgis import env
from ocgis.api.operations import OcgOperations
import itertools
from unittest.case import SkipTest
from ocgis.api.request import RequestDataset


class Test(TestBase):
    
    def get_modification_datetime(self,filename):
        t = os.path.getmtime(filename)
        return(datetime.datetime.fromtimestamp(t))
    
    def test_CachedObject(self):
        start = datetime.datetime(2000,1,1)
        delta = datetime.timedelta(days=1)
        dates = [start]
        for ii in range(10):
            dates.append(dates[-1]+delta)
        path = mkstemp(dir=self._test_dir)[1]
        co = CachedObject(path)
        co.write(dates)
        new_dates = co.get()
        self.assertTrue(np.all(new_dates == dates))
        
    def test_CachedObject_real_data(self):
        rd = self.test_data.get_rd('cancm4_tas')
        path = mkstemp(dir=self._test_dir)[1]
        co = CachedObject(path)
        temporal = rd.ds.temporal
        temporal.dataset = None
        co.write(temporal)
        new_temporal = co.get()
        self.assertTrue(np.all(temporal.value == new_temporal.value))
        self.assertTrue(np.all(temporal.bounds == new_temporal.bounds))
        
    def test_CacheCabinet(self):
        cc = CacheCabinet(self._test_dir)
        self.assertTrue(os.path.exists(cc._cfg_path))
        
        cc.add('foo','something',mtime=None)
        foo_path = cc.get_path('foo')
        self.assertTrue(os.path.exists(foo_path))
        omtime = self.get_modification_datetime(foo_path)
        time.sleep(0.25)

        cc.add('foo','something',mtime=time.time())
        nomtime = self.get_modification_datetime(foo_path)
        self.assertNotEqual(omtime,nomtime)
        
        test_mtime = time.time()
        cc.add('foo2','something',mtime=test_mtime)
        cc.add('foo2','something',mtime=test_mtime)
        
    def test_CacheCabinet_limit(self):
        rd = self.test_data.get_rd('cancm4_tas')
        cc = CacheCabinet(self._test_dir)
        for ii in range(10):
            cc.add(str(ii),rd.ds.temporal.value)
        size_mb = cc.size_megabytes
        self.assertAlmostEqual(size_mb,2.41914668884)
        cc.limit = 1
        cc.add('too much',rd.ds.temporal.value)
        contents1 = os.listdir(cc.path)
        self.assertEqual(len(contents1),5)
        cc.add('way too much',rd.ds.temporal.value)
        contents2 = os.listdir(cc.path)
        self.assertEqual(len(contents2),6)
        cc.add('more',rd.ds.temporal.value)
        contents3 = os.listdir(cc.path)
        self.assertEqual(len(contents3),5)
        
    def test_CacheCabinet_array(self):
        cc = CacheCabinet(self._test_dir)
        self.assertEqual(0.0,cc.size_megabytes)
        rd = self.test_data.get_rd('cancm4_tas')
        for ii in range(3):
            cc.add(str(ii),rd.ds.temporal.value)
        arr = cc.array
        self.assertEqual(arr.shape,(3,))
        
    def test_CacheCabinet_get(self):
        cc = CacheCabinet(self._test_dir)
        rd = self.test_data.get_rd('cancm4_tas')
        rd.ds.temporal.dataset = None
        cc.add('1',rd.ds.temporal)
        new_temporal = cc.get('1')
        for attr in ['bounds','value']:
            test_value = getattr(rd.ds.temporal,attr)
            new_test_value = getattr(new_temporal,attr)
            self.assertTrue(np.all(test_value == new_test_value))
            
    def test_real_data(self):
        
#        key = 'narccap_pr_wrfg_ncep'
        key = 'cancm4_tas'
        n = 1
        
        times1 = []
        for ii in range(n):
#            print('first',ii)
            t1 = time.time()
            rd1 = self.test_data.get_rd(key)
            temporal1 = rd1.ds.temporal
            t2 = time.time()
            times1.append(t2-t1)
#        print('times1',np.mean(times1),np.std(times1))
        
        
        env.USE_CACHING = True
        env.DIR_CACHE = mkdtemp(dir=self._test_dir)
        times2 = []
        for ii in range(n+1):
#            print('second',ii)
            t1 = time.time()
            rd1 = self.test_data.get_rd(key)
            temporal2 = rd1.ds.temporal
            t2 = time.time()
            if ii != 0:
                times2.append(t2-t1)
#        print('times2',np.mean(times2),np.std(times2))

        self.assertTrue(np.all(temporal1.value == temporal2.value))
        
    def test_real_data_OcgOperations(self):
        raise(SkipTest('dev'))
        env.DIR_DATA = '/usr/local/climate_data/maurer/2010-concatenated'
        uri = 'Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
        env.DIR_CACHE = mkdtemp(dir=self._test_dir)
        test_temporal = None
        time_no_cache = []
        time_cache = []
        _snippet = [
#                    True,
                    False
                    ]
        _use_caching = [
                        True,
                        False
                        ]
        _time_range = [
                       None,
#                       [datetime.datetime(2001,1,1),datetime.datetime(2002,12,31)]
                       ]
        _calc = [
#                 None,
                 [{'func':'mean','name':'mean'}]
                 ]
        calc_grouping = ['month']
        for snippet,use_caching,time_range,calc in itertools.product(_snippet,_use_caching,_time_range,_calc):
            for ii in range(33):
                print(ii)
                t1 = time.time()
                env.USE_CACHING = use_caching
                rd = RequestDataset(uri,variable,time_range=time_range)
                ops = OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[25],
                                    snippet=snippet,calc=calc,calc_grouping=calc_grouping)
                ret = ops.execute()
                ref = ret[25].variables['tasmax'].temporal
                if ii == 0:
                    test_temporal = ref
                else:
                    self.assertTrue(np.all(test_temporal.value == ref.value))
                t2 = time.time() - t1
                if snippet is False:
                    if use_caching and ii != 0:
                        time_cache.append(t2)
                    else:
                        time_no_cache.append(t2)
        print()
        print('time no cache',np.mean(time_no_cache),np.std(time_no_cache))
        print('time cache',np.mean(time_cache),np.std(time_cache))

    def test_maurer_concatenated(self):
        raise(SkipTest('dev'))
        env.DIR_DATA = '/usr/local/climate_data/maurer/2010-concatenated'
        uri = 'Maurer02new_OBS_tasmax_daily.1971-2000.nc'
        variable = 'tasmax'
        env.DIR_CACHE = self._test_dir
        
        tc = []
        tnc = []
        for use_caching in [True,False]:
            for ii in range(10):
                print(use_caching,ii)
                t1 = time.time()
                env.USE_CACHING = use_caching
                rd = RequestDataset(uri,variable)
                temporal = rd.ds.temporal.value
                t2 = time.time() - t1
                if use_caching and ii != 0:
                    tc.append(t2)
                else:
                    tnc.append(t2)
                    
        print()
        print('time no cache',np.mean(tnc),np.std(tnc))
        print('time cache',np.mean(tc),np.std(tc))
        import ipdb;ipdb.set_trace()


if __name__ == '__main__':
    unittest.main()