import unittest
from ocgis.util.shp_cabinet import ShpCabinet
import time
from ocgis.util.helpers import get_temp_path
import shutil
import ConfigParser
import os.path
import ocgis
from osgeo import ogr
from unittest.case import SkipTest


class Test(unittest.TestCase):
    
    def test_sql_subset(self):
        sc = ShpCabinet()
        path = sc.get_shp_path('state_boundaries')
        ds = ogr.Open(path)
        ret = ds.ExecuteSQL('select * from state_boundaries where state_name = "New Jersey"')
        ret.ResetReading()
        for feat in ret:
            pass
    
    def test_bad_path(self):
        bp = '/a/bad/location'
        with self.assertRaises(ValueError):
            ShpCabinet(bp).get_geoms('state_boundaries')
            
            
    def test_none_path(self):
        try:
            ocgis.env.DIR_SHPCABINET = None
            with self.assertRaises(ValueError):
                ShpCabinet().get_geoms('state_boundaries')
                
        finally:
            ocgis.env.reset()

    def test_get_keys(self):
        sc = ShpCabinet()
        ret = sc.keys()
        self.assertTrue(len(ret) >= 1)
        
    def test_load_all(self):
        raise(SkipTest('too long for no benefit'))
        sc = ShpCabinet()
        for key in sc.keys():
            geoms = sc.get_geoms(key)
            self.assertTrue(len(geoms) > 2)
