import unittest
from ocgis.util.shp_cabinet import ShpCabinet
import ocgis
from osgeo import ogr
from unittest.case import SkipTest
from ocgis.test.base import TestBase
import os
import shutil


class Test(TestBase):
    
    def test_sql_subset(self):
        sc = ShpCabinet()
        path = sc.get_shp_path('state_boundaries')
        ds = ogr.Open(path)
        ret = ds.ExecuteSQL('select * from state_boundaries where state_name = "New Jersey"')
        ret.ResetReading()
        self.assertEqual(len(ret),1)
    
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

    def test_get_keys(self,dir_shpcabinet=None):
        ocgis.env.DIR_SHPCABINET = dir_shpcabinet or ocgis.env.DIR_SHPCABINET
        sc = ShpCabinet()
        ret = sc.keys()
        target_keys = ['state_boundaries','us_counties','mi_watersheds','world_countries','climate_divisions','urban_areas_2000','co_watersheds']
        self.assertEqual(len(set(target_keys).intersection(set(ret))),len(target_keys))
        
    def test_load_all(self):
        raise(SkipTest('too long for no benefit'))
        sc = ShpCabinet()
        for key in sc.keys():
            geoms = sc.get_geoms(key)
            self.assertTrue(len(geoms) > 2)
            
    def test_shapefiles_not_in_folders(self):
        for dirpath,dirnames,filenames in os.walk(ocgis.env.DIR_SHPCABINET):
            for filename in filenames:
                dst = os.path.join(self._test_dir,filename)
                src = os.path.join(dirpath,filename)
                shutil.copy2(src,dst)
        self.test_get_keys(dir_shpcabinet=self._test_dir)
        
        sc = ShpCabinet(path=self._test_dir)
        path = sc.get_shp_path('world_countries')
        self.assertEqual(path,os.path.join(self._test_dir,'world_countries.shp'))
        
    def test_huc_loading(self):
        raise(SkipTest('tests loading time for HUC data'))
        sc = ShpCabinet()
        geoms = sc.get_geoms('WBDHU12_June2013',select_ugid=[2221,5000])
        self.assertEqual(len(geoms),2)
        self.assertEqual(geoms[0]['UGID'],2221)
            
            
if __name__ == '__main__':
    unittest.main()