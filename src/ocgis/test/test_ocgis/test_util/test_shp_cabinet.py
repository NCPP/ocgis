import unittest
from ocgis.util.shp_cabinet import ShpCabinet, ShpCabinetIterator
import ocgis
from osgeo import ogr
from ocgis.test.base import TestBase
import os
import shutil


class Test(TestBase):
    
    def test_iteration_by_path(self):
        ## test that a shapefile may be retrieved by passing a full path to the
        ## file
        path = ShpCabinet().get_shp_path('state_boundaries')
        ocgis.env.DIR_SHPCABINET = None
        sci = ShpCabinetIterator(path=path)
        self.assertEqual(len(list(sci)),51)
    
    def test_iteration_by_path_with_bad_path(self):
        ## if the path does not exist on the filesystem, then an exception should
        ## be raised
        ocgis.env.DIR_SHPCABINET = None
        sci = ShpCabinetIterator(path='/foo/foo/foo/foo/foo')
        with self.assertRaises(RuntimeError):
            list(sci)
            
    def test_key_used_before_path(self):
        ## the key always takes preference over the path
        sci = ShpCabinetIterator(key='state_boundaries',path='/foo/foo/foo/foo/foo')
        self.assertEqual(len(list(sci)),51)
    
    def test_iter_geoms(self):
        sc = ShpCabinet()
        it = sc.iter_geoms('state_boundaries')
        geoms = list(it)
        self.assertEqual(len(geoms),51)
        self.assertEqual(geoms[12]['properties']['STATE_NAME'],'New Hampshire')
        
    def test_iter_geoms_select_ugid(self):
        sc = ShpCabinet()
        it = sc.iter_geoms('state_boundaries',select_ugid=[13])
        geoms = list(it)
        self.assertEqual(len(geoms),1)
        self.assertEqual(geoms[0]['properties']['STATE_NAME'],'New Hampshire')
    
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
            list(ShpCabinet(bp).iter_geoms('state_boundaries'))
            
    def test_none_path(self):
        try:
            ocgis.env.DIR_SHPCABINET = None
            with self.assertRaises(ValueError):
                list(ShpCabinet().iter_geoms('state_boundaries'))
        finally:
            ocgis.env.reset()

    def test_get_keys(self,dir_shpcabinet=None):
        ocgis.env.DIR_SHPCABINET = dir_shpcabinet or ocgis.env.DIR_SHPCABINET
        sc = ShpCabinet()
        ret = sc.keys()
        target_keys = ['state_boundaries','world_countries']
        self.assertEqual(len(set(target_keys).intersection(set(ret))),len(target_keys))
            
    def test_shapefiles_not_in_folders(self):
        for dirpath,dirnames,filenames in os.walk(ocgis.env.DIR_SHPCABINET):
            for filename in filenames:
                if filename.startswith('state_boundaries') or filename.startswith('world_countries'):
                    dst = os.path.join(self._test_dir,filename)
                    src = os.path.join(dirpath,filename)
                    shutil.copy2(src,dst)
        self.test_get_keys(dir_shpcabinet=self._test_dir)
        
        sc = ShpCabinet(path=self._test_dir)
        path = sc.get_shp_path('world_countries')
        self.assertEqual(path,os.path.join(self._test_dir,'world_countries.shp'))
            
            
if __name__ == '__main__':
    unittest.main()
    