import unittest
from ocgis.util.shp_cabinet import ShpCabinet
import time
from ocgis.util.helpers import get_temp_path
import shutil
import ConfigParser
import os.path


class Test(unittest.TestCase):

    def test_get_keys(self):
        sc = ShpCabinet()
        ret = sc.keys()
        self.assertTrue(len(ret) >= 1)
        
    def test_load_all(self):
        sc = ShpCabinet()
        for key in sc.keys():
            geoms = sc.get_geoms(key)
            self.assertTrue(len(geoms) > 2)
            
    def test_unwrap(self):
        sc = ShpCabinet()
        _key = ['state_boundaries','world_countries']
        for key in _key:
            geoms = sc.get_geoms(key,unwrap=True)
            for geom in geoms:
                x = geom['geom'].centroid.x
                self.assertTrue(x > 0)
                
    def test_unwrap_pm(self):
        _pm = [-4.0,-10.0,-20.0,5.0]
        sc = ShpCabinet()
        for pm in _pm:
            geoms = sc.get_geoms('world_countries',unwrap=True,pm=pm)
            path = '/tmp/shp{0}.shp'.format(time.time())
            sc.write(geoms,path)
            for geom in geoms:
                bounds = geom['geom'].bounds
                self.assertTrue(bounds[0] >= pm)
                
    def test_none_for_attributes(self):
        ## make temporary directory
        path = get_temp_path(only_dir=True,nest=True)
        try:
            ## the shapefile key to use
            key = 'state_boundaries'
            ## get path to shapefile and copy directory to temporary location
            sc = ShpCabinet()
            origin = os.path.split(sc.get_shp_path(key))[0]
            shutil.copytree(origin,os.path.join(path,key))
            ## remove original config file
            os.remove(os.path.join(path,key,key+'.cfg'))
            ## write config file
            config = ConfigParser.ConfigParser()
            config.add_section('mapping')
            config.set('mapping','ugid','none')
            config.set('mapping','attributes','none')
            with open(os.path.join(path,key,key+'.cfg'),'w') as f:
                config.write(f)
            ## load data
            sc = ShpCabinet(path)
            geoms = sc.get_geoms(key)
            
            for geom in geoms:
                self.assertEqual(set(['ugid','geom']),set(geom.keys()))
        finally:
            shutil.rmtree(path)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()