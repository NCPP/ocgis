import unittest
from ocgis.util.shp_cabinet import ShpCabinet
import time


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

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()