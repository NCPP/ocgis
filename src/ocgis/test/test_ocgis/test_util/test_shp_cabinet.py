import unittest
from ocgis.test.base import TestBase
from ocgis.util.shp_cabinet import ShpCabinet
from unittest.case import SkipTest


class TestShpCabinet(TestBase):


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
            
    def test_iter_all(self):
        raise(SkipTest('dev - long'))
        sc = ShpCabinet()
        for key in sc.keys():
            print key
            for geom in sc.iter_geoms(key):
                self.assertTrue(True)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()