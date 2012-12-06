import unittest
from ocgis.util.shp_cabinet import ShpCabinet


class TestShpCabinet(unittest.TestCase):

    def test_get_keys(self):
        sc = ShpCabinet()
        ret = sc.keys()
        self.assertTrue(len(ret) >= 1)
        
    def test_load_all(self):
        sc = ShpCabinet()
        for key in sc.keys():
            geoms = sc.get_geoms(key)
            self.assertTrue(len(geoms) > 2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()