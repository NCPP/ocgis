import unittest
from ocgis.util.shp_cabinet import ShpCabinet
import time
from ocgis.util.helpers import get_temp_path
import shutil
import ConfigParser
import os.path
import ocgis


class Test(unittest.TestCase):
    
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
        sc = ShpCabinet()
        for key in sc.keys():
            geoms = sc.get_geoms(key)
            self.assertTrue(len(geoms) > 2)
                
    def test_attribute_flags(self):
        attr_flags = ['none','all']
        for attr_flag in attr_flags:
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
                config.set('mapping','attributes',attr_flag)
                with open(os.path.join(path,key,key+'.cfg'),'w') as f:
                    config.write(f)
                ## load data
                sc = ShpCabinet(path)
                geoms = sc.get_geoms(key)
                
                for geom in geoms:
                    if attr_flag == 'none':
                        self.assertEqual(set(['ugid','geom']),set(geom.keys()))
                    else:
                        self.assertEqual(set(['ugid', 'state_name', 'state_fips', 'geom', 'state_abbr', 'id']),
                                         set(geom.keys()))
            finally:
                shutil.rmtree(path)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()