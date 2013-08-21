import unittest
from ocgis import env, OcgOperations
import os
import tempfile
from ocgis.test.base import TestBase
from ocgis.interface.projection import WGS84, LambertConformalConic
from ocgis.exc import OcgisEnvironmentError


class Test(TestBase):

    def test_simple(self):
        self.assertEqual(env.OVERWRITE,False)
        env.reset()
        os.environ['OCGIS_OVERWRITE'] = 't'
        self.assertEqual(env.OVERWRITE,True)
        env.OVERWRITE = False
        self.assertEqual(env.OVERWRITE,False)
        with self.assertRaises(AttributeError):
            env.FOO = 1
        
        env.OVERWRITE = True
        self.assertEqual(env.OVERWRITE,True)
        env.reset()
        os.environ.pop('OCGIS_OVERWRITE')
        self.assertEqual(env.OVERWRITE,False)
        
    def test_env_overload(self):
        ## check env overload
        out = tempfile.mkdtemp()
        try:
            env.DIR_OUTPUT = out
            env.PREFIX = 'my_prefix'
            rd = self.test_data.get_rd('daymet_tmax')
            ops = OcgOperations(dataset=rd,snippet=True)
            self.assertEqual(env.DIR_OUTPUT,ops.dir_output)
            self.assertEqual(env.PREFIX,ops.prefix)
        finally:
            os.rmdir(out)
            env.reset()
            
    def test_reference_projection(self):
        self.assertIsInstance(env.REFERENCE_PROJECTION,WGS84)
        
        ## the reference projection must be set at runtime.
        os.environ['OCGIS_REFERENCE_PROJECTION'] = 'foo'
        with self.assertRaises(OcgisEnvironmentError):
            env.reset()
        os.environ.pop('OCGIS_REFERENCE_PROJECTION')
        
    def test_str(self):
        ret = str(env)
        self.assertTrue(len(ret) > 300)

        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()