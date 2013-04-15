import unittest
from ocgis import env, OcgOperations
import os
import tempfile
from ocgis.test.base import TestBase


class Test(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_data = TestBase.get_tdata()

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
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()