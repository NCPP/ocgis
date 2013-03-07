import unittest
from ocgis import env
import os


class Test(unittest.TestCase):

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
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()