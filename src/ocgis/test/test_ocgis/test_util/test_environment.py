import unittest
from ocgis import env, OcgOperations
import os
import tempfile
from ocgis.test.base import TestBase
from ocgis.util.environment import EnvParmImport
from importlib import import_module


class TestEnvImportParm(TestBase):
    reset_env = False
    
    def test_constructor(self):
        pm = EnvParmImport('USE_NUMPY',None,'numpy')
        self.assertEqual(pm.value,True)
        self.assertEqual(pm.module_name,'numpy')
        
    def test_bad_import(self):
        pm = EnvParmImport('USE_FOO',None,'foo')
        self.assertEqual(pm.value,False)
        
    def test_import_available_overloaded(self):
        pm = EnvParmImport('USE_NUMPY',False,'numpy')
        self.assertEqual(pm.value,False)
        
    def test_environment_variable(self):
        os.environ['OCGIS_USE_FOOL_GOLD'] = 'True'
        pm = EnvParmImport('USE_FOOL_GOLD',None,'foo')
        self.assertEqual(pm.value,True)
        

class Test(TestBase):
    reset_env = False
    
    def get_is_available(self,module_name):
        try:
            import_module(module_name)
            av = True
        except ImportError:
            av = False
        return(av)
    
    def test_import_attributes(self):
        ## with both modules installed, these are expected to be true
        self.assertEqual(env.USE_CFUNITS,self.get_is_available('cfunits'))
        self.assertEqual(env.USE_SPATIAL_INDEX,self.get_is_available('rtree'))
        
        ## turn off the spatial index
        env.USE_SPATIAL_INDEX = False
        self.assertEqual(env.USE_SPATIAL_INDEX,False)
        env.reset()
        self.assertEqual(env.USE_SPATIAL_INDEX,self.get_is_available('rtree'))
        
    def test_import_attributes_overloaded(self):
        try:
            import rtree
            av = True
        except ImportError:
            av = False
        self.assertEqual(env.USE_SPATIAL_INDEX,av)
        
        os.environ['OCGIS_USE_SPATIAL_INDEX'] = 'False'
        env.reset()
        self.assertFalse(env.USE_SPATIAL_INDEX)
        
        os.environ['OCGIS_USE_SPATIAL_INDEX'] = 't'
        env.reset()
        self.assertTrue(env.USE_SPATIAL_INDEX)
        
        ## this cannot be transformed into a boolean value, and it is also not
        ## a realistic module name, so it will evaluate to false
        os.environ['OCGIS_USE_SPATIAL_INDEX'] = 'False'
        env.reset()
        self.assertFalse(env.USE_SPATIAL_INDEX)
        
        os.environ['OCGIS_USE_SPATIAL_INDEX'] = 'f'
        env.reset()
        self.assertFalse(env.USE_SPATIAL_INDEX)

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
            rd = self.test_data_nc.get_rd('daymet_tmax')
            ops = OcgOperations(dataset=rd,snippet=True)
            self.assertEqual(env.DIR_OUTPUT,ops.dir_output)
            self.assertEqual(env.PREFIX,ops.prefix)
        finally:
            os.rmdir(out)
            env.reset()
        
    def test_str(self):
        ret = str(env)
        self.assertTrue(len(ret) > 300)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
