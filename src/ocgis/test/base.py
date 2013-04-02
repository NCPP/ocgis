import unittest
import abc
import tempfile
from ocgis import env
import shutil


class TestBase(unittest.TestCase):
    __metaclass__ = abc.ABCMeta
    
    @classmethod
    def setUpClass(cls):
        env.DIR_OUTPUT = tempfile.mkdtemp(prefix='ocgis_test_',dir=env.DIR_OUTPUT)
        env.OVERWRITE = True
        
    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(env.DIR_OUTPUT)
        finally:
            env.reset()