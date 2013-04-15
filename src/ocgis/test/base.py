import unittest
import abc
import tempfile
from ocgis import env, RequestDataset
import shutil
from copy import deepcopy
import os
from collections import OrderedDict


class TestBase(unittest.TestCase):
    '''All tests should inherit from this. It allows test data to be written to
    a temporary folder and removed easily.'''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,*args,**kwds):
        self.test_data = TestData()
        self.test_data.update(['daymet'],'tmax','tmax.nc',key='daymet_tmax')
        self.test_data.update(['CanCM4'],'tas','tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',key='cancm4_tas')
        self.test_data.update(['CanCM4'],'tasmax','tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',key='cancm4_tasmax_2011')
        self.test_data.update(['CanCM4'],'tasmax','tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',key='cancm4_tasmax_2001')
        self.test_data.update(['CanCM4'],'rhs','rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',key='cancm4_rhs')
        super(self,TestBase).__init__(*args,**kwds)
    
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
            
            
class TestData(OrderedDict):
    
    def update(self,collection,variable,filename,key=None):
        OrderedDict.update(self,{key or filename:{'collection':collection,
         'filename':filename,'variable':variable}})
        
    def get_rd(self,key):
        ref = self[key]
        coll = deepcopy(ref['collection'])
        if env.DIR_TEST_DATA is None:
            raise(ValueError('The TestDataset object requires env.DIR_TEST_DATA have a path value.'))
        coll.insert(0,env.DIR_DATA)
        coll.append(ref['filename'])
        uri = os.path.join(*coll)
        rd = RequestDataset(uri,ref['variable'])
        return(rd)
