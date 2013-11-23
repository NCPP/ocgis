import unittest
from ocgis.exc import DefinitionValidationError
from ocgis.api.request.base import RequestDataset, RequestDatasetCollection
import ocgis
from ocgis import env
from ocgis.test.base import TestBase
import os
import pickle
from datetime import datetime as dt


class TestRequestDataset(TestBase):
    
    def setUp(self):
        TestBase.setUp(self)
        ## download test data
        self.test_data.get_rd('cancm4_rhs')
        self.uri = os.path.join(ocgis.env.DIR_TEST_DATA,'CanCM4','rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc')
        self.variable = 'rhs'
    
    def test_pickle(self):
        rd = RequestDataset(uri=self.uri,variable=self.variable)
        rd_path = os.path.join(ocgis.env.DIR_OUTPUT,'rd.pkl')
        with open(rd_path,'w') as f:
            pickle.dump(rd,f)
        with open(rd_path,'r') as f:
            rd2 = pickle.load(f)
        self.assertTrue(rd == rd2)
    
    def test_inspect_method(self):
        rd = RequestDataset(self.uri,self.variable)
        rd.inspect()
        
    def test_inspect_as_dct(self):
        variables = [
                     self.variable,
                     None,
                     'foo',
                     'time'
                     ]
        
        for variable in variables:
            try:
                rd = RequestDataset(self.uri,variable)   
                ret = rd.inspect_as_dct()
            except KeyError:
                if variable == 'foo':
                    continue
                else:
                    raise
            except ValueError:
                if variable == 'time':
                    continue
                else:
                    raise
            except AssertionError:
                if variable is not None:
                    raise
                else:
                    continue
            ref = ret['derived']
            
            if variable is None:
                self.assertEqual(ref,{'End Date': '2020-12-31 12:00:00', 'Start Date': '2011-01-01 12:00:00'})
            else:
                self.assertEqual(ref['End Date'],'2021-01-01 00:00:00')
    
    def test_env_dir_data(self):
        ## test setting the var to a single directory
        env.DIR_DATA = ocgis.env.DIR_TEST_DATA
        rd = self.test_data.get_rd('cancm4_rhs')
        target = os.path.join(env.DIR_DATA,'CanCM4','rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc')
        try:
            self.assertEqual(rd.uri,target)
        ## attempt to normalize the path
        except AssertionError:
            self.assertEqual(rd.uid,os.path.normpath(target))
        
        ## test none and not finding the data
        env.DIR_DATA = None
        with self.assertRaises(ValueError):
            RequestDataset('does_not_exists.nc',variable='foo')
            
        ## set data directory and not find it.
        env.DIR_DATA = os.path.join(ocgis.env.DIR_TEST_DATA,'CCSM4')
        with self.assertRaises(ValueError):
            RequestDataset('rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',variable='rhs')
    
    def test_RequestDataset(self):        
        rd = RequestDataset(self.uri,self.variable,alias='an_alias')
        self.assertEqual(rd.alias,'an_alias')
        rd = RequestDataset(self.uri,self.variable,alias=None)
        self.assertEqual(rd.alias,self.variable)
        
    def test_RequestDataset_time_range(self):        
        tr = [dt(2000,1,1),dt(2000,12,31)]
        rd = RequestDataset(self.uri,self.variable,time_range=tr)
        self.assertEqual(rd.time_range,tr)
        
        out = [dt(2000, 1, 1, 0, 0),dt(2000, 12, 31,)]
        tr = '2000-1-1|2000-12-31'
        rd = RequestDataset(self.uri,self.variable,time_range=tr)
        self.assertEqual(rd.time_range,out)
        
        tr = '2000-12-31|2000-1-1'
        with self.assertRaises(DefinitionValidationError):
            rd = RequestDataset(self.uri,self.variable,time_range=tr)
            
    def test_RequestDataset_level_range(self):
        lr = '1|1'
        rd = RequestDataset(self.uri,self.variable,level_range=lr)
        self.assertEqual(rd.level_range,[1,1])
        
        with self.assertRaises(DefinitionValidationError):
            rd = RequestDataset(self.uri,self.variable,level_range=[2,1])
    
    def test_RequestDatasetCollection(self):
        env.DIR_DATA = ocgis.env.DIR_TEST_DATA
        
        daymet = self.test_data.get_rd('daymet_tmax')
        tas = self.test_data.get_rd('cancm4_tas')
        
        uris = [daymet.uri,
                tas.uri]
        variables = ['foo1','foo2']
        rdc = RequestDatasetCollection()
        for uri,variable in zip(uris,variables):
            rd = RequestDataset(uri,variable)
            rdc.update(rd)
        self.assertEqual([1,2],[rd.did for rd in rdc])
            
        variables = ['foo1','foo1']
        rdc = RequestDatasetCollection()
        for ii,(uri,variable) in enumerate(zip(uris,variables)):
            rd = RequestDataset(uri,variable)
            if ii == 1:
                with self.assertRaises(KeyError):
                    rdc.update(rd)
            else:
                rdc.update(rd)
                
        aliases = ['a1','a2']
        for uri,variable,alias in zip(uris,variables,aliases):
            rd = RequestDataset(uri,variable,alias=alias)
            rdc.update(rd)
        for row in rdc:
            self.assertIsInstance(row,RequestDataset)
        self.assertIsInstance(rdc[0],RequestDataset)
        self.assertIsInstance(rdc['a2'],RequestDataset)
        
    def test_multiple_uris(self):
        rd = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        self.assertEqual(len(rd.uri),2)
        rd.inspect()
        
    def test_time_region(self):
        tr1 = {'month':[6],'year':[2001]}
        rd = RequestDataset(uri=self.uri,variable=self.variable,time_region=tr1)
        self.assertEqual(rd.time_region,tr1)
        
        tr2 = {'bad':15}
        with self.assertRaises(DefinitionValidationError):
            RequestDataset(uri=self.uri,variable=self.variable,time_region=tr2)
        
        with self.assertRaises(NotImplementedError):
            tr_str = 'month~6|year~2001'
            rd = RequestDataset(uri=self.uri,variable=self.variable,time_region=tr_str)
            self.assertEqual(rd.time_region,tr1)
            
            tr_str = 'month~6-8|year~2001-2003'
            rd = RequestDataset(uri=self.uri,variable=self.variable,time_region=tr_str)
            self.assertEqual(rd.time_region,{'month':[6,7,8],'year':[2001,2002,2003]})
            
            tr_str = 'month~6-8'
            rd = RequestDataset(uri=self.uri,variable=self.variable,time_region=tr_str)
            self.assertEqual(rd.time_region,{'month':[6,7,8],'year':None})
            
            tr_str = 'month~6-8|year~none'
            rd = RequestDataset(uri=self.uri,variable=self.variable,time_region=tr_str)
            self.assertEqual(rd.time_region,{'month':[6,7,8],'year':None})


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()