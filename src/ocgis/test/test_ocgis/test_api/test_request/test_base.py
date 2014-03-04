import unittest
from ocgis.exc import DefinitionValidationError, NoUnitsError
from ocgis.api.request.base import RequestDataset, RequestDatasetCollection
import ocgis
from ocgis import env, constants
from ocgis.test.base import TestBase
import os
import pickle
from datetime import datetime as dt
import shutil
from ocgis.test.test_simple.test_simple import nc_scope, ToTest
import datetime
from ocgis.api.operations import OcgOperations
import numpy as np
from cfunits.cfunits import Units


class TestRequestDataset(TestBase):
    
    def setUp(self):
        TestBase.setUp(self)
        ## download test data
        self.test_data.get_rd('cancm4_rhs')
        self.uri = os.path.join(ocgis.env.DIR_TEST_DATA,'CanCM4','rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc')
        self.variable = 'rhs'
        
    def test_source_dictionary_is_deepcopied(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        self.assertEqual(rd._source_metadata,field.meta)
        ## the source metadata dictionary should be deepcopied prior to passing
        ## to a request dataset
        rd._source_metadata['dim_map'] = None
        self.assertNotEqual(rd._source_metadata,field.meta)
        
    def test_source_index_matches_constant_value(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        self.assertEqual(field.temporal._src_idx.dtype,constants.np_int)
        
    def test_with_units(self):
        units = 'celsius'
        rd = self.test_data.get_rd('cancm4_tas',kwds={'units':units})
        self.assertEqual(rd.units,'celsius')
    
    def test_without_units_attempting_conform(self):
        ## this will work because the units read from the metadata are equivalent
        self.test_data.get_rd('cancm4_tas',kwds={'conform_units_to':'celsius'})
        ## this will not work because the units are not equivalent
        with self.assertRaises(ValueError):
            self.test_data.get_rd('cancm4_tas',kwds={'conform_units_to':'coulomb'})
            
    def test_with_bad_units_attempting_conform(self):
        ## pass bad units to the constructor and an attempt a conform. values from
        ## the source dataset are not used for overload.
        with self.assertRaises(ValueError):
            self.test_data.get_rd(
             'cancm4_tas',
             kwds={'conform_units_to':'celsius','units':'coulomb'})
            
    def test_nonsense_units(self):
        with self.assertRaises(ValueError):
            self.test_data.get_rd('cancm4_tas',
                                  kwds={'units':'nonsense','conform_units_to':'celsius'})
            
    def test_with_bad_units_passing_to_field(self):
        rd = self.test_data.get_rd('cancm4_tas',kwds={'units':'celsius'})
        field = rd.get()
        self.assertEqual(field.variables['tas'].units,'celsius')
        
    def test_get_field_with_overloaded_units(self):
        rd = self.test_data.get_rd('cancm4_tas',kwds={'conform_units_to':'celsius'})
        preload = [False,True]
        for pre in preload:
            field = rd.get()
            ## conform units argument needs to be attached to a field variable
            self.assertEqual(field.variables['tas']._conform_units_to,'celsius')
            sub = field.get_time_region({'year':[2009],'month':[5]})
            if pre:
                ## if we wanted to load the data prior to subset then do so and
                ## manually perform the units conversion
                to_test = Units.conform(sub.variables['tas'].value,sub.variables['tas'].cfunits,Units('celsius'))
            ## assert the conform attribute makes it though the subset
            self.assertEqual(sub.variables['tas']._conform_units_to,'celsius')
            value = sub.variables['tas'].value
            self.assertAlmostEqual(np.ma.mean(value),5.9219375118132564)
            self.assertAlmostEqual(np.ma.median(value),10.745431900024414)
            if pre:
                ## assert the manually converted array matches the loaded
                ## value
                self.assertNumpyAll(to_test,value)
                
    def test_get_field_nonequivalent_units_in_source_data(self):
        new_path = self.test_data.copy_file('cancm4_tas',self._test_dir)
        
        ## put non-equivalent units on the source data and attempto to conform
        with nc_scope(new_path,'a') as ds:
            ds.variables['tas'].units = 'coulomb'
        with self.assertRaises(ValueError):
            RequestDataset(uri=new_path,variable='tas',conform_units_to='celsius')
        
        ## remove units altogether
        with nc_scope(new_path,'a') as ds:
            ds.variables['tas'].delncattr('units')
        with self.assertRaises(NoUnitsError):
            RequestDataset(uri=new_path,variable='tas',conform_units_to='celsius')
    
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
    
    def test_with_alias(self):        
        rd = RequestDataset(self.uri,self.variable,alias='an_alias')
        self.assertEqual(rd.alias,'an_alias')
        rd = RequestDataset(self.uri,self.variable,alias=None)
        self.assertEqual(rd.alias,self.variable)
        
    def test_time_range(self):        
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
            
    def test_level_range(self):
        lr = '1|1'
        rd = RequestDataset(self.uri,self.variable,level_range=lr)
        self.assertEqual(rd.level_range,[1,1])
        
        with self.assertRaises(DefinitionValidationError):
            rd = RequestDataset(self.uri,self.variable,level_range=[2,1])
        
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


class TestRequestDatasetCollection(TestBase):
    
    def test(self):
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
        
    def test_with_overloads(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        ## loaded calendar should match file metadata
        self.assertEqual(field.temporal.calendar,'365_day')
        ## the overloaded calendar in the request dataset should still be None
        self.assertEqual(rd.t_calendar,None)
        
        dataset = [{'time_region': None,
                    'uri':[rd.uri],
                    'alias': u'tas', 
                    't_units': u'days since 1940-01-01 00:00:00',
                    'variable': u'tas',
                    't_calendar': u'will_not_work'}]
        rdc = RequestDatasetCollection(dataset)
        rd2 = RequestDataset(**dataset[0])
        ## the overloaded calendar should be passed to the request dataset
        self.assertEqual(rd2.t_calendar,'will_not_work')
        self.assertEqual(rdc[0].t_calendar,'will_not_work')
        ## when this bad calendar value is used it should raise an exception
        with self.assertRaises(ValueError):
            rdc[0].get().temporal.value_datetime
            
        dataset = [{'time_region': None,
                    'uri':[rd.uri],
                    'alias': u'tas', 
                    't_units': u'days since 1940-01-01 00:00:00',
                    'variable': u'tas'}]
        rdc = RequestDatasetCollection(dataset)
        ## ensure the overloaded units are properly passed
        self.assertEqual(rdc[0].get().temporal.units,'days since 1940-01-01 00:00:00')
        ## the calendar was not overloaded and the value should be read from
        ## the metadata
        self.assertEqual(rdc[0].get().temporal.calendar,'365_day')
        
    def test_with_overloads_real_data(self):
        ## copy the test file as the calendar attribute will be modified
        rd = self.test_data.get_rd('cancm4_tas')
        filename = os.path.split(rd.uri)[1]
        dest = os.path.join(self._test_dir,filename)
        shutil.copy2(rd.uri,dest)
        ## modify the calendar attribute
        with nc_scope(dest,'a') as ds:
            self.assertEqual(ds.variables['time'].calendar,'365_day')
            ds.variables['time'].calendar = '365_days'
        ## assert the calendar is in fact changed on the source file
        with nc_scope(dest,'r') as ds:
            self.assertEqual(ds.variables['time'].calendar,'365_days')
        rd2 = RequestDataset(uri=dest,variable='tas')
        field = rd2.get()
        ## the bad calendar will raise a value error when the datetimes are
        ## converted.
        with self.assertRaises(ValueError):
            field.temporal.value_datetime
        ## overload the calendar and confirm the datetime values are the same
        ## as the datetime values from the original good file
        rd3 = RequestDataset(uri=dest,variable='tas',t_calendar='365_day')
        field = rd3.get()
        self.assertNumpyAll(field.temporal.value_datetime,rd.get().temporal.value_datetime)
        
        ## pass as a dataset collection to operations and confirm the data may
        ## be written to a flat file. dates are converted in the process.
        time_range = (datetime.datetime(2001, 1, 1, 0, 0), datetime.datetime(2011, 1, 1, 0, 0))
        dataset = [{'time_region': None, 
          'uri': dest, 
          'time_range': time_range,
          'alias': u'tas',
          't_units': u'days since 1850-1-1',
          'variable': u'tas',
          't_calendar': u'365_day'}]
        rdc = RequestDatasetCollection(dataset)
        ops = OcgOperations(dataset=rdc,geom='state_boundaries',select_ugid=[25],output_format='csv+')
        ops.execute()
    

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
