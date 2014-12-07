import unittest
import json
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from ocgis.test.base import TestBase, nc_scope, attr
from ocgis.contrib.library_icclim import IcclimTG, IcclimSU, AbstractIcclimFunction,\
    IcclimDTR, IcclimETR, IcclimTN, IcclimTX,\
    AbstractIcclimUnivariateSetFunction, AbstractIcclimMultivariateFunction
from ocgis.calc.library.statistics import Mean
from ocgis.api.parms.definition import Calc, CalcGrouping
from ocgis.calc.library.register import FunctionRegistry, register_icclim
from ocgis.exc import DefinitionValidationError, UnitsValidationError
from ocgis.api.operations import OcgOperations
from ocgis.calc.library.thresholds import Threshold
import ocgis
from ocgis.util.helpers import itersubclasses
from ocgis.contrib import library_icclim


class TestLibraryIcclim(TestBase):
    
    def test_standard_AbstractIcclimFunction(self):
        shapes = ([('month',), 12],[('month', 'year'), 24],[('year',),2])
        ocgis.env.OVERWRITE = True
        keys = set(library_icclim._icclim_function_map.keys())
        for klass in [
                      AbstractIcclimUnivariateSetFunction,
                      AbstractIcclimMultivariateFunction]:
            for subclass in itersubclasses(klass):
                keys.remove(subclass.key)
                self.assertEqual([('month',),('month','year'),('year',)],subclass._allowed_temporal_groupings)
                for cg in CalcGrouping.iter_possible():
                    calc = [{'func':subclass.key,'name':subclass.key.split('_')[1]}]
                    if klass == AbstractIcclimUnivariateSetFunction:
                        rd = self.test_data.get_rd('cancm4_tas')
                        rd.time_region = {'year':[2001,2002]}
                        calc = [{'func':subclass.key,'name':subclass.key.split('_')[1]}]
                    else:
                        tasmin = self.test_data.get_rd('cancm4_tasmin_2001')
                        tasmax = self.test_data.get_rd('cancm4_tasmax_2001')
                        rd = [tasmin,tasmax]
                        for r in rd:
                            r.time_region = {'year':[2001,2002]}
                        calc[0].update({'kwds':{'tasmin':'tasmin','tasmax':'tasmax'}})
                    try:
                        ops = ocgis.OcgOperations(dataset=rd,
                                                  output_format='nc',
                                                  calc=calc,
                                                  calc_grouping=cg,
                                                  geom=[3.39,40.62,10.54,52.30])
                        ret = ops.execute()
                        to_test = None
                        for shape in shapes:
                            if shape[0] == cg:
                                to_test = shape[1]
                        with nc_scope(ret) as ds:
                            var = ds.variables[calc[0]['name']]
                            self.assertEqual(var.dtype,subclass.dtype)
                            self.assertEqual(var.shape,(to_test,5,4))
                    except DefinitionValidationError as e:
                        if e.message.startswith('''OcgOperations validation raised an exception on the argument/operation "calc_grouping" with the message: The following temporal groupings are supported for ICCLIM: [('month',), ('month', 'year'), ('year',)]. The requested temporal group is:'''):
                            pass
                        else:
                            raise(e)
        self.assertEqual(len(keys),0)
    
    def test_register_icclim(self):
        fr = FunctionRegistry()
        self.assertNotIn('icclim_TG',fr)
        register_icclim(fr)
        self.assertIn('icclim_TG',fr)
        self.assertIn('icclim_vDTR',fr)
    
    def test_calc_argument_to_operations(self):
        value = [{'func':'icclim_TG','name':'TG'}]
        calc = Calc(value)
        self.assertEqual(len(calc.value),1)
        self.assertEqual(calc.value[0]['ref'],IcclimTG)
        
    def test_bad_icclim_key_to_operations(self):
        value = [{'func':'icclim_TG_bad','name':'TG'}]
        with self.assertRaises(DefinitionValidationError):
            Calc(value)
            
            
class TestDTR(TestBase):
    
    def test_calculate(self):
        tasmin = self.test_data.get_rd('cancm4_tasmin_2001')
        tasmax = self.test_data.get_rd('cancm4_tasmax_2001')
        field = tasmin.get()
        field.variables.add_variable(deepcopy(tasmax.get().variables['tasmax']), assign_new_uid=True)
        field = field[:,0:600,:,25:50,25:50]
        tgd = field.temporal.get_grouping(['month'])
        dtr = IcclimDTR(field=field,tgd=tgd)
        ret = dtr.execute()
        self.assertEqual(ret['icclim_DTR'].value.shape,(1, 12, 1, 25, 25))
        
    def test_bad_keyword_mapping(self):
        tasmin = self.test_data.get_rd('cancm4_tasmin_2001')
        tas = self.test_data.get_rd('cancm4_tas')
        rds = [tasmin,tas]
        calc = [{'func':'icclim_DTR','name':'DTR','kwds':{'tas':'tasmin','tasmax':'tasmax'}}]
        with self.assertRaises(DefinitionValidationError):
            ocgis.OcgOperations(dataset=rds,calc=calc,calc_grouping=['month'],
                                output_format='nc')
            
        calc = [{'func':'icclim_DTR','name':'DTR'}]
        with self.assertRaises(DefinitionValidationError):
            ocgis.OcgOperations(dataset=rds,calc=calc,calc_grouping=['month'],
                                output_format='nc')
        
    def test_calculation_operations(self):
        ## note the kwds must contain a map of the required variables to their
        ## associated aliases.
        calc = [{'func':'icclim_DTR','name':'DTR','kwds':{'tasmin':'tasmin','tasmax':'tasmax'}}]
        tasmin = self.test_data.get_rd('cancm4_tasmin_2001')
        tasmin.time_region = {'year':[2002]}
        tasmax = self.test_data.get_rd('cancm4_tasmax_2001')
        tasmax.time_region = {'year':[2002]}
        rds = [tasmin,tasmax]
        ops = ocgis.OcgOperations(dataset=rds,calc=calc,calc_grouping=['month'],
                                  output_format='nc')
        ops.execute()


class TestETR(TestBase):
    
    def test_calculate(self):
        tasmin = self.test_data.get_rd('cancm4_tasmin_2001')
        tasmax = self.test_data.get_rd('cancm4_tasmax_2001')
        field = tasmin.get()
        field.variables.add_variable(tasmax.get().variables['tasmax'], assign_new_uid=True)
        field = field[:,0:600,:,25:50,25:50]
        tgd = field.temporal.get_grouping(['month'])
        dtr = IcclimETR(field=field,tgd=tgd)
        ret = dtr.execute()
        self.assertEqual(ret['icclim_ETR'].value.shape,(1, 12, 1, 25, 25))
        
    def test_calculate_rotated_pole(self):
        tasmin_fake = self.test_data.get_rd('rotated_pole_ichec')
        tasmin_fake.alias = 'tasmin'
        tasmax_fake = deepcopy(tasmin_fake)
        tasmax_fake.alias = 'tasmax'
        rds = [tasmin_fake,tasmax_fake]
        for rd in rds:
            rd.time_region = {'year':[1973]}
        calc_ETR = [{'func':'icclim_ETR','name':'ETR','kwds':{'tasmin':'tasmin','tasmax':'tasmax'}}]
        ops = ocgis.OcgOperations(dataset=[tasmin_fake,tasmax_fake],
                          calc=calc_ETR,
                          calc_grouping=['year', 'month'],
                          prefix = 'ETR_ocgis_icclim',
                          output_format = 'nc',
                          add_auxiliary_files=False)
        with nc_scope(ops.execute()) as ds:
            self.assertEqual(ds.variables['ETR'][:].shape,(12,103,106))


class TestTx(TestBase):
            
    def test_calculate_operations(self):
        rd = self.test_data.get_rd('cancm4_tas')
        slc = [None,None,None,[0,10],[0,10]]
        calc_icclim = [{'func':'icclim_TG','name':'TG'}]
        calc_ocgis = [{'func':'mean','name':'mean'}]
        _calc_grouping = [['month'],['month','year']]
        for cg in _calc_grouping:
            ops_ocgis = OcgOperations(calc=calc_ocgis,calc_grouping=cg,slice=slc,
                                      dataset=rd)
            ret_ocgis = ops_ocgis.execute()
            ops_icclim = OcgOperations(calc=calc_icclim,calc_grouping=cg,slice=slc,
                                      dataset=rd)
            ret_icclim = ops_icclim.execute()
            self.assertNumpyAll(ret_ocgis[1]['tas'].variables['mean'].value,
                                ret_icclim[1]['tas'].variables['TG'].value)
            
    def test_calculation_operations_to_nc(self):
        rd = self.test_data.get_rd('cancm4_tas')
        slc = [None,None,None,[0,10],[0,10]]
        ops_ocgis = OcgOperations(calc=[{'func':'icclim_TG','name':'TG'}],
                                  calc_grouping=['month'],
                                  slice=slc,
                                  dataset=rd,
                                  output_format='nc')
        ret = ops_ocgis.execute()
        with nc_scope(ret) as ds:
            self.assertIn('Calculation of TG indice (monthly climatology)',ds.history)
            self.assertEqual(ds.title,'ECA temperature indice TG')
            var = ds.variables['TG']
            ## check the JSON serialization
            self.assertEqual(ds.__dict__[AbstractIcclimFunction._global_attribute_source_name],
                             u'{"institution": "CCCma (Canadian Centre for Climate Modelling and Analysis, Victoria, BC, Canada)", "institute_id": "CCCma", "experiment_id": "decadal2000", "source": "CanCM4 2010 atmosphere: CanAM4 (AGCM15i, T63L35) ocean: CanOM4 (OGCM4.0, 256x192L40) sea ice: CanSIM1 (Cavitating Fluid, T63 Gaussian Grid) land: CLASS2.7", "model_id": "CanCM4", "forcing": "GHG,Oz,SA,BC,OC,LU,Sl,Vl (GHG includes CO2,CH4,N2O,CFC11,effective CFC12)", "parent_experiment_id": "N/A", "parent_experiment_rip": "N/A", "branch_time": 0.0, "contact": "cccma_info@ec.gc.ca", "references": "http://www.cccma.ec.gc.ca/models", "initialization_method": 1, "physics_version": 1, "tracking_id": "fac7bd83-dd7a-425b-b4dc-b5ab2e915939", "branch_time_YMDH": "2001:01:01:00", "CCCma_runid": "DHFP1B_E002_I2001_M01", "CCCma_parent_runid": "DHFP1_E002", "CCCma_data_licence": "1) GRANT OF LICENCE - The Government of Canada (Environment Canada) is the \\nowner of all intellectual property rights (including copyright) that may exist in this Data \\nproduct. You (as \\"The Licensee\\") are hereby granted a non-exclusive, non-assignable, \\nnon-transferable unrestricted licence to use this data product for any purpose including \\nthe right to share these data with others and to make value-added and derivative \\nproducts from it. This licence is not a sale of any or all of the owner\'s rights.\\n2) NO WARRANTY - This Data product is provided \\"as-is\\"; it has not been designed or \\nprepared to meet the Licensee\'s particular requirements. Environment Canada makes no \\nwarranty, either express or implied, including but not limited to, warranties of \\nmerchantability and fitness for a particular purpose. In no event will Environment Canada \\nbe liable for any indirect, special, consequential or other damages attributed to the \\nLicensee\'s use of the Data product.", "product": "output", "experiment": "10- or 30-year run initialized in year 2000", "frequency": "day", "creation_date": "2011-05-08T01:01:51Z", "history": "2011-05-08T01:01:51Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.", "Conventions": "CF-1.4", "project_id": "CMIP5", "table_id": "Table day (28 March 2011) f9d6cfec5981bb8be1801b35a81002f0", "title": "CanCM4 model output prepared for CMIP5 10- or 30-year run initialized in year 2000", "parent_experiment": "N/A", "modeling_realm": "atmos", "realization": 2, "cmor_version": "2.5.4"}')
            ## load the original source attributes from the JSON string
            json.loads(ds.__dict__[AbstractIcclimFunction._global_attribute_source_name])
            self.assertEqual(dict(var.__dict__),{'_FillValue':np.float32(1e20),u'units': u'K', 'grid_mapping': 'latitude_longitude', u'standard_name': AbstractIcclimFunction.standard_name, u'long_name': u'Mean of daily mean temperature'})

    def test_calculate(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        field = field[:,:,:,0:10,0:10]
        klasses = [IcclimTG,IcclimTN,IcclimTX]
        for klass in klasses:
            for calc_grouping in [['month'],['month','year']]:
                tgd = field.temporal.get_grouping(calc_grouping)
                itg = klass(field=field,tgd=tgd)
                ret_icclim = itg.execute()
                mean = Mean(field=field,tgd=tgd)
                ret_ocgis = mean.execute()
                self.assertNumpyAll(ret_icclim[klass.key].value,
                                    ret_ocgis['mean'].value)


class TestSU(TestBase):
    
    def test_calculate(self):
        rd = self.test_data.get_rd('cancm4_tasmax_2011')
        field = rd.get()
        field = field[:,:,:,0:10,0:10]
        for calc_grouping in [['month'],['month','year']]:
            tgd = field.temporal.get_grouping(calc_grouping)
            itg = IcclimSU(field=field,tgd=tgd)
            ret_icclim = itg.execute()
            threshold = Threshold(field=field,tgd=tgd,parms={'threshold':298.15,'operation':'gt'})
            ret_ocgis = threshold.execute()
            self.assertNumpyAll(ret_icclim['icclim_SU'].value,ret_ocgis['threshold'].value)
            
    def test_calculation_operations_bad_units(self):
        rd = self.test_data.get_rd('daymet_tmax')
        calc_icclim = [{'func':'icclim_SU','name':'SU'}]
        ops_icclim = OcgOperations(calc=calc_icclim,calc_grouping=['month','year'],dataset=rd)
        with self.assertRaises(UnitsValidationError):
            ops_icclim.execute()
            
    def test_calculation_operations_to_nc(self):
        rd = self.test_data.get_rd('cancm4_tasmax_2011')
        slc = [None, None, None, [0, 10], [0, 10]]
        ops_ocgis = OcgOperations(calc=[{'func': 'icclim_SU', 'name': 'SU'}], calc_grouping=['month'], slice=slc,
                                  dataset=rd, output_format='nc')
        ret = ops_ocgis.execute()
        with nc_scope(ret) as ds:
            to_test = deepcopy(ds.__dict__)
            history = to_test.pop('history')
            self.assertEqual(history[111:187],
                             ' Calculation of SU indice (monthly climatology) from 2011-1-1 to 2020-12-31.')
            self.assertDictEqual(to_test, OrderedDict([(u'source_data_global_attributes',
                                                        u'{"institution": "CCCma (Canadian Centre for Climate Modelling and Analysis, Victoria, BC, Canada)", "institute_id": "CCCma", "experiment_id": "decadal2010", "source": "CanCM4 2010 atmosphere: CanAM4 (AGCM15i, T63L35) ocean: CanOM4 (OGCM4.0, 256x192L40) sea ice: CanSIM1 (Cavitating Fluid, T63 Gaussian Grid) land: CLASS2.7", "model_id": "CanCM4", "forcing": "GHG,Oz,SA,BC,OC,LU,Sl,Vl (GHG includes CO2,CH4,N2O,CFC11,effective CFC12)", "parent_experiment_id": "N/A", "parent_experiment_rip": "N/A", "branch_time": 0.0, "contact": "cccma_info@ec.gc.ca", "references": "http://www.cccma.ec.gc.ca/models", "initialization_method": 1, "physics_version": 1, "tracking_id": "64384802-3f0f-4ab4-b569-697bd5430854", "branch_time_YMDH": "2011:01:01:00", "CCCma_runid": "DHFP1B_E002_I2011_M01", "CCCma_parent_runid": "DHFP1_E002", "CCCma_data_licence": "1) GRANT OF LICENCE - The Government of Canada (Environment Canada) is the \\nowner of all intellectual property rights (including copyright) that may exist in this Data \\nproduct. You (as \\"The Licensee\\") are hereby granted a non-exclusive, non-assignable, \\nnon-transferable unrestricted licence to use this data product for any purpose including \\nthe right to share these data with others and to make value-added and derivative \\nproducts from it. This licence is not a sale of any or all of the owner\'s rights.\\n2) NO WARRANTY - This Data product is provided \\"as-is\\"; it has not been designed or \\nprepared to meet the Licensee\'s particular requirements. Environment Canada makes no \\nwarranty, either express or implied, including but not limited to, warranties of \\nmerchantability and fitness for a particular purpose. In no event will Environment Canada \\nbe liable for any indirect, special, consequential or other damages attributed to the \\nLicensee\'s use of the Data product.", "product": "output", "experiment": "10- or 30-year run initialized in year 2010", "frequency": "day", "creation_date": "2012-03-28T15:32:08Z", "history": "2012-03-28T15:32:08Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.", "Conventions": "CF-1.4", "project_id": "CMIP5", "table_id": "Table day (28 March 2011) f9d6cfec5981bb8be1801b35a81002f0", "title": "CanCM4 model output prepared for CMIP5 10- or 30-year run initialized in year 2010", "parent_experiment": "N/A", "modeling_realm": "atmos", "realization": 2, "cmor_version": "2.8.0"}'),
                                                       (u'title', u'ECA heat indice SU'), (
                u'references', u'ATBD of the ECA indices calculation (http://eca.knmi.nl/documents/atbd.pdf)'), (
                                                       u'institution',
                                                       u'Climate impact portal (http://climate4impact.eu)'),
                                                       (u'comment', u' ')]))
            var = ds.variables['SU']
            to_test = dict(var.__dict__)
            self.assertEqual(to_test, {'_FillValue': 999999, u'units': u'days',
                                       u'standard_name': AbstractIcclimFunction.standard_name,
                                       u'long_name': 'Summer days (number of days where daily maximum temperature > 25 degrees)',
                                       'grid_mapping': 'latitude_longitude'})
    
    @attr('remote')
    def test_calculate_opendap(self):
        ## test against an opendap target ensuring icclim and ocgis operations
        ## are equivalent in the netcdf output
        url = 'http://opendap.nmdc.eu/knmi/thredds/dodsC/IS-ENES/TESTSETS/tasmax_day_EC-EARTH_rcp26_r8i1p1_20760101-21001231.nc'
        calc_grouping = ['month']
        rd = ocgis.RequestDataset(uri=url,variable='tasmax')
        
        calc_icclim = [{'func':'icclim_SU','name':'SU'}]
        ops = ocgis.OcgOperations(dataset=rd,calc=calc_icclim,calc_grouping=calc_grouping,
                                  output_format='nc',geom='state_boundaries',select_ugid=[10],
                                  prefix='icclim')
        ret_icclim = ops.execute()
        
        calc_ocgis = [{'func':'threshold','name':'SU','kwds':{'threshold':298.15,'operation':'gt'}}]
        ops = ocgis.OcgOperations(dataset=rd,calc=calc_ocgis,calc_grouping=calc_grouping,
                                  output_format='nc',geom='state_boundaries',select_ugid=[10],
                                  prefix='ocgis')
        ret_ocgis = ops.execute()
        
        ## variable and datasets will have different attributes, so adjust those
        ## before testing if the netCDFs are equal...
        with nc_scope(ret_icclim,'r') as ds_icclim:
            with nc_scope(ret_ocgis,'a') as ds_ocgis:
                ## strip the current attributes
                for key in ds_ocgis.ncattrs():
                    ds_ocgis.delncattr(key)
                for key in ds_ocgis.variables['SU'].ncattrs():
                    if not key.startswith('_'):
                        ds_ocgis.variables['SU'].delncattr(key)
                ## make equivalent attributes
                for key in ds_icclim.ncattrs():
                    setattr(ds_ocgis,key,getattr(ds_icclim,key))
                ## update the target variable attributes
                for key,value in ds_icclim.variables['SU'].__dict__.iteritems():
                    if not key.startswith('_'):
                        setattr(ds_ocgis.variables['SU'],key,value)
        
        self.assertNcEqual(ret_icclim,ret_ocgis)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()