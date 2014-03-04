import unittest
from ocgis.test.base import TestBase
from ocgis.contrib.library_icclim import IcclimTG, IcclimSU,\
    AbstractIcclimFunction
from ocgis.calc.library.statistics import Mean
from ocgis.api.parms.definition import Calc
from ocgis.calc.library.register import FunctionRegistry, register_icclim
from ocgis.exc import DefinitionValidationError, UnitsValidationError
from ocgis.api.operations import OcgOperations
from ocgis.calc.library.thresholds import Threshold
from ocgis.test.test_simple.test_simple import ToTest, nc_scope
import ocgis
from ocgis.test.test_base import longrunning


class TestLibraryIcclim(TestBase):
    
    def test_register_icclim(self):
        fr = FunctionRegistry()
        self.assertNotIn('icclim_TG',fr)
        register_icclim(fr)
        self.assertIn('icclim_TG',fr)
    
    def test_calc_argument_to_operations(self):
        value = [{'func':'icclim_TG','name':'TG'}]
        calc = Calc(value)
        self.assertEqual(len(calc.value),1)
        self.assertEqual(calc.value[0]['ref'],IcclimTG)
        
    def test_bad_icclim_key_to_operations(self):
        value = [{'func':'icclim_TG_bad','name':'TG'}]
        with self.assertRaises(DefinitionValidationError):
            Calc(value)
            
            
class TestTG(TestBase):
            
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
            self.assertNumpyAll(ret_ocgis[1]['tas'].variables['mean_tas'].value,
                                ret_icclim[1]['tas'].variables['TG_tas'].value)
            
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
            var = ds.variables['TG_tas']
            to_test = dict(var.__dict__)
            to_test.pop('_FillValue')
            self.assertEqual(to_test,{u'units': u'K', u'standard_name': AbstractIcclimFunction.standard_name, u'long_name': u'Mean of daily mean temperature'})

    def test_calculate(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        field = field[:,:,:,0:10,0:10]
        for calc_grouping in [['month'],['month','year']]:
            tgd = field.temporal.get_grouping(calc_grouping)
            itg = IcclimTG(field=field,tgd=tgd)
            ret_icclim = itg.execute()
            mean = Mean(field=field,tgd=tgd)
            ret_ocgis = mean.execute()
            self.assertNumpyAll(ret_icclim['icclim_TG_tas'].value,ret_ocgis['mean_tas'].value)
            
            
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
            self.assertNumpyAll(ret_icclim['icclim_SU_tasmax'].value,ret_ocgis['threshold_tasmax'].value)
            
    def test_calculation_operations_bad_units(self):
        rd = self.test_data.get_rd('daymet_tmax')
        calc_icclim = [{'func':'icclim_SU','name':'SU'}]
        ops_icclim = OcgOperations(calc=calc_icclim,calc_grouping=['year'],dataset=rd)
        with self.assertRaises(UnitsValidationError):
            ops_icclim.execute()
    
    @longrunning
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
        with nc_scope(ret_icclim,'a') as ds_icclim:
            with nc_scope(ret_ocgis,'a') as ds_ocgis:
                ds_ocgis.variables['SU_tasmax'].setncatts(ds_icclim.variables['SU_tasmax'].__dict__)
                ds_ocgis.setncatts(ds_icclim.__dict__)
        
        self.assertNcEqual(ret_icclim,ret_ocgis)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()