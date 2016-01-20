from datetime import datetime as dt

import ocgis
from ocgis.calc.library.index.heat_index import HeatIndex
from ocgis.exc import UnitsValidationError
from ocgis.interface.base.variable import VariableCollection
from ocgis.test.base import attr
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField


class TestHeatIndex(AbstractTestField):
    @attr('data')
    def test_units_raise_exception(self):
        kwds = {'time_range':[dt(2011,1,1),dt(2011,12,31,23,59,59)]}
        ds = [self.test_data.get_rd('cancm4_tasmax_2011',kwds=kwds),self.test_data.get_rd('cancm4_rhsmax',kwds=kwds)]
        calc = [{'func':'heat_index','name':'heat_index','kwds':{'tas':'tasmax','rhs':'rhsmax'}}]
        ops = ocgis.OcgOperations(dataset=ds,calc=calc,slice=[0,0,0,0,0])
        self.assertEqual(ops.calc_grouping,None)
        with self.assertRaises(UnitsValidationError):
            ops.execute()

    @attr('data')
    def test_units_conform_to(self):
        ocgis.env.OVERWRITE = True
        kwds = {'time_range':[dt(2011,1,1),dt(2011,12,31,23,59,59)]}
        ds = [self.test_data.get_rd('cancm4_tasmax_2011',kwds=kwds),self.test_data.get_rd('cancm4_rhsmax',kwds=kwds)]
        
        ## set the conform to units
        ds[0].conform_units_to = 'fahrenheit'
        ds[1].conform_units_to = 'percent'
        
        calc = [{'func':'heat_index','name':'heat_index','kwds':{'tas':'tasmax','rhs':'rhsmax'}}]
        select_ugid = [25]
        
        ## operations on entire data arrays
        ops = ocgis.OcgOperations(dataset=ds,calc=calc)
        self.assertEqual(ops.calc_grouping,None)
        ret = ops.execute()
        ref = ret[1]
        self.assertEqual(ref.keys(),['tasmax_rhsmax'])
        self.assertEqual(ref['tasmax_rhsmax'].variables.keys(),['heat_index'])
        hi = ref['tasmax_rhsmax'].variables['heat_index'].value
        self.assertEqual(hi.shape,(1,365,1,64,128))
        
        ## ensure the units are none
        self.assertEqual(ret[1]['tasmax_rhsmax'].variables['heat_index'].units,None)
        
        ## confirm no masked geometries
        self.assertFalse(ref['tasmax_rhsmax'].spatial.geom.point.value.mask.any())
        ## confirm some masked data in calculation output
        self.assertTrue(hi.mask.any())
                
        # try temporal grouping
        ops = ocgis.OcgOperations(dataset=ds,calc=calc,calc_grouping=['month'],geom='state_boundaries',select_ugid=select_ugid)
        ret = ops.execute()
        self.assertEqual(ret[25]['tasmax_rhsmax'].variables['heat_index'].value.shape,(1,12,1,5,4))
        
    def test_units_validation_wrong_units(self):
        ## heat index coefficients require the data be in specific units
        field = self.get_field(name='tasmax',units='kelvin',with_value=True)
        field_rhs = self.get_field(name='rhsmax',units='percent',with_value=True)
        field.variables.add_variable(field_rhs.variables['rhsmax'], assign_new_uid=True)
        self.assertEqual(set(field.variables.keys()),set(['tasmax','rhsmax']))
        hi = HeatIndex(field=field,parms={'tas':'tasmax','rhs':'rhsmax'})
        with self.assertRaises(UnitsValidationError):
            hi.execute()
            
    def test_units_validation_equal_units(self):
        ## heat index coefficients require the data be in specific units
        field = self.get_field(name='tasmax',units='fahrenheit',with_value=True)
        field_rhs = self.get_field(name='rhsmax',units='percent',with_value=True)
        field.variables.add_variable(field_rhs.variables['rhsmax'], assign_new_uid=True)
        self.assertEqual(set(field.variables.keys()),set(['tasmax','rhsmax']))
        hi = HeatIndex(field=field,parms={'tas':'tasmax','rhs':'rhsmax'})
        vc = hi.execute()
        self.assertIsInstance(vc,VariableCollection)
