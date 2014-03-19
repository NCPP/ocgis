from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
from ocgis.calc.base import AbstractUnivariateFunction,\
    AbstractUnivariateSetFunction
from ocgis import constants
from cfunits.cfunits import Units
from ocgis.exc import UnitsValidationError
import numpy as np


class FooNeedsUnits(AbstractUnivariateFunction):
    description = 'calculation with units'
    dtype = constants.np_float
    key = 'fnu'
    required_units = ['K','kelvin']
    
    def calculate(self,values):
        return(values)
            
            
class FooNeedsUnitsSet(AbstractUnivariateSetFunction):
    description = 'calculation with units'
    dtype = constants.np_float
    key = 'fnu'
    required_units = ['K','kelvin']
    
    def calculate(self,values):
        return(np.ma.mean(values,axis=0))
            
            
class TestAbstractUnivariateFunction(AbstractTestField):
    
    def test_validate_units(self):
        field = self.get_field(with_value=True)
        fnu = FooNeedsUnits(field=field)
        ret = fnu.execute()
        self.assertNumpyAll(field.variables['tmax'].value.astype(FooNeedsUnits.dtype),
                            ret['fnu'].value)
        
    def test_validate_units_bad_units(self):
        field = self.get_field(with_value=True)
        field.variables['tmax'].units = 'celsius'
        self.assertEqual(field.variables['tmax'].cfunits,Units('celsius'))
        fnu = FooNeedsUnits(field=field)
        with self.assertRaises(UnitsValidationError):
            fnu.execute()
            
            
class TestAbstractUnivariateSetFunction(AbstractTestField):
    
    def test_validate_units(self):
        field = self.get_field(with_value=True)
        tgd = field.temporal.get_grouping(['month'])
        fnu = FooNeedsUnitsSet(field=field,tgd=tgd)
        fnu.execute()
        
    def test_validate_units_bad_units(self):
        field = self.get_field(with_value=True)
        tgd = field.temporal.get_grouping(['month'])
        field.variables['tmax'].units = 'celsius'
        self.assertEqual(field.variables['tmax'].cfunits,Units('celsius'))
        fnu = FooNeedsUnitsSet(field=field,tgd=tgd)
        with self.assertRaises(UnitsValidationError):
            fnu.execute()
