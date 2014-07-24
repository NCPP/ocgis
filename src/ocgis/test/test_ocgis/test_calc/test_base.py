from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
from ocgis.calc.base import AbstractUnivariateFunction,\
    AbstractUnivariateSetFunction, AbstractFunction
from ocgis import constants
from cfunits.cfunits import Units
from ocgis.exc import UnitsValidationError
import numpy as np


class FooNeedsUnits(AbstractUnivariateFunction):
    description = 'calculation with units'
    dtype = constants.np_float
    key = 'fnu'
    required_units = ['K','kelvin']
    standard_name = 'foo_needs_units'
    long_name = 'Foo Needs Units'
    
    def calculate(self,values):
        return(values)
            
            
class FooNeedsUnitsSet(AbstractUnivariateSetFunction):
    description = 'calculation with units'
    dtype = constants.np_float
    key = 'fnu'
    required_units = ['K','kelvin']
    standard_name = ''
    long_name = ''
    
    def calculate(self,values):
        return(np.ma.mean(values,axis=0))


class TestAbstractFunction(AbstractTestField):

    def test_execute_meta_attrs(self):
        """Test overloaded metadata attributes are appropriately applied."""

        for has_meta in [True, False]:
            field = self.get_field(with_value=True)
            if has_meta:
                field.meta = {'attrs': 'already has something'}
            else:
                field.meta = {}
            for oload in [True, False]:
                if oload:
                    meta_attrs = {'something_new': 'is about to happen', 'standard_name': 'never!'}
                else:
                    meta_attrs = {'something_new': 'is about to happen'}
                fb = FooNeedsUnits(field=field, meta_attrs=meta_attrs)
                ret = fb.execute()
                if oload:
                    actual = {'attrs': {'long_name': 'Foo Needs Units', 'standard_name': 'never!',
                                         'something_new': 'is about to happen'}}
                else:
                    actual = {'attrs': {'long_name': 'Foo Needs Units', 'standard_name': 'foo_needs_units',
                                         'something_new': 'is about to happen'}}
                self.assertEqual(ret['fnu'].meta, actual)
                if oload:
                    self.assertDictEqual(meta_attrs, {'something_new': 'is about to happen', 'standard_name': 'never!'})
                else:
                    self.assertDictEqual(meta_attrs, {'something_new': 'is about to happen'})


            
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
