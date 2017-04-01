from datetime import datetime as dt

import ocgis
from ocgis.base import get_variable_names, orphaned
from ocgis.calc.library.index.heat_index import HeatIndex
from ocgis.constants import TagNames
from ocgis.exc import UnitsValidationError
from ocgis.test.base import attr, AbstractTestField
from ocgis.variable.base import VariableCollection


class TestHeatIndex(AbstractTestField):
    @attr('data')
    def test_system_units_raise_exception(self):
        kwds = {'time_range': [dt(2011, 1, 1), dt(2011, 12, 31, 23, 59, 59)]}
        ds = [self.test_data.get_rd('cancm4_tasmax_2011', kwds=kwds), self.test_data.get_rd('cancm4_rhsmax', kwds=kwds)]
        calc = [{'func': 'heat_index', 'name': 'heat_index', 'kwds': {'tas': 'tasmax', 'rhs': 'rhsmax'}}]
        ops = ocgis.OcgOperations(dataset=ds, calc=calc, slice=[0, 0, 0, 0, 0])
        self.assertEqual(ops.calc_grouping, None)
        with self.assertRaises(UnitsValidationError):
            ops.execute()

    @attr('data', 'cfunits')
    def test_system_units_conform_to(self):
        ocgis.env.OVERWRITE = True
        kwds = {'time_range': [dt(2011, 1, 1), dt(2011, 12, 31, 23, 59, 59)]}
        ds = [self.test_data.get_rd('cancm4_tasmax_2011', kwds=kwds), self.test_data.get_rd('cancm4_rhsmax', kwds=kwds)]

        # Set the conform to units
        ds[0].conform_units_to = 'fahrenheit'
        ds[1].conform_units_to = 'percent'

        calc = [{'func': 'heat_index', 'name': 'heat_index', 'kwds': {'tas': 'tasmax', 'rhs': 'rhsmax'}}]
        select_ugid = [25]

        # Operations on entire data arrays
        ops = ocgis.OcgOperations(dataset=ds, calc=calc)
        self.assertEqual(ops.calc_grouping, None)
        ret = ops.execute()
        ref = ret.get_element()
        hiv = ref['heat_index']
        hi = hiv.get_value()
        self.assertEqual(hi.shape, (365, 64, 128))

        self.assertEqual(hiv.units, None)
        self.assertTrue(hiv.get_mask().any())

        # Try temporal grouping
        ops = ocgis.OcgOperations(dataset=ds, calc=calc, calc_grouping=['month'], geom='state_boundaries',
                                  select_ugid=select_ugid)
        ret = ops.execute()
        actual = ret.get_element()['heat_index'].shape
        self.assertEqual(actual, (12, 4, 4))

    def test_system_units_validation_wrong_units(self):
        # Heat index coefficients require the data be in specific units.
        field = self.get_field(name='tasmax', units='kelvin', with_value=True)
        field_rhs = self.get_field(name='rhsmax', units='percent', with_value=True)

        with orphaned(field_rhs['rhsmax']):
            field.add_variable(field_rhs['rhsmax'], is_data=True)

        self.assertEqual(set(get_variable_names(field.get_by_tag(TagNames.DATA_VARIABLES))),
                         {'tasmax', 'rhsmax'})
        hi = HeatIndex(field=field, parms={'tas': 'tasmax', 'rhs': 'rhsmax'})
        with self.assertRaises(UnitsValidationError):
            hi.execute()

    def test_system_units_validation_equal_units(self):
        # Heat index coefficients require the data be in specific units.
        field = self.get_field(name='tasmax', units='fahrenheit', with_value=True)
        field_rhs = self.get_field(name='rhsmax', units='percent', with_value=True)
        with orphaned(field_rhs['rhsmax']):
            field.add_variable(field_rhs['rhsmax'], is_data=True)
        self.assertEqual(set(get_variable_names(field.get_by_tag(TagNames.DATA_VARIABLES))),
                         set(['tasmax', 'rhsmax']))
        hi = HeatIndex(field=field, parms={'tas': 'tasmax', 'rhs': 'rhsmax'})
        vc = hi.execute()
        self.assertIsInstance(vc, VariableCollection)
