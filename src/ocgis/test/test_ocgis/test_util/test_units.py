import numpy as np

from ocgis.test.base import TestBase, attr
from ocgis.util.units import get_are_units_equivalent, get_are_units_equal, get_are_units_equal_by_string_or_cfunits, \
    get_units_object, get_conformed_units


@attr('cfunits')
class Test(TestBase):
    _create_dir = False

    def test_get_are_units_equivalent(self):
        units = [get_units_object('celsius'), get_units_object('kelvin'), get_units_object('fahrenheit')]
        self.assertTrue(get_are_units_equivalent(units))

        units = [get_units_object('celsius'), get_units_object('kelvin'), get_units_object('coulomb')]
        self.assertFalse(get_are_units_equivalent(units))

        units = [get_units_object('celsius')]
        with self.assertRaises(ValueError):
            get_are_units_equivalent(units)

    def test_get_are_units_equal(self):
        units = [get_units_object('celsius'), get_units_object('kelvin'), get_units_object('fahrenheit')]
        self.assertFalse(get_are_units_equal(units))

        units = [get_units_object('celsius'), get_units_object('celsius'), get_units_object('celsius')]
        self.assertTrue(get_are_units_equal(units))

        units = [get_units_object('celsius')]
        with self.assertRaises(ValueError):
            get_are_units_equal(units)

    def test_get_are_units_equal_by_string_or_cfunits(self):
        _try_cfunits = [True, False]

        source = 'K'
        target = 'K'
        for try_cfunits in _try_cfunits:
            match = get_are_units_equal_by_string_or_cfunits(source, target, try_cfunits=try_cfunits)
            self.assertTrue(match)

        source = 'K'
        target = 'Kelvin'
        for try_cfunits in _try_cfunits:
            match = get_are_units_equal_by_string_or_cfunits(source, target, try_cfunits=try_cfunits)
            # CF units packages will allow comparison of abbreviations.
            if try_cfunits:
                self.assertTrue(match)
            else:
                self.assertFalse(match)

    def test_get_conformed_units(self):
        value = np.array([5, 6, 7])
        res = get_conformed_units(value, 'celsius', 'fahrenheit')
        try:
            import cf_units
            # Test original values are unchanged.
            self.assertEqual(np.mean(value), 6)
            test_value = res
        except ImportError:
            import cfunits
            # Test inplace conversion is used.
            test_value = value
        # Test values are conformed.
        self.assertAlmostEqual(np.mean(test_value), 42.799999999999876)
