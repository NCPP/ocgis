from cfunits.cfunits import Units

from ocgis.test.base import TestBase
from ocgis.util.units import get_are_units_equivalent, get_are_units_equal, get_are_units_equal_by_string_or_cfunits


class TestUnits(TestBase):
    _create_dir = False

    def test_get_are_units_equivalent(self):
        units = [Units('celsius'), Units('kelvin'), Units('fahrenheit')]
        self.assertTrue(get_are_units_equivalent(units))

        units = [Units('celsius'), Units('kelvin'), Units('coulomb')]
        self.assertFalse(get_are_units_equivalent(units))

        units = [Units('celsius')]
        with self.assertRaises(ValueError):
            get_are_units_equivalent(units)

    def test_get_are_units_equal(self):
        units = [Units('celsius'), Units('kelvin'), Units('fahrenheit')]
        self.assertFalse(get_are_units_equal(units))

        units = [Units('celsius'), Units('celsius'), Units('celsius')]
        self.assertTrue(get_are_units_equal(units))

        units = [Units('celsius')]
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
            # cfunits.Units will allow comparison of abbreviated and full name form while string comparison will not
            if try_cfunits:
                self.assertTrue(match)
            else:
                self.assertFalse(match)
