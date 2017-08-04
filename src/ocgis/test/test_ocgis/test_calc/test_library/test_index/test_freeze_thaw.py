import numpy as np

import ocgis
from ocgis.calc.library.index.freeze_thaw import FreezeThaw, freezethaw1d
from ocgis.exc import UnitsValidationError
from ocgis.test.base import AbstractTestField


class TestFreezeThawCycles(AbstractTestField):

    def test_freezethaw1d(self):

        x = np.array([0, 7, -1, 8, 0, -15, 0])
        self.assertEquals(freezethaw1d(x, 15), 0)
        x = np.array([16, -15, 0])
        self.assertEquals(freezethaw1d(x, 15), 1)

        x = np.array([0, 15, 0, -15, 0])
        self.assertEquals(freezethaw1d(x, 15), 1)

        x = np.array([0, 15, -15, 0])
        self.assertEquals(freezethaw1d(x, 15), 1)

        x = np.array([0, 7, -1, 9, 0, -15, 0])
        self.assertEquals(freezethaw1d(x, 15), 1)

        x = np.array([-10, 15, 0, -15, 0])
        self.assertEquals(freezethaw1d(x, 15), 1)

        x = np.array([0, 1, 2, 3, 0, -1, 0, 1, -2, -3, 3])
        self.assertEquals(freezethaw1d(x, 2), 2)

        x = np.array([0, 1, 2, 3, 4])
        self.assertEquals(freezethaw1d(x, 2), 0)

        x = np.array([0, -2, 0, -2, 0, -2, 2, 0, 2, 0, -2])
        self.assertEquals(freezethaw1d(x, 2), 2)

        x = np.array([3, 4, 5, 2, 3, -3, 4, 5, -5, -6, -3, 0, -1, 4, 5, 2, -3, -5, 6])
        self.assertEquals(freezethaw1d(x, 2), 6)

        x = np.array([2, 3, 4, -1, 0, 2.5, 0, 0, -1, -1, -2, -3, -4, -5, 1, -5, -6, 4, 5, ])
        self.assertEquals(freezethaw1d(x, 2), 2)

        x = np.array([3, 4, 5, 2, 3, -3, 4, 5, -5, -6, -3, 0, -1, 4, 5, 2, -3,
                      -5, ])
        self.assertEquals(freezethaw1d(x, 2), 5)

    def test_execute(self):
        # Just a smoke test for the class.
        field = self.get_field(with_value=True, month_count=23, name='tas', units='K')

        tgd = field.temporal.get_grouping(['year'])
        dv = FreezeThaw(field=field, parms={'threshold': 20}, tgd=tgd)
        ret = dv.execute()
        shp_out = (2, 2, 2, 3, 4)
        self.assertEqual(ret['freezethaw'].get_value().shape, shp_out)
        self.assertNumpyAll(ret['freezethaw'].get_value()[:], np.zeros(shp_out))

    def test_units_check(self):
        """Check that using a variable with units other than Kelvin raises an
        error."""
        field = self.get_field(with_value=True, month_count=23, name='pr', units='mm')
        tgd = field.temporal.get_grouping(['year'])
        dv = FreezeThaw(field=field, parms={'threshold': 20}, tgd=tgd)
        self.assertRaises(ocgis.exc.UnitsValidationError, dv.execute)

    def test_unit_conversion(self):
        field = self.get_field(with_value=True, month_count=23, name='tas',
                               units='C')
        tgd = field.temporal.get_grouping(['year'])
        dv = FreezeThaw(field=field, parms={'threshold': 1}, tgd=tgd)
        retC = dv.execute()

        field['tas'].set_value(field['tas'].get_value() + 273.15)
        field['tas'].units = 'K'
        dv = FreezeThaw(field=field, parms={'threshold': 1}, tgd=tgd)
        retK = dv.execute()

        self.assertNumpyAll(retC['freezethaw'].get_value(), retK['freezethaw'].get_value())

    def test_missing(self):
        x = np.ma.masked_values([0,-1, 1, -1, 2, -2, 0], 2)
        self.assertEquals(freezethaw1d(x, 1), 2)

