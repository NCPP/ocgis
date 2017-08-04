from datetime import datetime as dt
import numpy as np

import ocgis
from ocgis.base import get_variable_names, orphaned
from ocgis.calc.library.index.freeze_thaw import FreezeThawCycles, freezethaw1d
from ocgis.constants import TagName
from ocgis.exc import UnitsValidationError
from ocgis.test.base import attr, AbstractTestField
from ocgis.variable.base import VariableCollection

from ocgis import RequestDataset, OcgOperations


class TestFreezeThawCycles(AbstractTestField):


    def test_freezethaw1d(self):
        x = np.array([0, 15, 0, -15, 0]) + 273.15
        self.assertEquals(freezethaw1d(x, 15), 1)

        x = np.array([0, 15, -15, 0]) + 273.15
        self.assertEquals(freezethaw1d(x, 15), 1)

        x = np.array([0, 7, -1, 8, 0, -15, 0]) + 273.15
        self.assertEquals(freezethaw1d(x, 15), 0)

        x = np.array([0, 7, -1, 9, 0, -15, 0]) + 273.15
        self.assertEquals(freezethaw1d(x, 15), 1)

        x = np.array([-10, 15, 0, -15, 0]) + 273.15
        self.assertEquals(freezethaw1d(x, 15), 1)

        x = np.array([0, 1, 2, 3, 0, -1, 0, 1, -2, -3, 3]) + 273.15
        self.assertEquals(freezethaw1d(x, 2), 2)

        x = np.array([0, 1, 2, 3, 4]) + 273.15
        self.assertEquals(freezethaw1d(x, 2), 0)

        x = np.array([0, -2, 0, -2, 0, -2, 2, 0, 2, 0, -2]) + 273.15
        self.assertEquals(freezethaw1d(x, 2), 2)


        x = np.array([3, 4, 5, 2, 3, -3, 4, 5, -5, -6, -3, 0, -1, 4, 5, 2, -3, -5, 6]) + 273.15
        self.assertEquals(freezethaw1d(x, 2), 6)

        x = np.array([2, 3, 4, -1, 0, 2.5, 0, 0, -1, -1, -2, -3, -4, -5, 1, -5, -6, 4, 5, ]) + 273.15
        self.assertEquals(freezethaw1d(x, 2), 2)

        x = np.array([3, 4, 5, 2, 3, -3, 4, 5, -5, -6, -3, 0, -1, 4, 5, 2, -3,
                      -5, ]) + 273.15
        self.assertEquals(freezethaw1d(x, 2), 5)


    def test_execute(self):
        field = self.get_field(with_value=True, month_count=23, name='tas')

        grouping = ['year']
        tgd = field.temporal.get_grouping(grouping)
        dv = FreezeThawCycles(field=field, parms={'threshold': 20}, tgd=tgd)
        ret = dv.execute()
        shp_out = (2, 2, 2, 3, 4)
        self.assertEqual(ret['freezethawcycles'].get_value().shape, shp_out)
        self.assertNumpyAll(ret['freezethawcycles'].get_value()[:], np.zeros(shp_out))
