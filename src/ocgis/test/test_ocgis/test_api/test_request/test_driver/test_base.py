from copy import deepcopy
from ocgis.api.request.driver.nc import DriverNetcdf
from ocgis.test.base import TestBase


class TestAbstractDriver(TestBase):

    def test_eq(self):
        rd = self.test_data.get_rd('cancm4_tas')
        d = DriverNetcdf(rd)
        d2 = deepcopy(d)
        self.assertEqual(d, deepcopy(d))

        d2.key = 'bad'
        self.assertNotEqual(d, d2)