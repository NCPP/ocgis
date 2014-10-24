from copy import deepcopy
from ocgis.api.request.driver.base import AbstractDriver
from ocgis.api.request.driver.nc import DriverNetcdf
from ocgis.interface.base.crs import CFWGS84
from ocgis.test.base import TestBase


class TestAbstractDriver(TestBase):

    def test_get_field(self):
        # test updating of regrid source flag
        rd = self.test_data.get_rd('cancm4_tas')
        driver = DriverNetcdf(rd)
        field = driver.get_field()
        self.assertTrue(field._should_regrid)
        rd.regrid_source = False
        driver = DriverNetcdf(rd)
        field = driver.get_field()
        self.assertFalse(field._should_regrid)

        # test flag with an assigned coordinate system
        rd = self.test_data.get_rd('cancm4_tas')
        driver = DriverNetcdf(rd)
        field = driver.get_field()
        self.assertFalse(field._has_assigned_coordinate_system)
        rd = self.test_data.get_rd('cancm4_tas', kwds={'crs': CFWGS84()})
        driver = DriverNetcdf(rd)
        field = driver.get_field()
        self.assertTrue(field._has_assigned_coordinate_system)

    def test_eq(self):
        rd = self.test_data.get_rd('cancm4_tas')
        d = DriverNetcdf(rd)
        d2 = deepcopy(d)
        self.assertEqual(d, deepcopy(d))

        d2.key = 'bad'
        self.assertNotEqual(d, d2)