from ocgis import CoordinateReferenceSystem
from ocgis.test.base import TestBase


class TestDependencies(TestBase):

    def test_osr(self):
        crs = CoordinateReferenceSystem(epsg=4326)
        self.assertNotEqual(crs.value, {})
