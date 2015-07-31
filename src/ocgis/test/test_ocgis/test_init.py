from ocgis import GeomCabinet, GeomCabinetIterator
from ocgis.test.base import TestBase


class TestInit(TestBase):
    def test(self):
        from ocgis import ShpCabinet, ShpCabinetIterator

        self.assertEqual(ShpCabinet.__bases__, (GeomCabinet,))
        self.assertEqual(ShpCabinetIterator.__bases__, (GeomCabinetIterator,))
