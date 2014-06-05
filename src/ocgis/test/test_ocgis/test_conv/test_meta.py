from ocgis import OcgOperations
from ocgis.conv.meta import MetaConverter
from ocgis.test.base import TestBase


class TestMetaConverter(TestBase):

    def test_init(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)
        conv = MetaConverter(ops)
        self.assertTrue(len(conv.write()) > 4000)
