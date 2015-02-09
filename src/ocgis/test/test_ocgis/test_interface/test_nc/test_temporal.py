from ocgis.interface.nc.dimension import NcVectorDimension
from ocgis.interface.base.dimension.temporal import TemporalDimension, TemporalGroupDimension

from ocgis.test.base import TestBase
from ocgis.interface.nc.temporal import NcTemporalDimension


class TestNcTemporalDimension(TestBase):

    def test_init(self):
        ntd = NcTemporalDimension(value=[5])
        self.assertIsInstance(ntd, TemporalDimension)
        self.assertIsInstance(ntd, NcVectorDimension)

    def test_get_grouping(self):
        ntd = NcTemporalDimension(value=[5000., 5001.])
        tgd = ntd.get_grouping(['month'])
        self.assertIsInstance(tgd, TemporalGroupDimension)
