import numpy
import netCDF4

import fiona
import osgeo
import shapely

from ocgis.test.base import TestBase


class TestVersions(TestBase):

    def test_cfunits(self):
        import cfunits

        self.assertEqual(cfunits.__version__, '0.9.6')

    def test_esmf(self):
        import ESMF

        self.assertEqual(ESMF.__release__, 'ESMF_6_3_0rp1')

    def test_fiona(self):
        self.assertEqual(fiona.__version__, '1.4.5')

    def test_icclim(self):
        import icclim

        self.assertEqual(icclim.__version__, '3.0')

    def test_netCDF4(self):
        self.assertEqual(netCDF4.__version__, '1.1.1')

    def test_numpy(self):
        self.assertEqual(numpy.__version__, '1.8.2')

    def test_osgeo(self):
        self.assertEqual(osgeo.__version__, '1.11.1')

    def test_rtree(self):
        import rtree

        self.assertEqual(rtree.__version__, '0.8.0')

    def test_shapely(self):
        self.assertEqual(shapely.__version__, '1.4.3')