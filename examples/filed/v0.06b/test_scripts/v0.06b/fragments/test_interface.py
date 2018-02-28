import unittest

import netCDF4 as nc
import ocgis.meta.interface.interface as interface
import ocgis.meta.interface.models as models


class TestInterface(unittest.TestCase):
    uri = '/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc'
    var_name = 'albisccp'

    @property
    def dataset(self):
        return (nc.Dataset(self.uri, 'r'))

    def test_GlobalInterface(self):
        iface = interface.GlobalInterface(self.dataset)

    def test_SpatialInterfacePolygon(self):
        name_map = {models.RowBounds: 'lat_bnds'}
        isp = interface.SpatialInterfacePolygon(self.dataset, name_map=name_map)


if __name__ == "__main__":
    #    import sys;sys.argv = ['', 'TestInterface.test_SpatialInterfacePolygon']
    unittest.main()
