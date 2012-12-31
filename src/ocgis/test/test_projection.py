import unittest
from netCDF4 import Dataset
from ocgis.interface.projection import get_projection
from ocgis import Inspect
from ocgis.api.operations import OcgOperations


class Test(unittest.TestCase):
    hostetler = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/climate_data/hostetler/RegCM3_Daily_srm_GFDL.ncml.nc'
    
    def test_get_projection(self):
        ip = Inspect(self.hostetler,variable='TG',interface_overload={'t_calendar':'noleap'})
        import ipdb;ipdb.set_trace()
        
        ds = Dataset(self.hostetler)
        proj = get_projection(ds)
        self.assertEqual(proj.sr.ExportToProj4(),'+proj=lcc +lat_1=30 +lat_2=60 +lat_0=35 +lon_0=-102.300003052 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ')

    def test_shp(self):
        ds = {'uri':self.hostetler,'variable':'TG'}
        iface = {'t_calendar':'noleap'}
        ops = OcgOperations(dataset=ds,output_format='shp',snippet=True,
                            interface=iface)
        ops.execute()


if __name__ == "__main__":
#    import sys;sys.argv = ['', 'TestSimpleMultiCalc01.test']
    unittest.main()