import unittest
from netCDF4 import Dataset
from ocgis.interface.projection import get_projection, WGS84
from ocgis import Inspect
from ocgis.api.operations import OcgOperations
from ocgis.util.shp_cabinet import ShpCabinet
from ocgis.api.dataset.dataset import OcgDataset


class Test(unittest.TestCase):
    hostetler = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/climate_data/hostetler/RegCM3_Daily_srm_GFDL.ncml.nc'
    
    def test_get_projection(self):
        ip = Inspect(self.hostetler,variable='TG',interface_overload={'t_calendar':'noleap'})
        ip.__repr__()
        
        ds = Dataset(self.hostetler)
        proj = get_projection(ds)
        self.assertEqual(proj.sr.ExportToProj4(),'+proj=lcc +lat_1=30 +lat_2=60 +lat_0=35 +lon_0=-102.300003052 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ')

    def test_shp(self):
        ds = {'uri':self.hostetler,'variable':'TG'}
        iface = {'t_calendar':'noleap'}
        
        proj = get_projection(Dataset(self.hostetler))
#        sc = ShpCabinet()
        
#        sr = WGS84().sr
        
#        geoms = sc.get_geoms('state_boundaries')
#        projected = proj.project_to_match(geoms)
#        sc.write(projected,'/tmp/out.shp',sr=proj.sr)
        
        ops = OcgOperations(dataset=ds,output_format='shp',snippet=True,
                            interface=iface)
        ops.execute()

if __name__ == "__main__":
#    import sys;sys.argv = ['', 'TestSimpleMultiCalc01.test']
    unittest.main()