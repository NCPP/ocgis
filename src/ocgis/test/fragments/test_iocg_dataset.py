import unittest
from ocg.api.interp.iocg.dataset import OcgDataset
import datetime
from shapely.geometry.polygon import Polygon
import os.path
from ocg.util.helpers import get_shp_as_multi
from shapely import wkt


class TestDataset(unittest.TestCase):
    
    uri = ('/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/python/'
           'OpenClimateGIS/bin/climate_data/cmip5/'
           'albisccp_cfDay_CCSM4_1pctCO2_r2i1p1_00200101-00391231.nc')
    shp = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/python/OpenClimateGIS/bin/shp/watersheds_4326.shp'
    var_name = 'albisccp'

    def get_poly(self,n=1):
        ret = []
        gd = self.load_geom(uid_field='id')
        for ii in range(n):
            try:
                ret.append(gd[ii])
            except IndexError:
                break
        return(ret)
    
    def constructor(self):
        ds = OcgDataset(self.uri)
        return(ds)
    
    def load_geom(self,uid_field='objectid'):
        ret = get_shp_as_multi(self.shp,uid_field=uid_field)
        for ii in ret:
            ii['id'] = ii.pop(uid_field)
            ii['geom'] = wkt.loads(ii['geom'])
        return(ret)
        
    def test_subset(self):
        ds = self.constructor()
        polygons = self.get_poly(n=9999)
        for poly in polygons:
            coll = ds.subset(self.var_name,
                             polygon=poly['geom'],
                             time_range=None)
            import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()