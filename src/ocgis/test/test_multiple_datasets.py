import unittest
from ocgis.util.shp_cabinet import ShpCabinet
from ocgis.util.inspect import Inspect
from ocgis.api.operations import OcgOperations
from ocgis.api.interpreter import OcgInterpreter


class Test(unittest.TestCase):

    def test(self):
        maurer = {'uri':'/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/climate_data/maurer/bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc','variable':'Prcp'}
        cancm4 = {'uri':'/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc','variable':'tasmax'}
#        ip = Inspect(maurer,variable='Prcp')
        
        dataset = [
                   maurer,
#                   cancm4
                   ]
        snippet = True
        geom = self.california
        ops = OcgOperations(dataset=dataset,snippet=True,geom=geom)
        ret = OcgInterpreter(ops).execute()
        import ipdb;ipdb.set_trace()
        
    @property
    def california(self):
        sc = ShpCabinet()
        ret = sc.get_geoms('state_boundaries',{'ugid':[25]})
        return(ret)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()