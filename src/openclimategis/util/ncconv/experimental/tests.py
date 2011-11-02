import unittest
from ocg_dataset import OcgDataset
from shapely.geometry.point import Point
import ipdb
import datetime


class TestOcgDataset(unittest.TestCase):
    nc_path = '/home/bkoziol/git/OpenClimateGIS/bin/climate_data/wcrp_cmip3/pcmdi.ipcc4.bccr_bcm2_0.1pctto2x.run1.monthly.cl_A1_1.nc'
    nc_opts = dict(rowbnds_name='lat_bnds',
                   colbnds_name='lon_bnds',
                   calendar='gregorian',
                   time_units='days since 1800-1-1 00:00:0.0',
                   level_name='lev')
    _od = None
    
    @property
    def od(self):
        if self._od is None:
            self._od = OcgDataset(self.nc_path,**self.nc_opts)
        return(self._od)

    def get_igeom(self):
        pt = Point(200,0.0)
        return(pt.buffer(8,1))

    def test_constructor(self):
        self.assertEqual(self.od.res,2.8125)
        
    def test_subset(self):
        polygon = self.get_igeom()
        
        ## three time periods. two levels.
        sub = self.od.subset('cl',
                             polygon=polygon,
                             time_range=[datetime.datetime(2000,10,1),datetime.datetime(2000,12,31)],
                             level_range=[1,2])
        
        ## one time periods. one level.
        sub = self.od.subset('cl',
                             polygon=polygon,
                             time_range=[datetime.datetime(2000,10,1),datetime.datetime(2000,10,31)],
                             level_range=[1,1])

        ## one time periods. two levels.
        sub = self.od.subset('cl',
                             polygon=polygon,
                             time_range=[datetime.datetime(2000,10,1),datetime.datetime(2000,10,31)],
                             level_range=[1,2])
        
        ## one time period. no levels.
        sub = self.od.subset('cl',
                             polygon=polygon,
                             time_range=[datetime.datetime(2000,10,1),datetime.datetime(2000,10,31)])

        ## three time periods. no levels.
        sub = self.od.subset('cl',
                             polygon=polygon,
                             time_range=[datetime.datetime(2000,10,1),datetime.datetime(2000,12,31)])

#        sub.display(overlays=[polygon])

    def test_subset_clip(self):
        polygon = self.get_igeom()
        
        ## three time periods. two levels.
        sub = self.od.subset('cl',
                             polygon=polygon,
                             time_range=[datetime.datetime(2000,10,1),datetime.datetime(2000,12,31)],
                             level_range=[1,2])
        sub.clip(polygon)
        self.assertTrue((sub.weight < 1.0).any())
#        ipdb.set_trace()
#        sub.display(overlays=[polygon])

    def test_subset_union(self):
        polygon = self.get_igeom()
        
        ## three time periods. two levels.
        sub = self.od.subset('cl',
                             polygon=polygon,
                             time_range=[datetime.datetime(2000,10,1),datetime.datetime(2000,12,31)],
                             level_range=[1,2])
        sub.clip(polygon)
        sub.union()
#        ipdb.set_trace()
        sub.display(overlays=[polygon])
        
    def test_motherlode(self):
        sub = self.od.subset('ps',
                             time_range=[datetime.datetime(2000,10,1),datetime.datetime(2000,10,31)])
        sub.display()
        ipdb.set_trace()
        


if __name__ == "__main__":
    import sys;sys.argv = ['', 'TestOcgDataset.test_motherlode']
    unittest.main()