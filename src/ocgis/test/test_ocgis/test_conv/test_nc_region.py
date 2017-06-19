import ocgis
from ocgis.test.base import TestBase
from ocgis.constants import OutputFormatName

from ocgis.test.strings import GERMANY_WKT, NEBRASKA_WKT
from shapely import wkt

ocgis.env.OVERWRITE = True
class TestNcConverterRegion():
    
    def test_simple(self):
        #rd = self.test_data.get_rd('cancm4_tas')
        rd = ocgis.RequestDataset('/home/david/data/cmip5/tas_Amon_GFDL-CM3_historical_r1i1p1_193501-193912.nc') 
        ops = ocgis.OcgOperations(dataset=rd, 
                output_format=OutputFormatName.NETCDF_REGION,
                output_format_options={'data_model':'NETCDF4'},
                geom=[self.germany, self.nebraska],
                aggregate=True, 
                spatial_operation='clip',
                calc=[{'func': 'mean', 'name': 'monthly_mean'}],
                calc_grouping=['month'],
                dir_output='/tmp'
                                      )
        return ops
        coll = ops.execute()


    @property
    def germany(self):
        germany = self.get_buffered(wkt.loads(GERMANY_WKT))
        germany = {'geom': germany, 'properties': {'UGID': 2, 'DESC': 'Germany'}}
        return germany

    @property
    def nebraska(self):
        nebraska = self.get_buffered(wkt.loads(NEBRASKA_WKT))
        nebraska = {'geom': nebraska, 'properties': {'UGID': 1, 'DESC': 'Nebraska'}}
        return nebraska

    def get_buffered(self, geom):
        ret = geom.buffer(0)
#        self.assertTrue(ret.is_valid)
        return ret

ops = TestNcConverterRegion().test_simple()
ret = ops.execute()