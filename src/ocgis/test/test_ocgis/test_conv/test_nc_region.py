from ocgis.test.base import TestBase
from ocgis.constants import OutputFormatName

class TestNcConverterRegion(TestBase):
    
    def test_simple(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, 
                output_format=OutputFormatName.NETCDF_REGION,
                geom=[self.germany, self.nebraska],
                aggregate=True, 
                spatial_operation='clip',
                calc=[{'func': 'mean', 'name': 'monthly_mean'}],
                calc_grouping=['month']
                                      )
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
