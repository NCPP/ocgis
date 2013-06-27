import ocgis
from ocgis.test.base import TestBase
import itertools


class Test(TestBase):
    
    def test_time_region_subset(self):
        
        _month = [[6,7],[12],None,[1,3,8]]
        _year = [[2011],None,[2012],[2011,2013]]
        
        def run_test(month,year):
            rd = self.test_data.get_rd('cancm4_rhs')
            time_region = {'month':month,'year':year}
            rd.time_region = time_region
            
            ops = ocgis.OcgOperations(dataset=rd,geom='state_boundaries',
                                      select_ugid=[25])
            ret = ops.execute()
            
            ret = ret[25].variables['rhs'].temporal.value
            
            years = [dt.year for dt in ret.flat]
            months = [dt.month for dt in ret.flat]
            
            if year is not None:
                self.assertEqual(set(years),set(year))
            if month is not None:
                self.assertEqual(set(months),set(month))
            
        for month,year in itertools.product(_month,_year):
            run_test(month,year)