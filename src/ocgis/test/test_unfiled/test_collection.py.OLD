import ocgis
import unittest
from ocgis.test.base import TestBase
import csv


class TestRawCollection(TestBase):
    
    def test_date_parts(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ret = ocgis.OcgOperations(dataset=rd,snippet=True,output_format='csv+',
                                  headers=['ugid','gid','year','month','day','value']).execute()
        with open(ret,'r') as f:
            reader = csv.reader(f)
            for row in reader:
                for value in ['YEAR','MONTH','DAY']:
                    self.assertIn(value,row)
                break
    
    
if __name__ == '__main__':
    unittest.main()