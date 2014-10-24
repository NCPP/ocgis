import unittest
from ocgis.test.base import TestBase
import ocgis
from ocgis.exc import DefinitionValidationError


class Test(TestBase):

    def test_nc(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_rhsmax')
        rd = [rd1,rd2]
        for output_format in ['shp','csv','csv+','nc']:
            if output_format == 'nc':
                with self.assertRaises(DefinitionValidationError):
                    ops = ocgis.OcgOperations(dataset=rd,output_format=output_format,
                                              geom='state_boundaries',select_ugid=[25],
                                              snippet=True,prefix=output_format)
            else:
                ops = ocgis.OcgOperations(dataset=rd,output_format=output_format,
                                              geom='state_boundaries',select_ugid=[25],
                                              snippet=True,prefix=output_format)
                ret = ops.execute()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()