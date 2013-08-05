import unittest
from ocgis.test.base import TestBase
import ocgis


class Test(TestBase):

    def test_geodataportal_prism(self):
        uri = 'http://cida.usgs.gov/thredds/dodsC/prism'
        for variable in ['tmx','tmn','ppt']:
#            ocgis.env.VERBOSE = True
#            ocgis.env.DEBUG = True
            rd = ocgis.RequestDataset(uri,variable,t_calendar='standard')
#            dct = rd.inspect_as_dct()
            ops = ocgis.OcgOperations(dataset=rd,geom='state_boundaries',select_ugid=[25],
                                      snippet=True,output_format='numpy',aggregate=False,
                                      prefix=variable)
            ret = ops.execute()
#            print(ret)
            self.assertEqual(ret[25].variables[variable].value.shape,(1,1,227,246))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()