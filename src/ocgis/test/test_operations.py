import unittest
import datetime
from ocgis.api.operations import OcgOperations


class TestOperations(unittest.TestCase):

    def test(self):
        uris = ['/tmp/foo1.nc','/tmp/foo2.nc']
        vars = ['tasmin','tasmax']
        time_range = [datetime.datetime(2000,1,1),datetime.datetime(2000,12,31)]
        
        datasets = [{'uri':uri,'variable':var} for uri,var in zip(uris,vars)]
        ops = OcgOperations()
        import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()