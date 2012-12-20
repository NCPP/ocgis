import unittest
from ocgis.api.operations import OcgOperations
from datetime import datetime as dt
import numpy as np
from ocgis.exc import DefinitionValidationError


class TestOperations(unittest.TestCase):

    def test(self):
        uris = ['/tmp/foo1.nc','/tmp/foo2.nc','/tmp/foo3.nc']
        vars = ['tasmin','tasmax','tas']
        time_range = [dt(2000,1,1),dt(2000,12,31)]
        level_range = 2
    
        datasets = [{'uri':uri,'variable':var} for uri,var in zip(uris,vars)]
        
        ## check that parms come out equal lengths and level range is correctly
        ## set.
        ops = OcgOperations(dataset=datasets,time_range=time_range,level_range=level_range)
        self.assertEqual(len(ops.dataset),len(ops.time_range))
        self.assertEqual(len(ops.dataset),len(ops.level_range))
        self.assertEqual(np.array(ops.level_range,dtype=int).max(),level_range)
        
        ## assert error is raised with arguments of differing lengths
        level_range = [[2,2],[3,3]]
        with self.assertRaises(DefinitionValidationError):
            ops = OcgOperations(dataset=datasets,
                                time_range=time_range,
                                level_range=level_range)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()