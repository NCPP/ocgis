import unittest
from ocgis.api.operations import OcgOperations
from datetime import datetime as dt
import numpy as np
from ocgis.exc import DefinitionValidationError


class TestOperations(unittest.TestCase):
    uris = ['/tmp/foo1.nc','/tmp/foo2.nc','/tmp/foo3.nc']
    vars = ['tasmin','tasmax','tas']
    time_range = [dt(2000,1,1),dt(2000,12,31)]
    level_range = 2
    datasets = [{'uri':uri,'variable':var} for uri,var in zip(uris,vars)]

    def test_equal_length_level_range(self):
        
        ## check that parms come out equal lengths and level range is correctly
        ## set.
        ops = OcgOperations(dataset=self.datasets,
                            time_range=self.time_range,
                            level_range=self.level_range)
        self.assertEqual(len(ops.dataset),len(ops.time_range))
        self.assertEqual(len(ops.dataset),len(ops.level_range))
        self.assertEqual(np.array(ops.level_range,dtype=int).max(),self.level_range)
        
        ## assert error is raised with arguments of differing lengths
        level_range = [[2,2],[3,3]]
        with self.assertRaises(DefinitionValidationError):
            ops = OcgOperations(dataset=self.datasets,
                                time_range=self.time_range,
                                level_range=level_range)

    def test_iter(self):
        ops = OcgOperations(dataset=self.datasets,
                            time_range=self.time_range,
                            level_range=self.level_range)
        for row in ops:
            self.assertEqual(row['time_range'],self.time_range)
            self.assertEqual(row['level_range'],[self.level_range,self.level_range])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()