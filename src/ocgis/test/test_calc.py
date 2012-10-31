import unittest
import numpy as np
import ocgis.calc.wrap.library as library


class TestCalc(unittest.TestCase):

    def test_Mean(self):
        agg = True
        raw = True
        
        weights = None
        values = np.ones((12,2,4,4))
        values = np.ma.array(values,mask=False)
        
        out_shape = (1,2,1,1)
        
        mean = library.Mean(agg=agg,raw=raw,weights=weights)
        ret = mean.calculate(values,out_shape)
        
        import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()