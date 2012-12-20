import unittest
import numpy as np
from ocgis.util.helpers import iter_array


class Test(unittest.TestCase):

    def test_iter_array(self):
        values = np.random.rand(2,2,4,4)
        mask = np.random.random_integers(0,1,values.shape)
        values = np.ma.array(values,mask=mask)
        for idx in iter_array(values):
            self.assertFalse(values.mask[idx])
        self.assertEqual(len(list(iter_array(values,use_mask=True))),len(values.compressed()))
        self.assertEqual(len(list(iter_array(values,use_mask=False))),len(values.data.flatten()))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()