import unittest

import misc


class TestMisc(unittest.TestCase):

    def test_gen_example_data(self):
        for data in misc.gen_example_data(niter=1):
            pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
