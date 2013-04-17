import unittest
import sys
from nose.plugins.skip import SkipTest
sys.path.append('/home/local/WX/ben.koziol/links/git/examples')


raise(SkipTest(__name__))

class Test(unittest.TestCase):

    def test_nws(self):
        import nws
        nws.main()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()