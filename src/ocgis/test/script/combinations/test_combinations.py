import unittest
from ocgis.test.base import TestBase
import run


class TestCombinationRunner(TestBase):

    def test_get_parameters(self):
        cr = run.CombinationRunner()
        ps = cr.get_parameters()
        self.assertTrue(len(ps) > 0)
    
    def test_iter(self):
        cr = run.CombinationRunner(ops_only=False)
        for row in cr:
            pass