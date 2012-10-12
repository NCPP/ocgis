import unittest
from ocg.calc import engine
from ocg.test import misc


class TestCalcEngine(unittest.TestCase):


    def test(self):
        niter = float('inf')
        grouping = ['year']
        raw = True
        for desc in misc.gen_descriptor_classes(niter=niter):
            if desc['calc'] is None: continue
            if desc['time_range'] is not None: continue
            for data in misc.gen_example_data(niter=niter):
                eng = engine.OcgCalculationEngine(
                       grouping,
                       data['timevec'],
                       desc['calc'],
                       raw=raw)
                coll = eng.execute(data)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()