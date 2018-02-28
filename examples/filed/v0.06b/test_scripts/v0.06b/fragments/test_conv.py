import unittest

from ocg.calc import engine
from ocg.conv.csv_ import CsvConverter
from ocg.test import misc


class TestCsvConverter(unittest.TestCase):

    def test_raw(self):
        niter = 1
        grouping = ['year']
        raw = False
        for desc in misc.gen_descriptor_classes(niter=niter):
            for data in misc.gen_example_data(niter=niter):
                eng = engine.OcgCalculationEngine(
                    grouping,
                    data['tid'],
                    data['timevec'],
                    desc['calc'],
                    raw=raw)
                #                coll = eng.execute(data)
                conv = CsvConverter(desc, data, base_name='ocg', wd=None, conn=None,
                                    cengine=eng, write_attr=False,
                                    wkt=False,
                                    wkb=False)
                conv.run()
        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
