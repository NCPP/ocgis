import unittest
from ocgis.util.justify import justify_row


class Test(unittest.TestCase):

    def test_justify_row(self):
        row = "Food is any substance[1] consumed to provide nutritional support for the body. It is usually of plant or animal origin, and contains essential nutrients, such as carbohydrates, fats, proteins, vitamins, or minerals. The substance is ingested by an organism and assimilated by the organism's cells in an effort to produce energy, maintain life, or stimulate growth."
        eq = ['    Food is any substance[1] consumed to provide nutritional support for the',
              '    body. It is usually of plant or animal origin, and contains essential',
              '    nutrients, such as carbohydrates, fats, proteins, vitamins, or minerals.',
              "    The substance is ingested by an organism and assimilated by the organism's",
              '    cells in an effort to produce energy, maintain life, or stimulate growth.']
        justified = justify_row(row)
        self.assertEqual(justified,eq)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()