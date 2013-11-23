import unittest
from ocgis.util.justify import justify_row
from ocgis.test.base import TestBase


class Test(TestBase):

    def test_justify_row(self):
        row = "Food is any substance[1] consumed to provide nutritional support for the body. It is usually of plant or animal origin, and contains essential nutrients, such as carbohydrates, fats, proteins, vitamins, or minerals. The substance is ingested by an organism and assimilated by the organism's cells in an effort to produce energy, maintain life, or stimulate growth."
        eq = ['    Food is any substance[1] consumed to provide nutritional support for the',
              '    body. It is usually of plant or animal origin, and contains essential',
              '    nutrients, such as carbohydrates, fats, proteins, vitamins, or minerals.',
              "    The substance is ingested by an organism and assimilated by the organism's",
              '    cells in an effort to produce energy, maintain life, or stimulate growth.']
        justified = justify_row(row)
        self.assertEqual(justified,eq)
        
    def test_justify_row_without_words(self):
        row = 'a'*400
        ret = justify_row(row)
        self.assertEqual(ret,['    '+row])
        
        row = 'b'*5
        ret = justify_row(row)
        self.assertEqual(ret,['    '+row])
    
    def test_justify_long_word(self):
        aes = 'a'*400
        row = ' '.join(['short',aes,'end'])
        ret = justify_row(row)
        self.assertEqual(ret,['    short aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa end'])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()