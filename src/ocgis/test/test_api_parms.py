import unittest
from ocgis.api.parms.definition import Snippet, SpatialOperation, OutputFormat,\
    SelectUgid
from ocgis.exc import DefinitionValidationError


class Test(unittest.TestCase):

    def test_snippet(self):
        self.assertFalse(Snippet().value)
        for ii in ['t','TRUE','tRue',1,'1',' 1 ']:
            self.assertTrue(Snippet(ii).value)
        s = Snippet()
        s.value = False
        self.assertFalse(s.value)
        s.value = '0'
        self.assertFalse(s.value)
        self.assertEqual(s.get_url_string(),'0')
        s.value = 1
        self.assertEqual(s.get_url_string(),'1')
        with self.assertRaises(DefinitionValidationError):
            s.value = 'none'

    def test_spatial_operation(self):
        so = SpatialOperation()
        self.assertEqual(so.value,'intersects')
        with self.assertRaises(DefinitionValidationError):
            so.value = 'clips'
            
    def test_output_format(self):
        so = OutputFormat('csv')
        self.assertEqual(so.value,'csv')
        so.value = 'NUMPY'
        self.assertEqual(so.value,'numpy')
        
    def test_select_ugid(self):
        so = SelectUgid()
        self.assertEqual(so.value,None)
        with self.assertRaises(DefinitionValidationError):
            so.value = 98.5
        so.value = 'none'
        self.assertEqual(so.value,None)
        with self.assertRaises(DefinitionValidationError):
            so.value = 1
        so = SelectUgid('10')
        self.assertEqual(so.value,(10,))
        with self.assertRaises(DefinitionValidationError):
            so.value = ('1,1,2')
        with self.assertRaises(DefinitionValidationError):
            so.value = '22.5'
        so = SelectUgid('22,23,24')
        self.assertEqual(so.value,(22,23,24))
        self.assertEqual(so.get_url_string(),'22,23,24')
        with self.assertRaises(DefinitionValidationError):
            so.value = '22,23.5,24'


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()