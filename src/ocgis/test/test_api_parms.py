import unittest
from ocgis.api.parms.definition import *
from ocgis.exc import DefinitionValidationError, CannotEncodeUrl
from ocgis.util.helpers import make_poly
from ocgis.util.shp_cabinet import ShpCabinet


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
            
        s.get_meta()

    def test_spatial_operation(self):
        so = SpatialOperation()
        self.assertEqual(so.value,'intersects')
        with self.assertRaises(DefinitionValidationError):
            so.value = 'clips'
        so.value = 'clip'
        print so.get_meta()
            
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
            
    def test_prefix(self):
        pp = Prefix()
        self.assertEqual(pp.value,'ocgis_output')
        pp.value = ' Old__man '
        self.assertEqual(pp.value,'old__man')
        
    def test_calc_grouping(self):
        cg = CalcGrouping(['day','month'])
        self.assertEqual(cg.value,('day','month'))
        with self.assertRaises(DefinitionValidationError):
            cg.value = ['d','foo']
            
    def test_dataset(self):
        uri = '/usr/local/climate_data/CanCM4/tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc'
        variable = 'tas'
        rd = RequestDataset(uri,variable)
        dd = Dataset(rd)
        us = dd.get_url_string()
        self.assertEqual(us,'uri=/usr/local/climate_data/cancm4/tas_day_cancm4_decadal2000_r2i1p1_20010101-20101231.nc&variable=tas&alias=tas&t_units=none&t_calendar=none&s_proj=none')

    def test_geom(self):
        geom_base = make_poly((37.762,38.222),(-102.281,-101.754))
        geom = [{'ugid':1,'geom':geom_base}]
        g = Geom(geom)
        self.assertEqual(type(g.value),SelectionGeometry)
        
        g = Geom(None)
        self.assertNotEqual(g.value,None)
        self.assertEqual(str(g),'geom=none')
        
        g = Geom('-120|40|-110|50')
        self.assertEqual(g.value[0]['geom'].bounds,(-120.0, 40.0, -110.0, 50.0))
        self.assertEqual(str(g),'geom=-120.0|40.0|-110.0|50.0')
        self.assertEqual(g.get_url_string(),'-120.0|40.0|-110.0|50.0')
        
        g = Geom('mi_watersheds')
        self.assertEqual(str(g),'geom=mi_watersheds')
        
        geoms = ShpCabinet().get_geoms('mi_watersheds')
        g = Geom(geoms)
        self.assertEqual(len(g.value),len(geoms))
        
        su = SelectUgid([1,2,3])
        g = Geom('mi_watersheds',select_ugid=su)
        self.assertEqual(len(g.value),3)
        
        geoms = [geom[0],geom[0]]
        g = Geom(geoms)
        with self.assertRaises(CannotEncodeUrl):
            g.get_url_string()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()