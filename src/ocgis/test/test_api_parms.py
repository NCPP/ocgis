import unittest
from ocgis.api.parms.definition import *
from ocgis.util.helpers import make_poly
import pickle
import tempfile
import os
from ocgis.test.base import TestBase
from ocgis.util.shp_cabinet import ShpCabinet


class Test(TestBase):
    
    def test_dir_output(self):
        ## raise an exception if the directory does not exist
        do = '/does/not/exist'
        with self.assertRaises(DefinitionValidationError):
            DirOutput(do)
          
        ## make sure directory name does not change case
        do = 'Some'
        new_dir = os.path.join(tempfile.gettempdir(),do)
        os.mkdir(new_dir)
        try:
            dd = DirOutput(new_dir)
            self.assertEqual(new_dir,dd.value)
        finally:
            os.rmdir(new_dir)

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
            so.value = ('1|1|2')
        with self.assertRaises(DefinitionValidationError):
            so.value = '22.5'
        so = SelectUgid('22|23|24')
        self.assertEqual(so.value,(22,23,24))
        self.assertEqual(so.get_url_string(),'22|23|24')
        with self.assertRaises(DefinitionValidationError):
            so.value = '22|23.5|24'
            
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
        self.assertEqual(cg.get_url_string(),'day|month')
            
    def test_dataset(self):
        rd = self.test_data.get_rd('cancm4_tas')
        dd = Dataset(rd)
        us = dd.get_url_string()
        self.assertEqual(us,'uri=/usr/local/climate_data/CanCM4/tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc&variable=tas&alias=tas&t_units=none&t_calendar=none&s_proj=none')
        
        with open('/tmp/dd.pkl','w') as f:
#            import ipdb;ipdb.set_trace()
            pickle.dump(dd,f)
        
        uri = '/a/bad/path'
        with self.assertRaises(ValueError):
            rd = RequestDataset(uri,'foo')

    def test_geom(self):
        geom_base = make_poly((37.762,38.222),(-102.281,-101.754))
        geom = geom_base
        g = Geom(geom)
        self.assertEqual(type(g.value),GeometryDataset)
        g.value = None
        self.assertEqual(None,g.value)
        
        g = Geom(None)
        self.assertEqual(g.value,None)
        self.assertEqual(str(g),'geom=None')
        
        g = Geom('-120|40|-110|50')
        self.assertEqual(g.value.spatial.geom.bounds,(-120.0, 40.0, -110.0, 50.0))
        self.assertEqual(str(g),'geom=-120.0|40.0|-110.0|50.0')
        self.assertEqual(g.get_url_string(),'-120.0|40.0|-110.0|50.0')
        
        g = Geom('mi_watersheds')
        self.assertEqual(str(g),'geom=mi_watersheds')
        
        geoms = ShpCabinet().get_geoms('mi_watersheds')
        g = Geom('mi_watersheds')
        self.assertEqual(len(g.value),len(geoms))
        
        su = SelectUgid([1,2,3])
        g = Geom('mi_watersheds',select_ugid=su)
        self.assertEqual(len(g.value),3)
        
        geoms = GeometryDataset(uid=[1,2],geom=[geom,geom])
        g = Geom(geoms)
        with self.assertRaises(CannotEncodeUrl):
            g.get_url_string()
            
    def test_calc(self):
        calc = [{'func':'mean','name':'my_mean'}]
        cc = Calc(calc)
        eq = [{'ref':library.Mean,'name':'my_mean','func':'mean','kwds':{}}, 
              {'ref':library.SampleSize,'name':'n','func':'n','kwds':{}}]
        self.assertEqual(cc.value,eq)
        cc.value = 'mean~my_mean'
        self.assertEqual(cc.value,eq)
        cc.value = 'mean~my_mean|max~my_max|between~between5_10!lower~5!upper~10'
        self.assertEqual(cc.get_url_string(),'mean~my_mean|max~my_max|between~between5_10!lower~5.0!upper~10.0')
        
        ## test duplicate parameters
        calc = [{'func':'mean','name':'my_mean'},
                {'func':'mean','name':'my_mean'}]
        with self.assertRaises(DefinitionValidationError):
            Calc(calc)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()