import unittest
from ocgis.api.parms.definition import *
from ocgis.util.helpers import make_poly
import pickle
import tempfile
from ocgis.test.base import TestBase
from ocgis.calc.library.statistics import Mean
from ocgis.util.shp_cabinet import ShpCabinet
import numpy as np


class Test(TestBase):
    
    def test_callback(self):
        c = Callback()
        self.assertEqual(c.value,None)
        
        with self.assertRaises(DefinitionValidationError):
            Callback('foo')
            
        def callback(percent,message):
            pass
        
        c = Callback(callback)
        self.assertEqual(callback,c.value)
    
    def test_optimizations(self):
        o = Optimizations()
        self.assertEqual(o.value,None)
        with self.assertRaises(DefinitionValidationError):
            Optimizations({})
        with self.assertRaises(DefinitionValidationError):
            Optimizations({'foo':'foo'})
        o = Optimizations({'tgds':{'tas':'TemporalGroupDimension'}})
        self.assertEqual(o.value,{'tgds':{'tas':'TemporalGroupDimension'}})
        
    def test_optimizations_deepcopy(self):
        ## we should not deepcopy optimizations
        arr = np.array([1,2,3,4])
        value = {'tgds':{'tas':arr}}
        o = Optimizations(value)
        self.assertTrue(np.may_share_memory(o.value['tgds']['tas'],arr))
    
    def test_add_auxiliary_files(self):
        for val in [True,False]:
            p = AddAuxiliaryFiles(val)
            self.assertEqual(p.value,val)
        p = AddAuxiliaryFiles()
        self.assertEqual(p.value,True)
    
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
            
    def test_slice(self):
        slc = Slice(None)
        self.assertEqual(slc.value,None)
        
        slc = Slice([None,0,0,0,0])
        self.assertEqual(slc.value,(slice(None),slice(0,1),slice(0, 1),slice(0, 1),slice(0, 1)))
        
        slc = Slice([None,0,None,[0,1],[0,100]])
        self.assertEqual(slc.value,(slice(None),slice(0,1),slice(None),slice(0,1),slice(0,100)))
        
        with self.assertRaises(DefinitionValidationError):
            slc.value = 4
        with self.assertRaises(DefinitionValidationError):
            slc.value = [None,None]

    def test_snippet(self):
        self.assertFalse(Snippet().value)
        for ii in ['t','TRUE','tRue',1,'1',' 1 ']:
            self.assertTrue(Snippet(ii).value)
        s = Snippet()
        s.value = False
        self.assertFalse(s.value)
        s.value = '0'
        self.assertFalse(s.value)
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
        with self.assertRaises(DefinitionValidationError):
            so.value = '22|23.5|24'
            
    def test_prefix(self):
        pp = Prefix()
        self.assertEqual(pp.value,'ocgis_output')
        pp.value = ' Old__man '
        self.assertEqual(pp.value,'Old__man')
        
    def test_calc_grouping(self):
        cg = CalcGrouping(['day','month'])
        self.assertEqual(cg.value,('day','month'))
        with self.assertRaises(DefinitionValidationError):
            cg.value = ['d','foo']
            
    def test_calc_grouping_all(self):
        cg = CalcGrouping('all')
        self.assertEqual(cg.value,'all')
    
    def test_calc_grouping_seasonal_aggregation(self):
        cg = CalcGrouping([[1,2,3],[4,5,6]])
        self.assertEqual(cg.value,([1,2,3],[4,5,6]))
        
        ## element groups must be composed of unique elements
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1,2,3],[4,4,6]])
        
        ## element groups must have an empty intersection
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1,2,3],[1,4,6]])
        
        ## months must be between 1 and 12
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1,2,3],[4,5,66]])
    
    def test_calc_grouping_seasonal_aggregation_with_year(self):
        cg = CalcGrouping([[1,2,3],[4,5,6],'year'])
        self.assertEqual(cg.value,([1,2,3],[4,5,6],'year'))
        
    def test_calc_grouping_seasonal_aggregation_with_unique(self):
        cg = CalcGrouping([[1,2,3],[4,5,6],'unique'])
        self.assertEqual(cg.value,([1,2,3],[4,5,6],'unique'))
        
    def test_calc_grouping_seasonal_aggregation_with_bad_flag(self):
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1,2,3],[4,5,6],'foo'])
        with self.assertRaises(DefinitionValidationError):
            CalcGrouping([[1,2,3],[4,5,6],'fod'])
            
    def test_dataset(self):
        rd = self.test_data.get_rd('cancm4_tas')
        dd = Dataset(rd)
        
        with open('/tmp/dd.pkl','w') as f:
            pickle.dump(dd,f)
        
        uri = '/a/bad/path'
        with self.assertRaises(ValueError):
            rd = RequestDataset(uri,'foo')

    def test_geom(self):
        geom = make_poly((37.762,38.222),(-102.281,-101.754))

        g = Geom(geom)
        self.assertEqual(type(g.value),list)
        g.value = None
        self.assertEqual(None,g.value)
        
        g = Geom(None)
        self.assertEqual(g.value,None)
        self.assertEqual(str(g),'geom=None')
        
        g = Geom('-120|40|-110|50')
        self.assertEqual(g.value[0]['geom'].bounds,(-120.0, 40.0, -110.0, 50.0))
        self.assertEqual(str(g),'geom=-120.0|40.0|-110.0|50.0')
        
        g = Geom('mi_watersheds')
        self.assertEqual(str(g),'geom=mi_watersheds')
        
        geoms = list(ShpCabinetIterator('mi_watersheds'))
        g = Geom('mi_watersheds')
        self.assertEqual(len(list(g.value)),len(geoms))
        
        su = SelectUgid([1,2,3])
        g = Geom('mi_watersheds',select_ugid=su)
        self.assertEqual(len(list(g.value)),3)
        
        geoms = [{'geom':geom,'properties':{'UGID':1}},{'geom':geom,'properties':{'UGID':2}}]
        g = Geom(geoms)
        
        bbox = [-120,40,-110,50]
        g = Geom(bbox)
        self.assertEqual(g.value[0]['geom'].bounds,tuple(map(float,bbox)))
        
    def test_geom_using_shp_path(self):
        ## pass a path to a shapefile as opposed to a key
        path = ShpCabinet().get_shp_path('state_boundaries')
        ocgis.env.DIR_SHPCABINET = None
        ## make sure there is path associated with the ShpCabinet
        with self.assertRaises(ValueError):
            ShpCabinet().keys()
        g = Geom(path)
        self.assertEqual(g._shp_key,path)
        self.assertEqual(len(list(g.value)),51)
        
    def test_geom_with_changing_select_ugid(self):
        select_ugid = [16,17]
        g = Geom('state_boundaries',select_ugid=select_ugid)
        self.assertEqual(len(list(g.value)),2)
        select_ugid.append(22)
        self.assertEqual(len(list(g.value)),3)
        
        g = Geom('state_boundaries')
        self.assertEqual(len(list(g.value)),51)
        g.select_ugid = [16,17]
        self.assertEqual(len(list(g.value)),2)
            
    def test_calc(self):
        calc = [{'func':'mean','name':'my_mean'}]
        cc = Calc(calc)
        eq = [{'ref':Mean,'name':'my_mean','func':'mean','kwds':{}}]

        self.assertEqual(cc.value,eq)
        cc.value = 'mean~my_mean'
        self.assertEqual(cc.value,eq)
        cc.value = 'mean~my_mean|max~my_max|between~between5_10!lower~5!upper~10'
        with self.assertRaises(NotImplementedError):
            self.assertEqual(cc.get_url_string(),'mean~my_mean|max~my_max|between~between5_10!lower~5.0!upper~10.0')

    def test_calc_bad_key(self):
        calc = [{'func':'bad_mean','name':'my_mean'}]
        with self.assertRaises(DefinitionValidationError):
            Calc(calc)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()