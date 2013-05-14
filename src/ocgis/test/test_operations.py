import unittest
from ocgis.api.operations import OcgOperations
from datetime import datetime as dt
from ocgis.exc import DefinitionValidationError, CannotEncodeUrl
from ocgis.util.helpers import make_poly
from ocgis.util.shp_cabinet import ShpCabinet
from ocgis import env
import os.path
from ocgis.api.parms import definition
from ocgis.api.geometry import SelectionGeometry
import pickle
import ocgis
from ocgis.test.base import TestBase
from nose.plugins.skip import SkipTest
from ocgis.api.request import RequestDataset, RequestDatasetCollection
from ocgis.interface.geometry import GeometryDataset
from ocgis.interface.shp import ShpDataset


class Test(TestBase):
    
    def setUp(self):
        TestBase.setUp(self)
        env.DIR_DATA = os.path.join(env.DIR_TEST_DATA,'CanCM4')
        uris = [
                'tasmin_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
                'tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                'tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc'
                ]
        vars = ['tasmin','tasmax','tas']
        time_range = [dt(2000,1,1),dt(2000,12,31)]
        level_range = [2,2]
        self.datasets = [{'uri':uri,'variable':var,'time_range':time_range,'level_range':level_range} for uri,var in zip(uris,vars)]
        self.datasets_no_range = [{'uri':uri,'variable':var} for uri,var in zip(uris,vars)]

    def test_get_meta(self):
        ops = OcgOperations(dataset=self.datasets)
        meta = ops.get_meta()
        self.assertTrue(len(meta) > 100)
        self.assertTrue('\n' in meta)
        
        ops = OcgOperations(dataset=self.datasets,calc=[{'func':'mean','name':'my_mean'}])
        meta = ops.get_meta()
        self.assertTrue(len(meta) > 100)
        self.assertTrue('\n' in meta)
        self.assertTrue('/subset?' in meta)

    def test_null_parms(self):
        ops = OcgOperations(dataset=self.datasets_no_range)
        self.assertEqual(ops.geom,None)
        self.assertEqual(len(ops.dataset),3)
        for ds in ops.dataset:
            self.assertEqual(ds.time_range,None)
            self.assertEqual(ds.level_range,None)
        ops.__repr__()
    
    def test_aggregate(self):
        A = definition.Aggregate
        
        a = A(True)
        self.assertEqual(a.value,True)
        
        a = A(False)
        self.assertEqual(a.value,False)
        
        a = A('True')
        self.assertEqual(a.value,True)
    
    def test_geom_string(self):
        ops = OcgOperations(dataset=self.datasets,geom='state_boundaries')
        self.assertEqual(len(ops.geom),51)
        ops.geom = None
        self.assertEqual(ops.geom,None)
        ops.geom = 'mi_watersheds'
        self.assertEqual(len(ops.geom),60)
        ops.geom = '-120|40|-110|50'
        self.assertEqual(ops.geom.spatial.geom[0].bounds,(-120.0,40.0,-110.0,50.0))
        ops.geom = [-120,40,-110,50]
        self.assertEqual(ops.geom.spatial.geom[0].bounds,(-120.0,40.0,-110.0,50.0))
        
    def test_geom(self):
        geom = make_poly((37.762,38.222),(-102.281,-101.754))
        g = definition.Geom(geom)
        self.assertEqual(type(g.value),GeometryDataset)
        
        g = definition.Geom(None)
        self.assertEqual(g.value,None)
        self.assertEqual(str(g),'geom=None')
        
        g = definition.Geom('-120|40|-110|50')
        self.assertEqual(str(g),'geom=-120.0|40.0|-110.0|50.0')
        
        g = definition.Geom('mi_watersheds')
        self.assertEqual(str(g),'geom=mi_watersheds')
        
        geoms = ShpDataset('mi_watersheds')
        g = definition.Geom(geoms)
        self.assertEqual(len(g.value),60)
        with self.assertRaises(CannotEncodeUrl):
            g.get_url_string()
        
    def test_calc_grouping(self):
        _cg = [
               None,
               ['day','month'],
               'day'
               ]
        
        for cg in _cg:
            if cg is not None:
                eq = tuple(cg)
            else:
                eq = cg
            obj = definition.CalcGrouping(cg)
            try:
                self.assertEqual(obj.value,eq)
            except AssertionError:
                self.assertEqual(obj.value,('day',))
                
    def test_dataset(self):
        env.DIR_DATA = ocgis.env.DIR_TEST_DATA
        rd = RequestDataset('tas_day_CanCM4_decadal2011_r2i1p1_20120101-20211231.nc','tas')
        ds = definition.Dataset(rd)
        self.assertEqual(ds.value,RequestDatasetCollection([rd]))
        
        dsa = {'uri':'tas_day_CanCM4_decadal2011_r2i1p1_20120101-20211231.nc','variable':'tas'}
        ds = definition.Dataset(dsa)
        self.assertEqual(ds.get_url_string(),'uri=/usr/local/climate_data/CanCM4/tas_day_CanCM4_decadal2011_r2i1p1_20120101-20211231.nc&variable=tas&alias=tas&t_units=none&t_calendar=none&s_proj=none')
        
        dsb = [dsa,{'uri':'albisccp_cfDay_CCSM4_1pctCO2_r2i1p1_00200101-00391231.nc','variable':'albisccp','alias':'knight'}]
        ds = definition.Dataset(dsb)
        str_cmp = 'uri1=/usr/local/climate_data/CanCM4/tas_day_CanCM4_decadal2011_r2i1p1_20120101-20211231.nc&variable1=tas&alias1=tas&t_units1=none&t_calendar1=none&s_proj1=none&uri2=/usr/local/climate_data/CCSM4/albisccp_cfDay_CCSM4_1pctCO2_r2i1p1_00200101-00391231.nc&variable2=albisccp&alias2=knight&t_units2=none&t_calendar2=none&s_proj2=none'
        self.assertEqual(ds.get_url_string(),str_cmp)
        
    def test_abstraction(self):
        K = definition.Abstraction
        
        k = K()
        self.assertEqual(k.value,'polygon')
        self.assertEqual(str(k),'abstraction=polygon')
        
        k = K('point')
        self.assertEqual(k.value,'point')
        
        with self.assertRaises(DefinitionValidationError):
            K('pt')
            
    def test_spatial_operation(self):
        values = (None,'clip','intersects')
        ast = ('intersects','clip','intersects')
        
        klass = definition.SpatialOperation
        for v,a in zip(values,ast):
            obj = klass(v)
            self.assertEqual(obj.value,a)
            
    def test_as_url(self):
        raise(SkipTest)
        ocgis.env.DIR_DATA = os.path.join(ocgis.env.DIR_TEST_DATA,'CanCM4')
        
        ## build request datasets
        filenames = [
    #                 'rhsmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
                     'tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc'
                     ]
        variables = [
    #                 'rhsmax',
                     'tasmax'
                     ]
        rds = [ocgis.RequestDataset(fn,var) for fn,var in zip(filenames,variables)]
        
        ## build calculations
        funcs = ['mean','std']
    #    funcs = ['mean','std','min','max','median']
        calc = [{'func':func,'name':func} for func in funcs]
        
        ## operations
        select_ugid = None
        calc_grouping = ['month']
        snippet = False
        geom = 'climate_divisions'
        output_format = 'csv'
        ops = ocgis.OcgOperations(dataset=rds,select_ugid=select_ugid,snippet=snippet,
         output_format=output_format,geom=geom,calc=calc,calc_grouping=calc_grouping,
         spatial_operation='clip',aggregate=True)
        url = ops.as_url()
        self.assertEqual(url,'/subset?snippet=0&abstraction=polygon&calc_raw=0&agg_selection=0&output_format=csv&spatial_operation=clip&uri=/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc&variable=tasmax&alias=tasmax&t_units=none&t_calendar=none&s_proj=none&calc_grouping=month&prefix=ocgis_output&geom=climate_divisions&allow_empty=0&vector_wrap=1&aggregate=1&select_ugid=none&calc=mean~mean|std~std&backend=ocg')
            
        
class TestRequestDatasets(TestBase):
    
    def setUp(self):
        TestBase.setUp(self)
        self.uri = os.path.join(ocgis.env.DIR_TEST_DATA,'CanCM4','rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc')
        self.variable = 'rhs'
    
    def test_pickle(self):
        rd = RequestDataset(uri=self.uri,variable=self.variable)
        rd_path = os.path.join(ocgis.env.DIR_OUTPUT,'rd.pkl')
        with open(rd_path,'w') as f:
            pickle.dump(rd,f)
        with open(rd_path,'r') as f:
            rd2 = pickle.load(f)
        self.assertTrue(rd == rd2)
    
    def test_inspect_method(self):
        rd = RequestDataset(self.uri,self.variable)
        rd.inspect()
    
    def test_env_dir_data(self):
        ## test setting the var to a single directory
        env.DIR_DATA = ocgis.env.DIR_TEST_DATA
        rd = self.test_data.get_rd('cancm4_rhs')
        target = os.path.join(env.DIR_DATA,'CanCM4','rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc')
        try:
            self.assertEqual(rd.uri,target)
        ## attempt to normalize the path
        except AssertionError:
            self.assertEqual(rd.uid,os.path.normpath(target))
        
        ## test none and not finding the data
        env.DIR_DATA = None
        with self.assertRaises(ValueError):
            RequestDataset('does_not_exists.nc',variable='foo')
            
        ## set data directory and not find it.
        env.DIR_DATA = os.path.join(ocgis.env.DIR_TEST_DATA,'CCSM4')
        with self.assertRaises(ValueError):
            RequestDataset('rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',variable='rhs')
    
    def test_RequestDataset(self):
        rd = RequestDataset(self.uri,self.variable)
        self.assertEqual(rd['uri'],self.uri)
        self.assertEqual(rd['alias'],self.variable)
        
        rd = RequestDataset(self.uri,self.variable,alias='an_alias')
        self.assertEqual(rd.alias,'an_alias')
        
    def test_RequestDataset_time_range(self):        
        tr = [dt(2000,1,1),dt(2000,12,31)]
        rd = RequestDataset(self.uri,self.variable,time_range=tr)
        self.assertEqual(rd.time_range,tr)
        
        out = [dt(2000, 1, 1, 0, 0),dt(2000, 12, 31, 23, 59, 59)]
        tr = '2000-1-1|2000-12-31'
        rd = RequestDataset(self.uri,self.variable,time_range=tr)
        self.assertEqual(rd.time_range,out)
        
        tr = '2000-12-31|2000-1-1'
        with self.assertRaises(DefinitionValidationError):
            rd = RequestDataset(self.uri,self.variable,time_range=tr)
            
    def test_RequestDataset_level_range(self):
        lr = '1|1'
        rd = RequestDataset(self.uri,self.variable,level_range=lr)
        self.assertEqual(rd.level_range,[1,1])
        
        with self.assertRaises(DefinitionValidationError):
            rd = RequestDataset(self.uri,self.variable,level_range=[2,1])
    
    def test_RequestDatasetCollection(self):
        env.DIR_DATA = ocgis.env.DIR_TEST_DATA
        uris = ['tas_day_CanCM4_decadal2011_r2i1p1_20120101-20211231.nc',
                'albisccp_cfDay_CCSM4_1pctCO2_r2i1p1_00200101-00391231.nc']
        variables = ['foo1','foo2']
        rdc = RequestDatasetCollection()
        for uri,variable in zip(uris,variables):
            rd = RequestDataset(uri,variable)
            rdc.update(rd)
        self.assertEqual([1,2],[rd.did for rd in rdc])
            
        variables = ['foo1','foo1']
        rdc = RequestDatasetCollection()
        for ii,(uri,variable) in enumerate(zip(uris,variables)):
            rd = RequestDataset(uri,variable)
            if ii == 1:
                with self.assertRaises(KeyError):
                    rdc.update(rd)
            else:
                rdc.update(rd)
                
        aliases = ['a1','a2']
        for uri,variable,alias in zip(uris,variables,aliases):
            rd = RequestDataset(uri,variable,alias=alias)
            rdc.update(rd)
        for row in rdc:
            self.assertIsInstance(row,RequestDataset)
        self.assertIsInstance(rdc[0],RequestDataset)
        self.assertIsInstance(rdc['a2'],RequestDataset)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()