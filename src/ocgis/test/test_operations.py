from ocgis.api.operations import OcgOperations
from datetime import datetime as dt
from ocgis.exc import DefinitionValidationError
from ocgis.util.helpers import make_poly
from ocgis import env
import os.path
from ocgis.api.parms import definition
import pickle
import ocgis
from ocgis.test.base import TestBase
from nose.plugins.skip import SkipTest
from ocgis.api.request import RequestDataset, RequestDatasetCollection
from ocgis.interface.geometry import GeometryDataset
from ocgis.interface.shp import ShpDataset
import itertools


class Test(TestBase):
    
    def setUp(self):
        TestBase.setUp(self)
        env.DIR_DATA = os.path.join(env.DIR_TEST_DATA,'CanCM4')
        
        ## data may need to be pulled from remote repository
        self.test_data.get_rd('cancm4_tasmin_2001')
        self.test_data.get_rd('cancm4_tasmax_2011')
        self.test_data.get_rd('cancm4_tas')
        
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

    def test_repr(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)
        ret = str(ops)

    def test_get_meta(self):
        ops = OcgOperations(dataset=self.datasets)
        meta = ops.get_meta()
        self.assertTrue(len(meta) > 100)
        self.assertTrue('\n' in meta)
        
        ops = OcgOperations(dataset=self.datasets,calc=[{'func':'mean','name':'my_mean'}])
        meta = ops.get_meta()
        self.assertTrue(len(meta) > 100)
        self.assertTrue('\n' in meta)

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
        
    def test_headers(self):
        headers = ['did','value']
        for htype in [list,tuple]:
            hvalue = htype(headers)
            hh = definition.Headers(hvalue)
            self.assertEqual(tuple(hvalue),hh.value)
            
        headers = ['foo']
        with self.assertRaises(DefinitionValidationError):
            hh = definition.Headers(headers)
            
        headers = []
        with self.assertRaises(DefinitionValidationError):
            hh = definition.Headers(headers)
                
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
        
        ## only month, year, and day combinations are currently supported
        rd = self.test_data.get_rd('cancm4_tas')
        calcs = [None,[{'func':'mean','name':'mean'}]]
        acceptable = ['day','month','year']
        for calc in calcs:
            for length in [1,2,3,4,5]:
                for combo in itertools.combinations(['day','month','year','hour','minute'],length):
                    try:
                        ops = OcgOperations(dataset=rd,calc=calc,calc_grouping=combo)
                    except DefinitionValidationError:
                        reraise = True
                        for c in combo:
                            if c not in acceptable:
                                reraise = False
                        if reraise:
                            raise
                
    def test_dataset(self):
        env.DIR_DATA = ocgis.env.DIR_TEST_DATA
        reference_rd = self.test_data.get_rd('cancm4_tas')
        rd = RequestDataset(reference_rd.uri,reference_rd.variable)
        ds = definition.Dataset(rd)
        self.assertEqual(ds.value,RequestDatasetCollection([rd]))
        
        dsa = {'uri':reference_rd.uri,'variable':reference_rd.variable}
        ds = definition.Dataset(dsa)
        
        reference_rd2 = self.test_data.get_rd('narccap_crcm')
        dsb = [dsa,{'uri':reference_rd2.uri,'variable':reference_rd2.variable,'alias':'knight'}]
        ds = definition.Dataset(dsb)
        
    def test_abstraction(self):
        K = definition.Abstraction
        
        k = K()
        self.assertEqual(k.value,'polygon')
        self.assertEqual(str(k),'abstraction="polygon"')
        
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
            
        
class TestRequestDataset(TestBase):
    
    def setUp(self):
        TestBase.setUp(self)
        ## download test data
        self.test_data.get_rd('cancm4_rhs')
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
        
    def test_inspect_as_dct(self):
        variables = [self.variable,None,'foo','time']
        
        for variable in variables:
            rd = RequestDataset(self.uri,variable)
            try:
                ret = rd.inspect_as_dct()
            except KeyError:
                if variable == 'foo':
                    continue
                else:
                    raise
            except ValueError:
                if variable == 'time':
                    continue
                else:
                    raise
            ref = ret['derived']
            
            if variable is None:
                self.assertEqual(ref,{'End Date': '2020-12-31 12:00:00', 'Start Date': '2011-01-01 12:00:00'})
            else:
                self.assertEqual(ref['End Date'],'2020-12-31 12:00:00')
    
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
        rd = RequestDataset(self.uri,self.variable,alias='an_alias')
        self.assertEqual(rd.alias,'an_alias')
        rd = RequestDataset(self.uri,self.variable,alias=None)
        self.assertEqual(rd.alias,self.variable)
        
    def test_RequestDataset_time_range(self):        
        tr = [dt(2000,1,1),dt(2000,12,31)]
        rd = RequestDataset(self.uri,self.variable,time_range=tr)
        self.assertEqual(rd.time_range,tr)
        
        out = [dt(2000, 1, 1, 0, 0),dt(2000, 12, 31,)]
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
        
        daymet = self.test_data.get_rd('daymet_tmax')
        tas = self.test_data.get_rd('cancm4_tas')
        
        uris = [daymet.uri,
                tas.uri]
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
        
    def test_multiple_uris(self):
        rd = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        self.assertEqual(len(rd.uri),2)
        rd.inspect()
        
    def test_time_region(self):
        tr1 = {'month':[6],'year':[2001]}
        rd = RequestDataset(uri=self.uri,variable=self.variable,time_region=tr1)
        self.assertEqual(rd.time_region,tr1)
        
        tr2 = {'bad':15}
        with self.assertRaises(DefinitionValidationError):
            RequestDataset(uri=self.uri,variable=self.variable,time_region=tr2)
            
        tr_str = 'month~6|year~2001'
        rd = RequestDataset(uri=self.uri,variable=self.variable,time_region=tr_str)
        self.assertEqual(rd.time_region,tr1)
        
        tr_str = 'month~6-8|year~2001-2003'
        rd = RequestDataset(uri=self.uri,variable=self.variable,time_region=tr_str)
        self.assertEqual(rd.time_region,{'month':[6,7,8],'year':[2001,2002,2003]})
        
        tr_str = 'month~6-8'
        rd = RequestDataset(uri=self.uri,variable=self.variable,time_region=tr_str)
        self.assertEqual(rd.time_region,{'month':[6,7,8],'year':None})
        
        tr_str = 'month~6-8|year~none'
        rd = RequestDataset(uri=self.uri,variable=self.variable,time_region=tr_str)
        self.assertEqual(rd.time_region,{'month':[6,7,8],'year':None})
