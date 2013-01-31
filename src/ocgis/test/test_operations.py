import unittest
from ocgis.api.operations import OcgOperations
from datetime import datetime as dt
import numpy as np
from ocgis.exc import DefinitionValidationError, CannotEncodeUrl
from ocgis.api import definition
from urlparse import parse_qs
from ocgis.util.helpers import reduce_query, make_poly
from nose.plugins.skip import SkipTest
from ocgis.calc.library import SampleSize, Mean, StandardDeviation
from ocgis.util.shp_cabinet import ShpCabinet
import inspect
import traceback
from ocgis.api.definition import RequestDataset, RequestDatasetCollection

#from nose.plugins.skip import SkipTest
#raise SkipTest(__name__)

class Test(unittest.TestCase):
    uris = ['/tmp/foo1.nc','/tmp/foo2.nc','/tmp/foo3.nc']
    vars = ['tasmin','tasmax','tas']
    time_range = [dt(2000,1,1),dt(2000,12,31)]
    level_range = [2,2]
    datasets = [{'uri':uri,'variable':var,'time_range':time_range,'level_range':level_range} for uri,var in zip(uris,vars)]
    datasets_no_range = [{'uri':uri,'variable':var} for uri,var in zip(uris,vars)]

    def test_null_parms(self):
        ops = OcgOperations(dataset=self.datasets_no_range)
        self.assertNotEqual(ops.geom,None)
        self.assertEqual(len(ops.dataset),3)
        for ds in ops.dataset:
            self.assertEqual(ds.time_range,None)
            self.assertEqual(ds.level_range,None)
    
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
        self.assertEqual(ops.geom,[{'ugid': 1,'geom': None}])
        ops.geom = 'mi_watersheds'
        self.assertEqual(len(ops.geom),60)
        ops.geom = '-120|40|-110|50'
        self.assertEqual(ops.geom[0]['geom'].bounds,(-120.0,40.0,-110.0,50.0))
        ops.geom = [-120,40,-110,50]
        self.assertEqual(ops.geom[0]['geom'].bounds,(-120.0,40.0,-110.0,50.0))
        
    def test_calc(self):
        str_version = "[{'ref': <class 'ocgis.calc.library.Mean'>, 'name': 'mean', 'func': 'mean', 'kwds': {}}, {'ref': <class 'ocgis.calc.library.SampleSize'>, 'name': 'n', 'func': 'n', 'kwds': {}}]"
        _calc = [
                [{'func':'mean','name':'mean'},str_version],
                [[{'func':'mean','name':'mean'}],str_version],
                [None,'None'],
                [[{'func':'mean','name':'mean'},{'func':'std','name':'my_std'}],"[{'ref': <class 'ocgis.calc.library.Mean'>, 'name': 'mean', 'func': 'mean', 'kwds': {}}, {'ref': <class 'ocgis.calc.library.StandardDeviation'>, 'name': 'my_std', 'func': 'std', 'kwds': {}}, {'ref': <class 'ocgis.calc.library.SampleSize'>, 'name': 'n', 'func': 'n', 'kwds': {}}]"]
                ]
        
        for calc in _calc:
            cd = definition.Calc(calc[0])
            self.assertEqual(str(cd.value),calc[1])
            
        url_str = 'mean~my_mean'
        cd = definition.Calc(url_str)
        self.assertEqual(cd.value,[{'ref': Mean, 'name': 'my_mean', 'func': 'mean', 'kwds': {}}, {'ref': SampleSize, 'name': 'n', 'func': 'n', 'kwds': {}}])
        
        url_str = 'mean~my_mean|std~my_std'
        cd = definition.Calc(url_str)
        self.assertEqual(cd.value,[{'ref': Mean, 'name': 'my_mean', 'func': 'mean', 'kwds': {}}, {'ref': StandardDeviation, 'name': 'my_std', 'func': 'std', 'kwds': {}}, {'ref': SampleSize, 'name': 'n', 'func': 'n', 'kwds': {}}])
            
    def test_geom(self):
        geom = make_poly((37.762,38.222),(-102.281,-101.754))
        geom = [{'ugid':1,'geom':geom}]
        g = definition.Geom(geom)
        self.assertEqual(type(g.value),list)
        
        g = definition.Geom(None)
        self.assertNotEqual(g.value,None)
        self.assertEqual(str(g),'geom=none')
        
        g = definition.Geom('-120|40|-110|50')
        self.assertEqual(str(g),'geom=-120|40|-110|50')
        
        g = definition.Geom('mi_watersheds')
        self.assertEqual(str(g),'geom=mi_watersheds')
        
        geoms = ShpCabinet().get_geoms('mi_watersheds')
        g = definition.Geom(geoms)
        self.assertEqual(len(g.value),60)
        with self.assertRaises(CannotEncodeUrl):
            str(g)
        
    def test_calc_grouping(self):
        _cg = [
               None,
               ['day','month'],
               'day'
               ]
        
        for cg in _cg:
            obj = definition.CalcGrouping(cg)
            try:
                self.assertEqual(obj.value,cg)
            except AssertionError:
                self.assertEqual(obj.value,['day'])
                
    def test_dataset(self):
        rd = RequestDataset('/path/foo','hi')
        ds = definition.Dataset(rd)
        self.assertEqual(ds.value,RequestDatasetCollection([rd]))
        
        dsa = {'uri':'/path/foo','variable':'foo_variable'}
        ds = definition.Dataset(dsa)
        self.assertEqual(str(ds),'uri=/path/foo&variable=foo_variable&alias=foo_variable&t_units=none&t_calendar=none&s_proj=none')
        
        dsb = [dsa,{'uri':'/some/other/path','variable':'foo_variable','alias':'knight'}]
        ds = definition.Dataset(dsb)
        str_cmp = 'uri1=/path/foo&variable1=foo_variable&alias1=foo_variable&t_units1=none&t_calendar1=none&s_proj1=none&uri2=/some/other/path&variable2=foo_variable&alias2=knight&t_units2=none&t_calendar2=none&s_proj2=none'
        self.assertEqual(str(ds),str_cmp)
        
        query = parse_qs(str_cmp)
        query = reduce_query(query)
        ds = definition.Dataset.parse_query(query)
        self.assertEqual(str(ds),str_cmp)
        
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
            
        query = {'spatial_operation':['intersects']}
        obj = klass()
        obj.parse_query(query)

#    def test_time_range(self):
#        valid = [
##                 [dt(2000,1,1),dt(2000,12,31)],
#                 '2000-1-1|2000-12-31'
#                 ]
#        for v in valid:
#            tr = definition.TimeRange(v)
#            import ipdb;ipdb.set_trace()
        
class TestRequestDatasets(unittest.TestCase):
    uri = '/foo/path'
    variable = 'foo_you'
    
    def test_RequestDataset(self):
        uri = '/foo/path'
        variable = 'foo_you'
        
        rd = RequestDataset(uri,variable)
        self.assertEqual(rd['uri'],uri)
        self.assertEqual(rd['alias'],variable)
        
        rd = RequestDataset(uri,variable,alias='an_alias')
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
        uris = ['/path/uri1','/path/uri2']
        variables = ['foo1','foo2']
        rdc = RequestDatasetCollection()
        for uri,variable in zip(uris,variables):
            rd = RequestDataset(uri,variable)
            rdc.update(rd)
        
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
        
        
class TestUrl(unittest.TestCase):
    url_single = 'uri=http://www.dataset.com&variable=foo&spatial_operation=intersects'
    url_alias = url_single + '&alias=my_alias'
    url_multi = 'uri3=http://www.dataset.com&variable3=foo&uri5=http://www.dataset2.com&variable5=foo2&aggregate=true'
    url_bad = 'uri2=http://www.dataset.com&variable3=foo'
    url_long = 'uri1=hi&variable1=there&time_range1=2001-1-2|2001-1-5&uri2=hi2&variable2=there2&time_range2=2012-1-1|2012-12-31&level_range2=1|1'
    url_interface = url_long + '&t_calendar1=noleap'
    
    def get_reduced_query(self,attr):
        ref = getattr(self,attr)
        query = parse_qs(ref)
        query = reduce_query(query)
        return(query)
    
    def test_all_urls(self):
        members = inspect.getmembers(self)
        for member in members:
            ref = member[0]
            if ref.startswith('url_') and ref != 'url_bad':
                query = self.get_reduced_query(ref)
#                try:
                ops = OcgOperations.parse_query(query)
#                except Exception as e:
#                    print traceback.format_exc()
#                    import ipdb;ipdb.set_trace()
    
    def test_dataset_from_query(self):
        query = parse_qs(self.url_long)
        query = reduce_query(query)
        ds = definition.Dataset.parse_query(query)
        rds = [RequestDataset('hi','there',time_range='2001-1-2|2001-1-5'),
               RequestDataset('hi2','there2',time_range='2012-1-1|2012-12-31',level_range='1|1')]
        rdc = RequestDatasetCollection(rds)
        self.assertEqual(ds.value,rdc)
        
        query = self.get_reduced_query('url_alias')
        rdc = RequestDatasetCollection([RequestDataset('http://www.dataset.com','foo',alias='my_alias')])
        ds = definition.Dataset.parse_query(query)
        self.assertEqual(ds.value,rdc)
        
    def test_qs_generation(self):
        raise(SkipTest('pause in URL concerns at the moment...'))
        ds = {'uri':'/path/to/foo','variable':'tas'}
        ops = OcgOperations(ds)
        qs = ops.as_qs()
        self.assertEqual(qs,'/subset?snippet=false&abstraction=polygon&calc_raw=false&agg_selection=false&output_format=numpy&spatial_operation=intersects&uri=/path/to/foo&variable=tas&alias=tas&t_units=none&t_calendar=none&s_proj=none&calc_grouping=none&prefix=none&geom=none&allow_empty=false&vector_wrap=true&aggregate=false&calc=none&select_ugid=none&backend=ocg')
        
    def test_url_parsing(self):
        query = parse_qs(self.url_multi)
        reduced = reduce_query(query)
        self.assertEqual({'variable': [['foo', 'foo2']], 'aggregate': ['true'], 'uri': [['http://www.dataset.com', 'http://www.dataset2.com']]},reduced)

        query = parse_qs(self.url_single)
        reduced = reduce_query(query)
        self.assertEqual(reduced,{'variable': ['foo'], 'spatial_operation': ['intersects'], 'uri': ['http://www.dataset.com']})
        
        query = parse_qs(self.url_bad)
        with self.assertRaises(DefinitionValidationError):
            reduce_query(query)
            
        query = parse_qs(self.url_long)
        reduced = reduce_query(query)
        self.assertEqual(reduced,{'variable': [['there', 'there2']], 'level_range': [[None, '1|1']], 'uri': [['hi', 'hi2']], 'time_range': [['2001-1-2|2001-1-5', '2012-1-1|2012-12-31']]})
        
        query = self.get_reduced_query('url_interface')
        self.assertEqual(query,{'variable': [['there', 'there2']], 'level_range': [[None, '1|1']], 'time_range': [['2001-1-2|2001-1-5', '2012-1-1|2012-12-31']], 'uri': [['hi', 'hi2']], 't_calendar': [['noleap', None]]})


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()