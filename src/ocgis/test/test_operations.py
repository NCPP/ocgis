import unittest
from ocgis.api.operations import OcgOperations
from datetime import datetime as dt
import numpy as np
from ocgis.exc import DefinitionValidationError, CannotEncodeUrl
from ocgis.api import definition
from urlparse import parse_qs
from ocgis.util.helpers import reduce_query
from nose.plugins.skip import SkipTest
from ocgis.calc.library import SampleSize
from ocgis.util.shp_cabinet import ShpCabinet
import inspect
import traceback

#from nose.plugins.skip import SkipTest
#raise SkipTest(__name__)

class Test(unittest.TestCase):
    uris = ['/tmp/foo1.nc','/tmp/foo2.nc','/tmp/foo3.nc']
    vars = ['tasmin','tasmax','tas']
    time_range = [dt(2000,1,1),dt(2000,12,31)]
    level_range = 2
    datasets = [{'uri':uri,'variable':var} for uri,var in zip(uris,vars)]

    def test_equal_length_level_range(self):
        
        ## check that parms come out equal lengths and level range is correctly
        ## set.
        ops = OcgOperations(dataset=self.datasets,
                            time_range=self.time_range,
                            level_range=self.level_range)
        self.assertEqual(len(ops.dataset),len(ops.time_range))
        self.assertEqual(len(ops.dataset),len(ops.level_range))
        self.assertEqual(np.array(ops.level_range,dtype=int).max(),self.level_range)
        
        ## assert error is raised with arguments of differing lengths
        level_range = [[2,2],[3,3]]
        with self.assertRaises(DefinitionValidationError):
            ops = OcgOperations(dataset=self.datasets,
                                time_range=self.time_range,
                                level_range=level_range)

    def test_iter(self):
        ops = OcgOperations(dataset=self.datasets,
                            time_range=self.time_range,
                            level_range=self.level_range)
        for row in ops:
            self.assertEqual(row['time_range'],self.time_range)
            self.assertEqual(row['level_range'],[self.level_range,self.level_range])

    def test_null_parms(self):
        ops = OcgOperations(dataset=self.datasets)
        self.assertNotEqual(ops.geom,None)
        self.assertEqual(ops.time_range,[None]*3)
        for row in ops:
            self.assertEqual(row['time_range'],None)
            self.assertEqual(row['level_range'],None)
            
        ops = OcgOperations(dataset=self.datasets[0])
        self.assertEqual(ops.time_range,None)
        self.assertEqual(ops.level_range,None)
        
        for row in ops:
            for attr in ['level_range','time_range']:
                self.assertEqual(row[attr],None)
            self.assertEqual(set(['uri','alias','s_proj','t_units','variable','t_calendar']),set(row['dataset'].keys()))
        
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
            
    def test_geom(self):
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
        
#        k = K()
#        self.assertEqual(k.value,'polygon')
#        self.assertEqual(str(k),'abstraction=polygon')
#        
#        k = K('point')
#        self.assertEqual(k.value,'point')
        
        k = K('pt')
        import ipdb;ipdb.set_trace()
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
        
        
class TestUrl(unittest.TestCase):
    url_single = 'uri=http://www.dataset.com&variable=foo&spatial_operation=intersects'
    url_alias = url_single + '&alias=my_alias'
    url_multi = 'uri3=http://www.dataset.com&variable3=foo&uri5=http://www.dataset2.com&variable5=foo2&aggregate=true'
    url_bad = 'uri2=http://www.dataset.com&variable3=foo'
    url_long = 'uri1=hi&variable1=there&time_range1=2001-1-2|2001-1-5&uri2=hi2&variable2=there2&time_range2=2012-1-1|2012-12-31&level_range2=1'
    url_interface = url_long + '&t_units1=noleap'
    
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
                try:
                    ops = OcgOperations.parse_query(query)
                except Exception as e:
                    print traceback.format_exc()
                    import ipdb;ipdb.set_trace()
    
    def test_time_range_parsing(self):
        query = self.get_reduced_query('url_long')
        tr = definition.TimeRange()
        tr.parse_query(query)
        self.assertEqual(tr.value,[[dt(2001,1,2,0,0),dt(2001,1,5,23,59,59)],[dt(2012,1,1,0,0),dt(2012,12,31,23,59,59)]])
        
    def test_level_range_parsing(self):
        query = self.get_reduced_query('url_long')
        lr = definition.LevelRange()
        lr.parse_query(query)
        self.assertEqual(lr.value,[None,[1, 1]])
    
    def test_dataset_from_query(self):
        query = parse_qs(self.url_long)
        query = reduce_query(query)
        ds = definition.Dataset.parse_query(query)
        self.assertEqual(ds.value,[{'uri': 'hi', 'alias': 'there', 's_proj': None, 't_units': None, 'variable': 'there', 't_calendar': None}, {'uri': 'hi2', 'alias': 'there2', 's_proj': None, 't_units': None, 'variable': 'there2', 't_calendar': None}])
        
        query = self.get_reduced_query('url_alias')
        ds = definition.Dataset.parse_query(query)
        self.assertEqual(ds.value,[{'uri': 'http://www.dataset.com', 'alias': 'my_alias', 's_proj': None, 't_units': None, 'variable': 'foo', 't_calendar': None}])
        
    def test_url_generation(self):
        raise(SkipTest('url not implemented'))
        ds = {'uri':'/path/to/foo','variable':'tas'}
        ops = OcgOperations(ds)
        url = ops.as_url()
        import ipdb;ipdb.set_trace()
        
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
        self.assertEqual(reduced,{'variable': [['there', 'there2']], 'level_range': [[None, '1']], 'uri': [['hi', 'hi2']], 'time_range': [['2001-1-2|2001-1-5', '2012-1-1|2012-12-31']]})
        
        query = self.get_reduced_query('url_interface')
        self.assertEqual(query,{'variable': [['there', 'there2']], 'level_range': [[None, '1']], 'time_range': [['2001-1-2|2001-1-5', '2012-1-1|2012-12-31']], 'uri': [['hi', 'hi2']], 't_units': [['noleap', None]]})
        ds = definition.Dataset.parse_query(query)
        self.assertEqual(ds.value,[{'uri': 'hi', 'alias': 'there', 's_proj': None, 't_units': 'noleap', 'variable': 'there', 't_calendar': None}, {'uri': 'hi2', 'alias': 'there2', 's_proj': None, 't_units': None, 'variable': 'there2', 't_calendar': None}])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()