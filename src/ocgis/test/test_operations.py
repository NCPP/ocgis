import unittest
from ocgis.api.operations import OcgOperations
from datetime import datetime as dt
import numpy as np
from ocgis.exc import DefinitionValidationError
from ocgis.api import definition
from urlparse import parse_qs
from ocgis.util.helpers import reduce_query

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
        self.assertEqual(ops.time_range,[None]*3)
        for row in ops:
            self.assertEqual(row['time_range'],None)
            self.assertEqual(row['level_range'],None)
            
        ops = OcgOperations(dataset=self.datasets[0])
        self.assertEqual(ops.time_range,[None])
        self.assertEqual(ops.level_range,[None])
        
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
        
        
class TestUrl(unittest.TestCase):
    url_single = 'uri=http://www.dataset.com&variable=foo&spatial_operation=intersects'
    url_multi = 'uri3=http://www.dataset.com&variable3=foo&uri5=http://www.dataset2.com&variable5=foo2&aggregate=true'
    url_bad = 'uri2=http://www.dataset.com&variable3=foo'
    url_long = 'uri1=hi&variable1=there&time_range1=2001-1-2|2001-1-5&uri2=hi2&variable2=there2&time_range2=2012-1-1|2012-12-31'
    
    def test_dataset_from_query(self):
        query = parse_qs(self.url_long)
        query = reduce_query(query)
        ds = definition.Dataset.parse_query(query)
        import ipdb;ipdb.set_trace()
        
    def test_url_generation(self):
        ds = {'uri':'/path/to/foo','variable':'tas'}
        ops = OcgOperations(ds)
        url = ops.url
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
        self.assertEqual(reduced,{'variable': [['there', 'there2']], 'uri': [['hi', 'hi2']], 'time_range': [['2001-1-2|2001-1-5', '2012-1-1|2012-12-31']]})


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()