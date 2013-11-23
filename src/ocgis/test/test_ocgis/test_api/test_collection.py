import unittest
from ocgis.api.collection import SpatialCollection
from ocgis.util.shp_cabinet import ShpCabinet
from shapely.geometry.multipolygon import MultiPolygon
import datetime
from ocgis import constants
from ocgis.calc.library.statistics import Mean
from ocgis.interface.base.variable import Variable
from ocgis.interface.base.field import DerivedField, DerivedMultivariateField
from copy import copy
from ocgis.calc.library.math import Divide
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
from ocgis.calc.library.thresholds import Threshold


class TestSpatialCollection(AbstractTestField):

    def test_constructor(self):
        field = self.get_field(with_value=True)
        sc = ShpCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta,key='state_boundaries')
        for row in sc.iter_geoms('state_boundaries'):
            sp.add_field(row['properties']['UGID'],row['geom'],field.variables.keys()[0],
                         field,properties=row['properties'])
        self.assertEqual(len(sp),51)
        self.assertIsInstance(sp.geoms[25],MultiPolygon)
        self.assertIsInstance(sp.properties[25],dict)
        self.assertEqual(sp[25]['tmax'].variables['tmax'].value.shape,(2, 31, 2, 3, 4))
    
    def test_iteration(self):
        field = self.get_field(with_value=True)
        
        field.temporal.name_uid = 'tid'
        field.level.name_uid = 'lid'
        field.spatial.geom.name_uid = 'gid'
        field.spatial.name_uid = 'gid'
        
        sc = ShpCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta,key='state_boundaries')
        for row in sc.iter_geoms('state_boundaries'):
            sp.add_field(row['properties']['UGID'],row['geom'],field.variables.keys()[0],
                         field,properties=row['properties'])
        for ii,row in enumerate(sp.get_iter_dict()):
            if ii == 1:
                self.assertEqual(row[1],{'lid': 1, 'ugid': 1, 'vid': 1, 'alias': 'tmax', 'did': 1, 'year': 2000, 'value': 0.7203244934421581, 'month': 1, 'variable': 'tmax', 'gid': 2, 'time': datetime.datetime(2000, 1, 1, 12, 0), 'tid': 1, 'level': 50, 'day': 1})
            self.assertIsInstance(row[0],MultiPolygon)
            self.assertEqual(len(row),2)
            self.assertEqual(len(row[1]),len(constants.raw_headers))
            
    def test_calculation_iteration(self):
        field = self.get_field(with_value=True,month_count=2)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value+5,
                                              name='tmin',alias='tmin'))
        field.temporal.name_uid = 'tid'
        field.level.name_uid = 'lid'
        field.spatial.geom.name_uid = 'gid'
        
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field,tgd=tgd,alias='my_mean')
        ret = mu.execute()
        
        kwds = copy(field.__dict__)
        kwds.pop('_raw')
        kwds.pop('_variables')
        kwds['temporal'] = tgd
        kwds['variables'] = ret
        cfield = DerivedField(**kwds)
        cfield.temporal.name_uid = 'tid'
        cfield.temporal.name_value = 'time'
        cfield.spatial.name_uid = 'gid'
                        
        sc = ShpCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta,key='state_boundaries',headers=constants.calc_headers)
        for row in sc.iter_geoms('state_boundaries'):
            sp.add_field(row['properties']['UGID'],row['geom'],cfield.variables.keys()[0],
                         cfield,properties=row['properties'])
        for ii,row in enumerate(sp.get_iter_dict()):
            if ii == 0:
                self.assertEqual(row[0].bounds,(-100.5, 39.5, -99.5, 40.5))
                self.assertDictEqual(row[1],{'lid': 1, 'ugid': 1, 'vid': 1, 'cid': 1, 'did': 1, 'year': 2000, 'time': datetime.datetime(2000, 1, 16, 0, 0), 'calc_alias': 'my_mean_tmax', 'value': 0.44808476666433006, 'month': 1, 'alias': 'tmax', 'variable': 'tmax', 'gid': 1, 'calc_key': 'mean', 'tid': 1, 'level': 50, 'day': 16})
            self.assertEqual(len(row),2)
            self.assertEqual(len(row[1]),len(constants.calc_headers))
            
    def test_calculation_iteration_two_calculations(self):
        field = self.get_field(with_value=True,month_count=2)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value+5,
                                              name='tmin',alias='tmin'))
        field.temporal.name_uid = 'tid'
        field.level.name_uid = 'lid'
        field.spatial.geom.name_uid = 'gid'
        
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field,tgd=tgd,alias='my_mean')
        ret = mu.execute()
        thresh = Threshold(field=field,vc=ret,tgd=tgd,alias='a_treshold',parms={'operation':'gte','threshold':0.5})
        ret = thresh.execute()
                
        kwds = copy(field.__dict__)
        kwds.pop('_raw')
        kwds.pop('_variables')
        kwds['temporal'] = tgd
        kwds['variables'] = ret
        cfield = DerivedField(**kwds)
        cfield.temporal.name_uid = 'tid'
        cfield.temporal.name_value = 'time'
        cfield.spatial.name_uid = 'gid'
                        
        sc = ShpCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta,key='state_boundaries',headers=constants.calc_headers)
        for row in sc.iter_geoms('state_boundaries'):
            sp.add_field(row['properties']['UGID'],row['geom'],cfield.variables.keys()[0],
                         cfield,properties=row['properties'])
        
        cids = set()
        for ii,row in enumerate(sp.get_iter_dict()):
            cids.update([row[1]['cid']])
            if ii == 0:
                self.assertEqual(row[0].bounds,(-100.5, 39.5, -99.5, 40.5))
                self.assertDictEqual(row[1],{'lid': 1, 'ugid': 1, 'vid': 1, 'cid': 1, 'did': 1, 'year': 2000, 'time': datetime.datetime(2000, 1, 16, 0, 0), 'calc_alias': 'my_mean_tmax', 'value': 0.44808476666433006, 'month': 1, 'alias': 'tmax', 'variable': 'tmax', 'gid': 1, 'calc_key': 'mean', 'tid': 1, 'level': 50, 'day': 16})
            self.assertEqual(len(row),2)
            self.assertEqual(len(row[1]),len(constants.calc_headers))
        self.assertEqual(ii+1,2*2*2*3*4*51*4)
        self.assertEqual(len(cids),4)
        
    def test_multivariate_iteration(self):
        field = self.get_field(with_value=True,month_count=1)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value+5,
                                              name='tmin',alias='tmin'))
        field.temporal.name_uid = 'tid'
        field.level.name_uid = 'lid'
        field.spatial.geom.name_uid = 'gid'
        
        div = Divide(field=field,parms={'arr1':'tmin','arr2':'tmax'},alias='some_division')
        ret = div.execute()
        
        cfield = DerivedMultivariateField(variables=ret,realization=field.realization,temporal=field.temporal,level=field.level,
                 spatial=field.spatial,meta=field.meta,uid=field.uid)
        cfield.spatial.name_uid = 'gid'
                
        sc = ShpCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta,key='state_boundaries',headers=constants.multi_headers)
        for row in sc.iter_geoms('state_boundaries'):
            sp.add_field(row['properties']['UGID'],row['geom'],cfield.variables.keys()[0],
                         cfield,properties=row['properties'])
        
        for ii,row in enumerate(sp.get_iter_dict()):
            if ii == 0:
                self.assertDictEqual(row[1],{'lid': 1, 'ugid': 1, 'cid': 1, 'did': None, 'year': 2000, 'time': datetime.datetime(2000, 1, 1, 12, 0), 'calc_alias': 'some_division', 'value': 12.989774984574424, 'month': 1, 'gid': 1, 'calc_key': 'divide', 'tid': 1, 'level': 50, 'day': 1})
        self.assertEqual(ii+1,2*31*2*3*4*51)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
