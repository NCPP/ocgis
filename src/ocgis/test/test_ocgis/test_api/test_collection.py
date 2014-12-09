import os
import datetime
from copy import copy, deepcopy

import fiona
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
import numpy as np

from ocgis.api.collection import SpatialCollection, AbstractCollection
from ocgis.interface.base.crs import CoordinateReferenceSystem, Spherical
from ocgis.test.base import TestBase
from ocgis.util.shp_cabinet import ShpCabinet
from ocgis import constants
from ocgis.calc.library.statistics import Mean
from ocgis.interface.base.variable import Variable
from ocgis.interface.base.field import DerivedField, DerivedMultivariateField,\
    Field
from ocgis.calc.library.math import Divide
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
from ocgis.calc.library.thresholds import Threshold


class TestAbstractCollection(TestBase):
    create_dir = False

    def get_coll(self):
        coll = AbstractCollection()
        coll[2] = 'a'
        coll[1] = 'b'
        return coll

    def test_init(self):
        self.assertIsInstance(self.get_coll(), AbstractCollection)

    def test_storage_uid_next(self):
        coll = self.get_coll()
        coll._storage_id.append(5)
        self.assertEqual(coll._storage_id_next, 6)

    def test_contains(self):
        coll = self.get_coll()
        self.assertTrue(1 in coll)
        self.assertFalse(3 in coll)

    def test_copy(self):
        coll = self.get_coll()
        self.assertEqual(coll, copy(coll))

    def test_deepcopy(self):
        coll = self.get_coll()
        self.assertEqual(coll, deepcopy(coll))

    def test_values(self):
        self.assertEqual(self.get_coll().values(), ['a', 'b'])

    def test_keys(self):
        self.assertEqual(self.get_coll().keys(), [2, 1])

    def test_pop(self):
        self.assertEqual(self.get_coll().pop('hi', 'there'), 'there')
        coll = self.get_coll()
        val = coll.pop(2)
        self.assertEqual(val, 'a')
        self.assertDictEqual(coll, {1: 'b'})

    def test_getitem(self):
        self.assertEqual(self.get_coll()[2], 'a')

    def test_setitem(self):
        coll = self.get_coll()
        coll['a'] = 400
        self.assertEqual(coll['a'], 400)

    def test_repr(self):
        self.assertEqual(repr(self.get_coll()), "AbstractCollection([(2, 'a'), (1, 'b')])")

    def test_str(self):
        coll = self.get_coll()
        self.assertEqual(repr(coll), str(coll))

    def test_first(self):
        self.assertEqual(self.get_coll().first(), 'a')

    def test_update(self):
        coll = self.get_coll()
        coll.update({'t': 'time'})
        self.assertEqual(coll.keys(), [2, 1, 't'])

    def test_iter(self):
        self.assertEqual(list(self.get_coll().__iter__()), [2, 1])


class TestSpatialCollection(AbstractTestField):
    
    def get_collection(self):
        field = self.get_field(with_value=True)
        sc = ShpCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta,key='state_boundaries')
        for row in sc.iter_geoms('state_boundaries'):
            sp.add_field(row['properties']['UGID'], row['geom'], field, properties=row['properties'])
        return sp

    def get_collection_for_write_ugeom(self, crs):
        pt1 = Point(1, 2)
        pt2 = Point(4, 5)
        coll = SpatialCollection(crs=crs)
        ugid1 = 10
        ugid2 = 11
        coll.geoms[ugid1] = pt1
        coll.geoms[ugid2] = pt2
        pvalue1 = [(ugid1, '06', 25.0, 'California', 'CA')]
        pvalue2 = [(ugid2, '08', 26.0, 'Ontario', 'CB')]
        pdtype = [('UGID', '<i8'), ('STATE_FIPS', 'O'), ('ID', '<f8'), ('STATE_NAME', 'O'), ('STATE_ABBR', 'O')]
        properties1 = np.array(pvalue1, dtype=pdtype)
        properties2 = np.array(pvalue2, dtype=pdtype)
        coll.properties[ugid1] = properties1
        coll.properties[ugid2] = properties2
        return coll, pdtype, pt1, pt2

    def test_init(self):
        sp = self.get_collection()
        self.assertEqual(len(sp),51)
        self.assertIsInstance(sp.geoms[25],MultiPolygon)
        self.assertIsInstance(sp.properties[25],dict)
        self.assertEqual(sp[25]['tmax'].variables['tmax'].value.shape,(2, 31, 2, 3, 4))

    def test_calculation_iteration(self):
        field = self.get_field(with_value=True,month_count=2)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value+5,
                                              name='tmin',alias='tmin'))
        field.temporal.name_uid = 'tid'
        field.level.name_uid = 'lid'
        field.spatial.geom.name_uid = 'gid'

        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field,tgd=tgd,alias='my_mean',dtype=np.float64)
        ret = mu.execute()

        kwds = copy(field.__dict__)
        kwds.pop('_raw')
        kwds.pop('_variables')
        kwds.pop('_should_regrid')
        kwds.pop('_has_assigned_coordinate_system')
        kwds.pop('_attrs')
        kwds['name'] = kwds.pop('_name')
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
            sp.add_field(row['properties']['UGID'],row['geom'],cfield,properties=row['properties'])
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
        mu = Mean(field=field,tgd=tgd,alias='my_mean',dtype=np.float64)
        ret = mu.execute()
        thresh = Threshold(field=field,vc=ret,tgd=tgd,alias='a_treshold',parms={'operation':'gte','threshold':0.5})
        ret = thresh.execute()

        kwds = copy(field.__dict__)
        kwds.pop('_raw')
        kwds.pop('_variables')
        kwds.pop('_should_regrid')
        kwds.pop('_has_assigned_coordinate_system')
        kwds.pop('_attrs')
        kwds['name'] = kwds.pop('_name')
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
            sp.add_field(row['properties']['UGID'],row['geom'],cfield,properties=row['properties'])

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

    def test_get_iter_melted(self):
        sp = self.get_collection()
        for row in sp.get_iter_melted():
            self.assertEqual(set(['ugid','field_alias','field','variable_alias','variable']),set(row.keys()))
            self.assertIsInstance(row['ugid'],int)
            self.assertIsInstance(row['field_alias'],basestring)
            self.assertIsInstance(row['field'],Field)
            self.assertIsInstance(row['variable_alias'],basestring)
            self.assertIsInstance(row['variable'],Variable)

    def test_iteration_methods(self):
        field = self.get_field(with_value=True)

        field.temporal.name_uid = 'tid'
        field.level.name_uid = 'lid'
        field.spatial.geom.name_uid = 'gid'
        field.spatial.name_uid = 'gid'

        sc = ShpCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta,key='state_boundaries')
        for row in sc.iter_geoms('state_boundaries'):
            sp.add_field(row['properties']['UGID'],row['geom'],field,properties=row['properties'])
        for ii,row in enumerate(sp.get_iter_dict()):
            if ii == 1:
                self.assertDictEqual(row[1],{'lid': 1, 'ugid': 1, 'vid': 1, 'alias': 'tmax', 'did': 1, 'year': 2000, 'value': 0.7203244934421581, 'month': 1, 'variable': 'tmax', 'gid': 2, 'time': datetime.datetime(2000, 1, 1, 12, 0), 'tid': 1, 'level': 50, 'day': 1})
            self.assertIsInstance(row[0],MultiPolygon)
            self.assertEqual(len(row),2)
            self.assertEqual(len(row[1]),len(constants.raw_headers))

    def test_multivariate_iteration(self):
        field = self.get_field(with_value=True,month_count=1)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value+5,
                                              name='tmin',alias='tmin'))
        field.temporal.name_uid = 'tid'
        field.level.name_uid = 'lid'
        field.spatial.geom.name_uid = 'gid'

        div = Divide(field=field,parms={'arr1':'tmin','arr2':'tmax'},alias='some_division',
                     dtype=np.float64)
        ret = div.execute()

        cfield = DerivedMultivariateField(variables=ret,realization=field.realization,temporal=field.temporal,level=field.level,
                 spatial=field.spatial,meta=field.meta,uid=field.uid)
        cfield.spatial.name_uid = 'gid'

        sc = ShpCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta,key='state_boundaries',headers=constants.multi_headers)
        for row in sc.iter_geoms('state_boundaries'):
            sp.add_field(row['properties']['UGID'],row['geom'],cfield,properties=row['properties'])

        for ii,row in enumerate(sp.get_iter_dict()):
            if ii == 0:
                self.assertDictEqual(row[1],{'lid': 1, 'ugid': 1, 'cid': 1, 'did': None, 'year': 2000, 'time': datetime.datetime(2000, 1, 1, 12, 0), 'calc_alias': 'some_division', 'value': 12.989774984574424, 'month': 1, 'gid': 1, 'calc_key': 'divide', 'tid': 1, 'level': 50, 'day': 1})
        self.assertEqual(ii+1,2*31*2*3*4*51)

    def test_write_ugeom(self):
        keywords = dict(crs=[None, Spherical()],
                        driver=[None, 'GeoJSON'],
                        with_properties=[True, False])

        for ctr, k in enumerate(self.iter_product_keywords(keywords)):
            # print k
            if k.driver is None:
                path = os.path.join(self.current_dir_output, '{0}.shp'.format(ctr))
            else:
                path = os.path.join(self.current_dir_output, '{0}.json'.format(ctr))

            coll, pdtype, pt1, pt2 = self.get_collection_for_write_ugeom(k.crs)

            if k.driver is None:
                coll.write_ugeom(path)
            else:
                coll.write_ugeom(path, driver=k.driver)

            with fiona.open(path) as fcoll:
                records = list(fcoll)
                fmeta = fcoll.meta

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]['properties'].keys(), [xx[0] for xx in pdtype])
            for xx, pt in zip(records, [pt1, pt2]):
                self.assertTrue(pt.almost_equals(shape(xx['geometry'])))
            if k.crs is None:
                actual = CoordinateReferenceSystem(epsg=4326)
            else:
                actual = k.crs
            try:
                self.assertEqual(actual, CoordinateReferenceSystem(value=fmeta['crs']))
            except AssertionError:
                # geojson does not support other coordinate systems
                self.assertEqual(k.driver, 'GeoJSON')

    def test_write_ugeom_fobject(self):
        path = os.path.join(self.current_dir_output, 'foo.shp')
        schema = {'geometry': 'Polygon', 'properties': {}}
        coll, pdtype, pt1, pt2, = self.get_collection_for_write_ugeom(None)
        with fiona.open(path, driver='ESRI Shapefile', schema=schema, mode='w') as fobject:
            # use the improperly configured fobject to attempt to write the collection
            with self.assertRaises(ValueError):
                coll.write_ugeom(fobject=fobject)

    def test_write_ugeom_single_and_multi_geometries(self):
        """
        Test mixing single and multi-geometries.
        """

        path = os.path.join(self.current_dir_output, 'foo.shp')
        coll, pdtype, pt1, pt2 = self.get_collection_for_write_ugeom(None)
        coll.geoms[11] = MultiPoint([pt1, pt2])
        coll.write_ugeom(path=path)
        with fiona.open(path) as fcoll:
            records = list(fcoll)
        self.assertAsSetEqual([xx['geometry']['type'] for xx in records], ['MultiPoint'])
