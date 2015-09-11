from collections import OrderedDict
import os
from copy import copy, deepcopy
import datetime

import numpy as np
import fiona
from numpy.core.multiarray import ndarray
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.base import BaseGeometry
from shapely.geometry.multipolygon import MultiPolygon

from ocgis.api.collection import SpatialCollection, AbstractCollection, get_ugeom_attribute
from ocgis.interface.base.crs import CoordinateReferenceSystem, Spherical, WGS84, CFWGS84
from ocgis.test.base import TestBase
from ocgis.util.addict import Dict
from ocgis.util.geom_cabinet import GeomCabinet
from ocgis import constants, SpatialDimension
from ocgis.calc.library.statistics import Mean
from ocgis.interface.base.variable import Variable
from ocgis.interface.base.field import DerivedField, DerivedMultivariateField, \
    Field
from ocgis.calc.library.math import Divide
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
from ocgis.calc.library.thresholds import Threshold


class Test(TestBase):
    def test_get_ugeom_attribute(self):
        ugeom = {1: None}

        def getter(v):
            return v.hello

        self.assertEqual(ugeom, get_ugeom_attribute(ugeom, getter))

        class Foo(object):
            def __init__(self):
                self.hello = 5

        ugeom = {4: Foo()}
        properties = get_ugeom_attribute(ugeom, getter)
        self.assertEqual({4: 5}, properties)
        self.assertIsInstance(properties, OrderedDict)


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
        sc = GeomCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta, key='state_boundaries')
        for row in sc.iter_geoms('state_boundaries', as_spatial_dimension=True):
            sp.add_field(field, ugeom=row)
        return sp

    def get_collection_for_write_ugeom(self, crs):
        pt1 = Point(1, 2)
        pt2 = Point(4, 5)

        props1 = [('UGID', 10), ('STATE_FIPS', '06'), ('ID', 25.0), ('STATE_NAME', 'California'), ('STATE_ABBR', 'CA')]
        props2 = [('UGID', 11), ('STATE_FIPS', '08'), ('ID', 26.0), ('STATE_NAME', 'Ontario'), ('STATE_ABBR', 'CB')]

        schema = {'geometry': 'Point',
                  'properties': OrderedDict([('UGID', 'int'),
                                             ('STATE_FIPS', 'str:2'),
                                             ('ID', 'float'),
                                             ('STATE_NAME', 'str'),
                                             ('STATE_ABBR', 'str:2')])}

        def get_sdim(geom, props, crs):
            record = {'geom': geom, 'properties': OrderedDict(props)}
            return SpatialDimension.from_records([record], schema, crs=crs)

        sdim1, sdim2 = [get_sdim(g, p, crs) for g, p in zip([pt1, pt2], [props1, props2])]

        coll = SpatialCollection(crs=crs)
        coll.ugeom[10] = sdim1
        coll.ugeom[11] = sdim2

        pdtype = [('UGID', '<i8'), ('STATE_FIPS', '|S2'), ('ID', '<f8'), ('STATE_NAME', '|S80'), ('STATE_ABBR', '|S2')]

        return coll, pdtype, pt1, pt2

    def get_spatial_collection_two_geometries(self, pt=None):
        pt = pt or Point(1, 2)
        record1 = {'geom': pt, 'properties': {'ID': 4, 'NAME': 'hello'}}
        record2 = {'geom': pt, 'properties': {'ID': 5, 'NAME': 'another'}}
        schema = {'geometry': 'Point', 'properties': {'ID': 'int', 'NAME': 'str'}}
        sd1 = SpatialDimension.from_records([record1], schema, uid='ID')
        sd2 = SpatialDimension.from_records([record2], schema, uid='ID')
        sc = SpatialCollection()
        field = self.get_field()
        sc.add_field(field, ugeom=sd1)
        sc.add_field(field, ugeom=sd2)
        return sc

    def test_init(self):
        sp = self.get_collection()
        self.assertIsInstance(sp, AbstractCollection)
        self.assertIsNone(sp.headers)
        self.assertEqual(len(sp), 51)
        self.assertIsInstance(sp.geoms[25], MultiPolygon)
        self.assertIsInstance(sp.properties[25], ndarray)
        self.assertEqual(sp[25]['tmax'].variables['tmax'].value.shape, (2, 31, 2, 3, 4))

        sc = SpatialCollection()
        self.assertIsNone(sc.crs)

    def test_add_field(self):
        sc = SpatialCollection()
        field = self.get_field()
        sc.add_field(field)
        self.assertIsInstance(sc[1][field.name], Field)
        with self.assertRaises(ValueError):
            sc.add_field(field)
        field2 = deepcopy(field)
        field2.name = 'another'
        sc.add_field(field2)
        self.assertIsInstance(sc[1]['another'], Field)
        sc.add_field(field, name='hiding')
        self.assertIsInstance(sc[1]['hiding'], Field)
        self.assertIsNone(sc.ugeom[1])

        record = {'geom': Point(1, 2), 'properties': {'ID': 4, 'NAME': 'hello'}}
        schema = {'geometry': 'Point', 'properties': {'ID': 'int', 'NAME': 'str'}}
        sd = SpatialDimension.from_records([record], schema, uid='ID')
        sc.add_field(field, ugeom=sd)
        self.assertEqual(sc.keys(), [1, 4])
        self.assertEqual(sc.ugeom[4].geom.point.value[0, 0], record['geom'])

        sc = SpatialCollection()
        sc.add_field(field)
        self.assertEqual(field.uid, 1)
        field.uid = 10
        sc = SpatialCollection()
        sc.add_field(field)
        self.assertEqual(field.uid, 10)
        field2 = deepcopy(field)
        field2.spatial.crs = Spherical()
        with self.assertRaises(ValueError):
            # coordinate systems must be same
            sc.add_field(field2, name='hover')

        sc = SpatialCollection()
        sc.add_field(None, name='food')
        self.assertIsNone(sc[1]['food'])

    def test_calculation_iteration(self):
        field = self.get_field(with_value=True, month_count=2)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value + 5, name='tmin', alias='tmin'))
        field.temporal.name_uid = 'tid'
        field.level.name_uid = 'lid'
        field.spatial.geom.name_uid = 'gid'

        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field, tgd=tgd, alias='my_mean', dtype=np.float64, add_parents=True)
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

        sc = GeomCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta, key='state_boundaries', headers=constants.HEADERS_CALC)
        for row in sc.iter_geoms('state_boundaries', as_spatial_dimension=True):
            sp.add_field(cfield, ugeom=row)
        for ii, row in enumerate(sp.get_iter_dict(melted=True)):
            if ii == 0:
                self.assertEqual(row[0].bounds, (-100.5, 39.5, -99.5, 40.5))
                self.assertDictEqual(row[1], {'lid': 1, 'ugid': 1, 'vid': 1, 'cid': 1, 'did': 1, 'year': 2000,
                                              'time': datetime.datetime(2000, 1, 16, 0, 0),
                                              'calc_alias': 'my_mean_tmax', 'value': 0.44808476666433006, 'month': 1,
                                              'alias': 'tmax', 'variable': 'tmax', 'gid': 1, 'calc_key': 'mean',
                                              'tid': 1, 'level': 50, 'day': 16})
            self.assertEqual(len(row), 2)
            self.assertEqual(len(row[1]), len(constants.HEADERS_CALC))

    def test_calculation_iteration_two_calculations(self):
        field = self.get_field(with_value=True, month_count=2)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value + 5, name='tmin', alias='tmin'))
        field.temporal.name_uid = 'tid'
        field.level.name_uid = 'lid'
        field.spatial.geom.name_uid = 'gid'

        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field, tgd=tgd, alias='my_mean', dtype=np.float64, add_parents=True)
        ret = mu.execute()
        thresh = Threshold(field=field, vc=ret, tgd=tgd, alias='a_treshold', add_parents=True,
                           parms={'operation': 'gte', 'threshold': 0.5})
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

        sc = GeomCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta, key='state_boundaries', headers=constants.HEADERS_CALC)
        for row in sc.iter_geoms('state_boundaries', as_spatial_dimension=True):
            sp.add_field(cfield, ugeom=row)

        cids = set()
        for ii, row in enumerate(sp.get_iter_dict(melted=True)):
            cids.update([row[1]['cid']])
            if ii == 0:
                self.assertEqual(row[0].bounds, (-100.5, 39.5, -99.5, 40.5))
                self.assertDictEqual(row[1], {'lid': 1, 'ugid': 1, 'vid': 1, 'cid': 1, 'did': 1, 'year': 2000,
                                              'time': datetime.datetime(2000, 1, 16, 0, 0),
                                              'calc_alias': 'my_mean_tmax', 'value': 0.44808476666433006, 'month': 1,
                                              'alias': 'tmax', 'variable': 'tmax', 'gid': 1, 'calc_key': 'mean',
                                              'tid': 1, 'level': 50, 'day': 16})
            self.assertEqual(len(row), 2)
            self.assertEqual(len(row[1]), len(constants.HEADERS_CALC))
        self.assertEqual(ii + 1, 2 * 2 * 2 * 3 * 4 * 51 * 4)
        self.assertEqual(len(cids), 4)

    def test_crs(self):
        sc = SpatialCollection()
        self.assertIsNone(sc.crs)
        field = self.get_field(crs=CFWGS84())
        sc.add_field(field)
        self.assertEqual(sc.crs, CFWGS84())
        self.assertIsNone(sc._crs)

        sc = SpatialCollection(crs=Spherical())
        self.assertEqual(sc.crs, Spherical())

        sc = SpatialCollection()
        field = self.get_field()
        field.spatial = None
        sc.add_field(field)
        self.assertIsNone(sc.crs)

    def test_geoms(self):
        pt = Point(1, 2)
        sc = self.get_spatial_collection_two_geometries(pt=pt)
        self.assertEqual({4: pt, 5: pt}, sc.geoms)

        sc = SpatialCollection()
        sc.ugeom[1] = None
        self.assertEqual({1: None}, sc.geoms)

    def test_get_iter_dict(self):
        field = self.get_field(with_value=True)
        new_var = deepcopy(field.variables.first())
        new_var.alias = 'hi'
        field.variables.add_variable(new_var, assign_new_uid=True)
        coll = field.as_spatial_collection()
        rows = list(coll.get_iter_dict())
        self.assertEqual(len(rows[4]), 2)
        self.assertIsInstance(rows[5], tuple)
        self.assertEqual(len(rows), 1488)
        self.assertEqual(len(list(coll.get_iter_dict(melted=True))), 1488 * 2)

        # test headers applied for non-melted iteration
        keywords = dict(melted=[False, True],
                        use_upper_keys=['NULL', False, True])
        for k in self.iter_product_keywords(keywords):
            headers = ['time', 'tmax']
            coll = field.as_spatial_collection(headers=headers)
            if k.use_upper_keys is True:
                headers = [xx.upper() for xx in headers]
            kwargs = Dict(melted=k.melted)
            if k.use_upper_keys != 'NULL':
                kwargs.use_upper_keys = k.use_upper_keys
            itr = coll.get_iter_dict(**kwargs)
            for ii, row in enumerate(itr):
                if ii == 3:
                    break
                self.assertIsInstance(row[0], BaseGeometry)
                self.assertIsInstance(row[1], OrderedDict)
                self.assertEqual(row[1].keys(), headers)

        # test ugid always in dictionary
        coll = field.as_spatial_collection()
        row = coll.get_iter_dict(melted=False).next()[1]
        self.assertEqual(row[constants.HEADERS.ID_SELECTION_GEOMETRY], 1)

    def test_get_iter_melted(self):
        sp = self.get_collection()
        for row in sp.get_iter_melted():
            self.assertEqual(set(['ugid', 'field_alias', 'field', 'variable_alias', 'variable']), set(row.keys()))
            self.assertIsInstance(row['ugid'], int)
            self.assertIsInstance(row['field_alias'], basestring)
            self.assertIsInstance(row['field'], Field)
            self.assertIsInstance(row['variable_alias'], basestring)
            self.assertIsInstance(row['variable'], Variable)

    def test_iteration_methods(self):
        field = self.get_field(with_value=True)

        field.temporal.name_uid = 'tid'
        field.level.name_uid = 'lid'
        field.spatial.geom.name_uid = 'gid'
        field.spatial.name_uid = 'gid'

        sc = GeomCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta, key='state_boundaries', headers=constants.HEADERS_RAW)
        for row in sc.iter_geoms('state_boundaries', as_spatial_dimension=True):
            sp.add_field(field, ugeom=row)
        for ii, row in enumerate(sp.get_iter_dict(melted=True)):
            if ii == 1:
                self.assertDictEqual(row[1], {'lid': 1, 'ugid': 1, 'vid': 1, 'alias': 'tmax', 'did': 1, 'year': 2000,
                                              'value': 0.7203244934421581, 'month': 1, 'variable': 'tmax', 'gid': 2,
                                              'time': datetime.datetime(2000, 1, 1, 12, 0), 'tid': 1, 'level': 50,
                                              'day': 1})
            self.assertIsInstance(row[0], MultiPolygon)
            self.assertEqual(len(row), 2)
            self.assertEqual(len(row[1]), len(constants.HEADERS_RAW))

    def test_multivariate_iteration(self):
        field = self.get_field(with_value=True, month_count=1)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value + 5,
                                              name='tmin', alias='tmin'))
        field.temporal.name_uid = 'tid'
        field.level.name_uid = 'lid'
        field.spatial.geom.name_uid = 'gid'

        div = Divide(field=field, parms={'arr1': 'tmin', 'arr2': 'tmax'}, alias='some_division',
                     dtype=np.float64)
        ret = div.execute()

        cfield = DerivedMultivariateField(variables=ret, realization=field.realization, temporal=field.temporal,
                                          level=field.level,
                                          spatial=field.spatial, meta=field.meta, uid=field.uid)
        cfield.spatial.name_uid = 'gid'

        sc = GeomCabinet()
        meta = sc.get_meta('state_boundaries')
        sp = SpatialCollection(meta=meta, key='state_boundaries', headers=constants.HEADERS_MULTI)
        for row in sc.iter_geoms('state_boundaries', as_spatial_dimension=True):
            sp.add_field(cfield, ugeom=row)

        for ii, row in enumerate(sp.get_iter_dict(melted=True)):
            if ii == 0:
                self.assertDictEqual(row[1], {'lid': 1, 'ugid': 1, 'cid': 1, 'did': None, 'year': 2000,
                                              'time': datetime.datetime(2000, 1, 1, 12, 0),
                                              'calc_alias': 'some_division', 'value': 12.989774984574424, 'month': 1,
                                              'gid': 1, 'calc_key': 'divide', 'tid': 1, 'level': 50, 'day': 1})
        self.assertEqual(ii + 1, 2 * 31 * 2 * 3 * 4 * 51)

    def test_properties(self):
        sc = self.get_spatial_collection_two_geometries()
        self.assertAsSetEqual(sc.keys(), [4, 5])
        self.assertEqual(sc.properties.values()[0].dtype.names, ('ID', 'NAME'))

        sc = SpatialCollection()
        sc.ugeom[1] = None
        self.assertEqual({1: None}, sc.properties)

    def test_write_ugeom(self):
        keywords = dict(crs=[None, Spherical()],
                        driver=[None, 'GeoJSON'],
                        with_properties=[True, False])

        for ctr, k in enumerate(self.iter_product_keywords(keywords)):
            if k.driver is None:
                path = os.path.join(self.current_dir_output, '{0}.shp'.format(ctr))
            else:
                path = os.path.join(self.current_dir_output, '{0}.json'.format(ctr))

            coll, pdtype, pt1, pt2 = self.get_collection_for_write_ugeom(k.crs)

            try:
                if k.driver is None:
                    coll.write_ugeom(path)
                else:
                    coll.write_ugeom(path, driver=k.driver)
            except ValueError:
                self.assertIsNone(k.crs)
                continue

            with fiona.open(path) as fcoll:
                records = list(fcoll)
                fmeta = fcoll.meta

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]['properties'].keys(), [xx[0] for xx in pdtype])
            for xx, pt in zip(records, [pt1, pt2]):
                self.assertTrue(pt.almost_equals(shape(xx['geometry'])))
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
        coll, pdtype, pt1, pt2 = self.get_collection_for_write_ugeom(WGS84())
        coll.ugeom[11].geom.point.value[0, 0] = MultiPoint([pt1, pt2])
        coll.write_ugeom(path=path)
        with fiona.open(path) as fcoll:
            records = list(fcoll)
        self.assertAsSetEqual([xx['geometry']['type'] for xx in records], ['MultiPoint'])
