import os
import itertools
from copy import deepcopy
from collections import OrderedDict
from datetime import datetime as dt
import datetime

import numpy as np
from shapely import wkb
import fiona
from shapely import wkt
from shapely.geometry import shape, Point
from shapely.ops import cascaded_union

from ocgis import constants, SpatialCollection, GeomCabinet
from ocgis import RequestDataset
from ocgis.constants import NAME_UID_FIELD, NAME_UID_DIMENSION_LEVEL
from ocgis.interface.base.attributes import Attributes
from ocgis.interface.base.crs import WGS84, Spherical
from ocgis.util.helpers import get_date_list, make_poly
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.interface.base.dimension.spatial import SpatialGridDimension, SpatialDimension
from ocgis.interface.base.field import Field, DerivedField
from ocgis.test.base import TestBase, nc_scope
from ocgis.exc import EmptySubsetError
from ocgis.interface.base.variable import Variable, VariableCollection
from ocgis.interface.base.dimension.temporal import TemporalDimension
from ocgis.util.itester import itr_products_keywords


class AbstractTestField(TestBase):
    def setUp(self):
        np.random.seed(1)
        super(AbstractTestField, self).setUp()

    def get_col(self, bounds=True, with_name=True):
        value = [-100., -99., -98., -97.]
        if bounds:
            bounds = [[v - 0.5, v + 0.5] for v in value]
        else:
            bounds = None
        name = 'longitude' if with_name else None
        col = VectorDimension(value=value, bounds=bounds, name=name)
        return col

    def get_row(self, bounds=True, with_name=True):
        value = [40., 39., 38.]
        if bounds:
            bounds = [[v + 0.5, v - 0.5] for v in value]
        else:
            bounds = None
        name = 'latitude' if with_name else None
        row = VectorDimension(value=value, bounds=bounds, name=name)
        return row

    def get_field(self, with_bounds=True, with_value=False, with_level=True, with_temporal=True, with_realization=True,
                  month_count=1, name='tmax', units='kelvin', field_name=None, crs=None, with_dimension_names=True):

        if with_temporal:
            temporal_start = dt(2000, 1, 1, 12)
            if month_count == 1:
                temporal_stop = dt(2000, 1, 31, 12)
            elif month_count == 2:
                temporal_stop = dt(2000, 2, 29, 12)
            else:
                raise NotImplementedError
            temporal_value = get_date_list(temporal_start, temporal_stop, 1)
            delta_bounds = datetime.timedelta(hours=12)
            if with_bounds:
                temporal_bounds = [[v - delta_bounds, v + delta_bounds] for v in temporal_value]
            else:
                temporal_bounds = None
            dname = 'time' if with_dimension_names else None
            temporal = TemporalDimension(value=temporal_value, bounds=temporal_bounds, name=dname)
            t_shape = temporal.shape[0]
        else:
            temporal = None
            t_shape = 1

        if with_level:
            level_value = [50, 150]
            if with_bounds:
                level_bounds = [[0, 100], [100, 200]]
            else:
                level_bounds = None
            dname = 'level' if with_dimension_names else None
            level = VectorDimension(value=level_value, bounds=level_bounds, name=dname, units='meters')
            l_shape = level.shape[0]
        else:
            level = None
            l_shape = 1

        with_name = True if with_dimension_names else False
        row = self.get_row(bounds=with_bounds, with_name=with_name)
        col = self.get_col(bounds=with_bounds, with_name=with_name)
        grid = SpatialGridDimension(row=row, col=col)
        spatial = SpatialDimension(grid=grid, crs=crs)
        row_shape = row.shape[0]
        col_shape = col.shape[0]

        if with_realization:
            dname = 'realization' if with_dimension_names else None
            realization = VectorDimension(value=[1, 2], name=dname)
            r_shape = realization.shape[0]
        else:
            realization = None
            r_shape = 1

        if with_value:
            value = np.random.rand(r_shape, t_shape, l_shape, row_shape, col_shape)
            data = None
        else:
            value = None
            data = 'foo'

        var = Variable(name, units=units, data=data, value=value)
        vc = VariableCollection(variables=var)
        field = Field(variables=vc, temporal=temporal, level=level, realization=realization, spatial=spatial,
                      name=field_name)

        return field


class TestField(AbstractTestField):
    def test_init(self):
        for b, wv in itertools.product([True, False], [True, False]):
            field = self.get_field(with_bounds=b, with_value=wv, with_dimension_names=False)
            self.assertEqual(field.name_uid, NAME_UID_FIELD)
            self.assertEqual(field.level.name, 'level')
            self.assertEqual(field.level.name_uid, NAME_UID_DIMENSION_LEVEL)
            self.assertEqual(field.spatial.grid.row.name, 'yc')
            with self.assertRaises(NotImplementedError):
                list(field)
            self.assertIsInstance(field, Attributes)
            self.assertEqual(field.attrs, OrderedDict())
            self.assertFalse(field.regrid_destination)
            ref = field.shape
            self.assertEqual(ref, (2, 31, 2, 3, 4))
            with self.assertRaises(AttributeError):
                field.value
            self.assertIsInstance(field.variables, VariableCollection)
            self.assertIsInstance(field.variables['tmax'], Variable)
            if wv:
                self.assertIsInstance(field.variables['tmax'].value, np.ma.MaskedArray)
                self.assertEqual(field.variables['tmax'].value.shape, field.shape)
            else:
                with self.assertRaises(Exception):
                    field.variables['tmax'].value

    def test_init_empty(self):
        with self.assertRaises(ValueError):
            Field()

    def test_as_spatial_collection(self):
        field = self.get_field(with_value=True)
        coll = field.as_spatial_collection()
        self.assertIsInstance(coll, SpatialCollection)
        self.assertIsInstance(coll[1][field.name], Field)
        self.assertIsNone(coll.properties[1])

    def test_crs(self):
        field = self.get_field(with_value=True)
        self.assertIsNone(field.spatial.crs)
        self.assertIsNone(field.crs)
        field.spatial.crs = WGS84()
        self.assertEqual(field.crs, WGS84())

    def test_deepcopy(self):
        field = self.get_field(with_value=True)
        deepcopy(field)

    def test_fancy_indexing(self):
        field = self.get_field(with_value=True)
        sub = field[:, (3, 5, 10, 15), :, :, :]
        self.assertEqual(sub.shape, (2, 4, 2, 3, 4))
        self.assertNumpyAll(sub.variables['tmax'].value, field.variables['tmax'].value[:, (3, 5, 10, 15), :, :, :])

        sub = field[:, (3, 15), :, :, :]
        self.assertEqual(sub.shape, (2, 2, 2, 3, 4))
        self.assertNumpyAll(sub.variables['tmax'].value, field.variables['tmax'].value[:, (3, 15), :, :, :])

        sub = field[:, 3:15, :, :, :]
        self.assertEqual(sub.shape, (2, 12, 2, 3, 4))
        self.assertNumpyAll(sub.variables['tmax'].value, field.variables['tmax'].value[:, 3:15, :, :, :])

    def test_getitem(self):
        field = self.get_field(with_value=True)
        with self.assertRaises(IndexError):
            field[0]
        sub = field[0, 0, 0, 0, 0]
        self.assertEqual(sub.shape, (1, 1, 1, 1, 1))
        self.assertEqual(sub.variables['tmax'].value.shape, (1, 1, 1, 1, 1))

        # test with a one-dimensional variable
        field = self.get_field(with_value=True)
        sub = field[0, 0, 0, 0, :]
        for variable in sub.variables.itervalues():
            self.assertEqual(variable.value.shape, sub.shape)
        sub2 = field[0, 0, 0, 0, 2]
        for variable in sub2.variables.itervalues():
            self.assertEqual(variable.value.shape, sub2.shape)

    def test_getitem_general(self):
        """Test slicing on different types of fields."""

        ibounds = [True, False]
        ivalue = [True, False]
        ilevel = [True, False]
        itemporal = [True, False]
        irealization = [True, False]
        for ib, iv, il, it, ir in itertools.product(ibounds, ivalue, ilevel, itemporal, irealization):
            field = self.get_field(with_bounds=ib, with_value=iv, with_level=il, with_temporal=it, with_realization=ir)

            if il:
                self.assertEqual(field.shape[2], 2)
            else:
                self.assertEqual(field.shape[2], 1)

            # # try a bad slice
            with self.assertRaises(IndexError):
                field[0]

            # # now good slices

            # # if data is loaded prior to slicing then memory is shared
            field.spatial.geom.point.value
            field_slc = field[:, :, :, :, :]
            self.assertTrue(np.may_share_memory(field.spatial.grid.value, field_slc.spatial.grid.value))
            self.assertTrue(np.may_share_memory(field.spatial.geom.point.value, field_slc.spatial.geom.point.value))

            field_value = field.variables['tmax']._value
            field_slc_value = field_slc.variables['tmax']._value
            try:
                self.assertNumpyAll(field_value, field_slc_value)
            except AttributeError:
                # with no attached value to the field, the private value will be nones
                if iv is None:
                    self.assertIsNone(field_value)
                    self.assertIsNone(field_slc_value)

            if iv == True:
                self.assertTrue(np.may_share_memory(field_value, field_slc_value))
            else:
                self.assertEqual(field_slc_value, None)

            field_slc = field[0, 0, 0, 0, 0]
            self.assertEqual(field_slc.shape, (1, 1, 1, 1, 1))
            if iv:
                self.assertEqual(field_slc.variables['tmax'].value.shape, (1, 1, 1, 1, 1))
                self.assertNumpyAll(field_slc.variables['tmax'].value,
                                    np.ma.array(field.variables['tmax'].value[0, 0, 0, 0, 0]).reshape(1, 1, 1, 1, 1))
            else:
                self.assertEqual(field_slc.variables['tmax']._value, None)
                self.assertEqual(field_slc.variables['tmax']._value, field.variables['tmax']._value)

    def test_getitem_specific(self):
        field = self.get_field(with_value=True)
        field_slc = field[:, 0:2, 0, :, :]
        self.assertEqual(field_slc.shape, (2, 2, 1, 3, 4))
        self.assertEqual(field_slc.variables['tmax'].value.shape, (2, 2, 1, 3, 4))
        ref_field_real_slc = field.variables['tmax'].value[:, 0:2, 0, :, :]
        self.assertNumpyAll(ref_field_real_slc.flatten(), field_slc.variables['tmax'].value.flatten())

    def test_get_clip_single_cell(self):
        single = wkt.loads(
            'POLYGON((-97.997731 39.339322,-97.709012 39.292322,-97.742584 38.996888,-97.668726 38.641026,-98.158876 38.708170,-98.340165 38.916316,-98.273021 39.218463,-97.997731 39.339322))')
        field = self.get_field(with_value=True)
        for b in [True, False]:
            ret = field.get_clip(single, use_spatial_index=b)
            self.assertEqual(ret.shape, (2, 31, 2, 1, 1))
            self.assertEqual(ret.spatial.grid._value.sum(), -59.0)
            self.assertTrue(ret.spatial.geom.polygon.value[0, 0].almost_equals(single))
            self.assertEqual(ret.spatial.uid, np.array([[7]]))

            self.assertEqual(ret.spatial.geom.point.value.shape, ret.spatial.geom.polygon.shape)
            ref_pt = ret.spatial.geom.point.value[0, 0]
            ref_poly = ret.spatial.geom.polygon.value[0, 0]
            self.assertTrue(ref_poly.intersects(ref_pt))

    def test_get_clip_irregular(self):
        for wv in [True, False]:
            single = wkt.loads(
                'POLYGON((-99.894355 40.230645,-98.725806 40.196774,-97.726613 40.027419,-97.032258 39.942742,-97.681452 39.626613,-97.850806 39.299194,-98.178226 39.643548,-98.844355 39.920161,-99.894355 40.230645))')
            field = self.get_field(with_value=wv)
            for b in [True, False]:
                ret = field.get_clip(single, use_spatial_index=b)
                self.assertEqual(ret.shape, (2, 31, 2, 2, 4))
                unioned = cascaded_union([geom for geom in ret.spatial.geom.polygon.value.compressed().flat])
                self.assertAlmostEqual(unioned.area, single.area)
                self.assertAlmostEqual(unioned.bounds, single.bounds)
                self.assertAlmostEqual(unioned.exterior.length, single.exterior.length)
                self.assertAlmostEqual(ret.spatial.weights[1, 2], 0.064016424)
                self.assertAlmostEqual(ret.spatial.weights.sum(), 1.7764349271673896)
                if not wv:
                    with self.assertRaises(NotImplementedError):
                        ret.variables['tmax'].value

    def test_get_fiona_dict(self):
        field = self.get_field(with_value=True, crs=WGS84())
        _, arch = field.get_iter().next()
        target = Field.get_fiona_dict(field, arch)
        self.assertAsSetEqual(target.keys(), ['crs', 'fconvert', 'schema'])

    def test_get_iter(self):
        field = self.get_field(with_value=True)
        rows = list(field.get_iter())
        self.assertEqual(len(rows), 2 * 31 * 2 * 3 * 4)
        self.assertEqual(len(rows[0]), 2)
        self.assertEqual(rows[100][0].bounds, (-100.5, 38.5, -99.5, 39.5))
        real = {'vid': 1, 'ub_time': datetime.datetime(2000, 1, 6, 0, 0),
                'year': 2000, 'gid': 5, 'ub_level': 100,
                'rid': 1, 'realization': 1, 'lb_level': 0,
                'variable': 'tmax', 'month': 1, 'lb_time': datetime.datetime(2000, 1, 5, 0, 0), 'day': 5,
                'level': 50, 'did': None, 'value': 0.32664490177209615, 'alias': 'tmax', 'lid': 1,
                'time': datetime.datetime(2000, 1, 5, 12, 0), 'tid': 5, 'name': 'tmax', 'ugid': 1}
        self.assertAsSetEqual(rows[100][1].keys(), real.keys())
        for k, v in rows[100][1].iteritems():
            self.assertEqual(real[k], v)
        self.assertEqual(set(field.variables['tmax'].value.flatten().tolist()), set([r[1]['value'] for r in rows]))

        # Test without names.
        field = self.get_field(with_value=True, with_dimension_names=False)
        rows = list(field.get_iter())
        self.assertAsSetEqual(rows[10][1].keys(),
                              ['lid', 'name', 'vid', 'ub_time', 'did', 'lb_level', 'time', 'year', 'value', 'month',
                               'alias', 'tid', 'ub_level', 'rlz', 'variable', 'gid', 'rid', 'level', 'lb_time', 'day',
                               'ugid'])

        # Test not melted.
        field = self.get_field(with_value=True)
        other_variable = deepcopy(field.variables.first())
        other_variable.alias = 'two'
        other_variable.value *= 2
        field.variables.add_variable(other_variable, assign_new_uid=True)
        rows = list(field.get_iter(melted=False))
        self.assertEqual(len(rows), 1488)
        for row in rows:
            attrs = row[1]
            # Test variable aliases are in the row dictionaries.
            for variable in field.variables.itervalues():
                self.assertIn(variable.alias, attrs)

        # Test for upper keys.
        field = self.get_field(with_value=True)[0, 0, 0, 0, 0]
        for row in field.get_iter(use_upper_keys=True):
            for key in row[1].keys():
                self.assertTrue(key.isupper())

        # Test passing limiting headers.
        field = self.get_field(with_value=True)
        headers = ['time', 'tid']
        for _, row in field.get_iter(headers=headers):
            self.assertEqual(row.keys(), headers)

        # Test passing a selection geometry identifier.
        field = self.get_field(with_value=True)[0, 0, 0, 0, 0]
        record = {'geom': Point(1, 2), 'properties': {'HI': 50, 'goodbye': 'forever'}}
        ugeom = SpatialDimension.from_records([record], uid='HI')
        _, row = field.get_iter(ugeom=ugeom).next()
        self.assertEqual(row['HI'], 50)

        # Test value keys.
        field = self.get_field(with_value=True)[0, 0, 0, 0, 0]
        fill = np.ma.array(np.zeros(2, dtype=[('a', float), ('b', float)]))
        value = np.ma.array(np.zeros(field.shape, dtype=object), mask=False)
        value.data[0, 0, 0, 0, 0] = fill
        field.variables['tmax']._value = value
        value_keys = ['a', 'b']
        _, row = field.get_iter(value_keys=value_keys, melted=True)
        for vk in value_keys:
            self.assertIn(vk, row[1])

    def test_get_iter_spatial_only(self):
        """Test with only a spatial dimension."""

        field = self.get_field(with_temporal=False, with_level=False, with_realization=False, with_value=True)
        self.assertIsNone(field.level)
        rows = list(field.get_iter())
        for xx in ['lid', 'level']:
            self.assertNotIn(xx, rows[0][1].keys())
        self.assertEqual(len(rows), 12)

    def test_get_intersects(self):
        # test with vector geometries
        path = GeomCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(path)
        field = rd.get()
        polygon = wkb.loads(
            '\x01\x06\x00\x00\x00\x03\x00\x00\x00\x01\x03\x00\x00\x00\x01\x00\x00\x00Y\x00\x00\x00\x06]\xcf\xdfo\xd4Q\xc0AV:L\xd7\xe2D@i\x856A\xbf\xd5Q\xc0^`\xe7\x0ch\xe4D@\xd8\xcb\xd4e\x1c\xd6Q\xc0\xf7\x8a\xf1\xab\x15\xe8D@*}\xf3#i\xd5Q\xc0\xda\x02\x0b\xc7\xcf\xedD@P\xbd\xdch\xeb\xd5Q\xc0R\xacZ\xab\x19\xf0D@Hd\x0bIQ\xd5Q\xc0K\x9e\xe3\'\xb1\xf2D@\xd4\xcd\xb4\xb0\x92\xd8Q\xc0^\xbf\x93a\xb8\xf1D@{u\xecSy\xd8Q\xc0o\xc6\x82\x80X\xfdD@\x80t%\xb5;\xd8Q\xc0\x84\x84\x0e\\\xc1\x01E@)\xe1\xbd\xe5\xd5\xdfQ\xc0Jp\xdd6/\x01E@\xf2\xd9\xdb\xaa\x0f\xf3Q\xc0dX\xfc\x0f\x8c\x00E@\x83\xbd\xf9\x8aY\xf3Q\xc0\x1e\xcd\x14\x15M\x02E@[\xbbY\x02\x14\x06R\xc0Z\xe9\xc5dM\x03E@\xc1\xbd\xad\xe5\xb9\x08R\xc0p\x8d\x1a\'a\x03E@\xfbw`\x10| R\xc0z3\xfd&\xf0\x03E@\xe4\x98\x9a\xf8\x8e$R\xc0r.\xe4%\xdb\x03E@\xb2\x07\xf7\xf7=%R\xc0cs\xba\x07\xc4\x02E@\x81\xa6\xef\x9b\xe6&R\xc0\xf8qV\x1f\xeb\x02E@\xa6~rz\x02\'R\xc0*\xf6\x9b\x9d\xe8\x03E@\xb3\xefU\x92`0R\xc0\xa4SJ\x1cU\x04E@\xb41\x00\xf4\x1f1R\xc0tK0\x05G\x00E@\r\x98h\xdbT4R\xc0\x1c\xc0$\xc5\xa3\xffD@\x03\xa2\xcd\xbc@4R\xc0\xc0\xe2)\xf8I\x04E@\x0cKd\xddc@R\xc0`\xfau\xf4\x9b\x04E@\x03\xd4\x96\xa3\xebBR\xc0\x82\x8en\xd1\xa5\x04E@\x1e\xca\xef\xa0\xfd^R\xc0\x02\xc7\xf9!\x12\x06E@P\xda\xb7\xff\xec_R\xc0\x12\xc1\xa68\xea\tE@2\xe2\x9d\xe7sVR\xc0\x1e\xfb\xe8\xd2\x9b@E@\xb2:3\x0f\x84PR\xc0H-0\xd7~_E@\xf8\x1c\xed\xafBAR\xc0&\xc0\xe3N\xc5^E@\x8c\x1e\x1ec\x12;R\xc0\xe4R\xa1\xf4a^E@5s\nW+\x1dR\xc0\x84\x8b\xf9\xba\xe8\\E@^L\x19*\xea\x11R\xc0\xf8\x16RF8\\E@\x9eZ\xcb\xa9\x88\xfbQ\xc0\xa8\xdb\'\xd6\x85ZE@?+\xbd\t\xa9\xf9Q\xc0~K\x9d\xd6IZE@\x0f\x8f\x0bda\xd2Q\xc0\xee\x90\xcb\xd5kYE@\xec\xa8\x91\x81\'\xd0Q\xc0?\x7fM\xd7\xef\\E@64"\x03d\xcfQ\xc0\xd6\x99\x80\xd2,_E@\xa9\xbb\x11\x1d\xed\xcbQ\xc0\xca\x7f \xb3\x8f^E@4r\xfa\x81\x96\xcbQ\xc0\x9c\x17\xed,VgE@\x9d\xee\xf0\xfa\xb7\xc7Q\xc0\xb2\xd4\x9fq\xbdhE@S\xe2r42\xc4Q\xc0\n\x1c\xe1\xef\xf3fE@!\x83y\x95\xa0\xc1Q\xc0O\x14\xf1.\xf3lE@\xb0\xf2^,\xf7\xbaQ\xc0\xfe\x89\x10\x93LqE@\xf6\x0f\xa9\xa7z\xb9Q\xc0FI\x942\x85qE@\xca>\xfb$b\xb6Q\xc0:\xa9\x7f\xda\x84nE@\x99f=\x9d\x16\xb4Q\xc0b6z\xff\xfbnE@\xf8\x1c\xcc*W\xafQ\xc0\xc8\x1fmU\xeeTE@\xcc\xb7\t\xfa\xf6\xa5Q\xc0p\xa3_"\xbaRE@Qf[{\x8a\xa8Q\xc0J\x07l\x06\x94JE@b\x8c\x1fK\n\xb4Q\xc0 \xffz\xa0\xf1EE@6\xa8\xee\xcf0\xb9Q\xc0:P\xe3MZ9E@\xf1H\xcc\xd5z\xbdQ\xc0\x82\x19u\xaaX7E@n\x18\xea\xb6/\xc2Q\xc0\x175gx\x8f$E@U\xadT\xc7\x15\xbbQ\xc0J_B\xaa\x04\x1eE@\xcb\x88\xa5\x86!\xb9Q\xc0z\xf3\xde\xa1\x04"E@\x1f\x04\x08@\xc7\xb4Q\xc0\x8co.NX!E@?\xce\x01\xf8\x92\xb1Q\xc0\x10\xf6\x91r\xd3\x1fE@nD\xd5\x08\xe8\xabQ\xc0\n\xf6\x9b\xf4\x9a\x13E@\x069\x91\xd5\x98\xa7Q\xc0\xc4\xd5\x10\xa1\xed\xfbD@L\x1b\xef\xe6\x94\xa2Q\xc0$\xad\x14j)\xf7D@\xa4\xe1T\xc3i\xa2Q\xc09\x04\xa28#\xe7D@\xd8\xea\xfa\xce\x1a\x9bQ\xc0g\x01\x88\x04/\xdfD@\xde\xc6#\x80\x86\x91Q\xc0\xff\x88\x16w_\xdcD@\x14hp\x07\xd5\x95Q\xc0(\x95L\xb3\x1c\xdbD@\xa1 \xbe\xf7"\x8dQ\xc0\x19(\xa4\x9a5\xdbD@\x9bKt\xce:\x81Q\xc0\xb8\xe3\x9b\xd3\x08\xe4D@\x175%X\x07\x80Q\xc0\x16\xe7\x88\xe3\x9c\xedD@\x8b\x0e\x11\x8cn\x86Q\xc0pZ\xae\xe7G\x00E@\xef\x08`YT\x90Q\xc0\x90\xc7\xcc\xfd\xb1\x07E@P\x02\xa0Q\xa5\x88Q\xc0\xc2\xf2\xd2~G\tE@\xf4\x84\xd0\xeb:\x83Q\xc0\xbd@\xb0\xbe]\x03E@\xb0G/\xf7\xb4}Q\xc0\xcdI<]\xb9\xf3D@\xae\x04l\xe9\xbczQ\xc0@\x99+wB\xe2D@5\xb6ME\x15}Q\xc0,\xc8f\x8f\xf3\xd5D@\xe6[z\x8br\x99Q\xc0x0\x10\xbdh\xceD@\xc8\x01\xfe\xf2\xb4\x9bQ\xc0\xf0\xb6\xcf\xc6\xed\xc8D@*\x82\xc1\xe3\xc6\xa8Q\xc0\x06n9O\x18\xc5D@\x0ei\x7f\x87\x8d\xaaQ\xc0z\x0cy./\xc7D@\xd8\x12$+\xaa\xa7Q\xc0\x9b\x93\x1bU)\xdeD@\x8f\x07\xb59\xb9\xb5Q\xc0\x9c\xcf\x98t7\xd0D@/\xb9#\xa1\x18\xb9Q\xc0\x94\xc3X\n$\xd1D@\xb2\xeeYk\x13\xc0Q\xc0\xa22ko\x93\xc2D@\xd0PQ\x18\x7f\xc7Q\xc0W\xa9\xe8\xaa\x1c\xbfD@0v(\x9f\t\xc9Q\xc0\xfa\x02g\xff\xdf\xd3D@H\x9dJF\xb9\xccQ\xc0\x0c\xc0\x99\x19\xd9\xd6D@\xb1\xfd\r\x8c\xa7\xceQ\xc0\xa4m\x9f\xba\x95\xdaD@\x96/\xfdo\x10\xd1Q\xc0Xm3\x97\xf7\xdfD@\x06]\xcf\xdfo\xd4Q\xc0AV:L\xd7\xe2D@\x01\x03\x00\x00\x00\x01\x00\x00\x00\x0f\x00\x00\x00\xc0\xb6\x07]\xad\xa6Q\xc02\xbc\x8c5\xff\xb6D@\xa9xP\x1aU\xa4Q\xc02\x92"\xe9v\xbbD@\xf2\xb9_\x96a\xa3Q\xc0o\xeeb\xfbl\xb5D@"\xfdj\xd8\xda\xa4Q\xc0\xc0\x90\x1a;\x84\xb4D@\x186*V\xf8\xa0Q\xc0Z"\x89M\x07\xb3D@\xc7j=\xf0\x1c\x9fQ\xc0\xed3hH\xb8\xabD@\x881\xcdxF\xafQ\xc0zJ`\x9a\xc5\xaaD@L7j\xfbB\xb1Q\xc0h>\xfc?*\xa6D@\xe3\xd2!\xca\x02\xb6Q\xc0\xac!n\xe7\x9e\xacD@p\xbc\xa4\xe0\x14\xb2Q\xc0\x7f\x12\xffI\x1f\xadD@\xbb\x0c\x1b\xdbV\xb1Q\xc0\\\xcd\xe5\xf4\x98\xa9D@\x80\xe8\xd2\xfc\x1c\xb0Q\xc0\x14C\x00\xed\xea\xb0D@\xc7\'\xb0 \xb8\xaaQ\xc0W\x81:c;\xbaD@\x86\x9c\x9f\x1e\xc6\xa6Q\xc02\xfc\xe8\xc4\xc1\xbcD@\xc0\xb6\x07]\xad\xa6Q\xc02\xbc\x8c5\xff\xb6D@\x01\x03\x00\x00\x00\x01\x00\x00\x00\r\x00\x00\x00k\x97\xa4\xa3\x07\x82Q\xc0@\xa7\xf3]\xed\xa7D@\xb8\xaa\xa0\xa1j\x80Q\xc0\x98+\xd84\x92\xa9D@T@x%\xb4\x81Q\xc0x\xd7\x92\xb5)\xabD@R8\x8a\xc8\x9b\x85Q\xc0\xa8\xc2\x93 \xff\xa5D@]r\xdd\x055\x82Q\xc0\xfcUH\x92\xc3\xacD@\xf7"J%\'\x83Q\xc0\xb0)@\xca+\xb2D@\xb8\xfb\xdf\x9e\xd2}Q\xc0\xe1A\x12\x00\xbf\xa5D@\x0f\xd7\xa3\xfd\xfa}Q\xc0\x99\x08\xc8\x84;\xa0D@\x85\xbc\xcfF\x99\x86Q\xc0\x10\xdd1\xf0\x7f\x9eD@eg\xec/\xa6\x8dQ\xc0\x99\xdf\xe4\x16\x96\xa2D@\'\xdc\xad\x10A\x8dQ\xc0\x1ag\xa1\xa7\xa4\xa5D@$\xc4\x04\x8aC\x86Q\xc0Uu\xb2l\x89\xa3D@k\x97\xa4\xa3\x07\x82Q\xc0@\xa7\xf3]\xed\xa7D@')
        ret = field.get_intersects(polygon)
        self.assertTrue(polygon.almost_equals(ret.spatial.geom.polygon.value.compressed()[0]))

    def test_get_intersects_domain_polygon(self):
        regular = make_poly((36.61, 41.39), (-101.41, -95.47))
        field = self.get_field(with_value=True)
        for b in [True, False]:
            ret = field.get_intersects(regular, use_spatial_index=b)
            self.assertNumpyAll(ret.variables['tmax'].value, field.variables['tmax'].value)
            self.assertNumpyAll(field.spatial.grid.value, ret.spatial.grid.value)

    def test_get_intersects_irregular_polygon(self):
        irregular = wkt.loads(
            'POLYGON((-100.106049 38.211305,-99.286894 38.251591,-99.286894 38.258306,-99.286894 38.258306,-99.260036 39.252035,-98.769886 39.252035,-98.722885 37.734583,-100.092620 37.714440,-100.106049 38.211305))')
        keywords = dict(b=[True, False],
                        with_corners=[True, False])
        for k in itr_products_keywords(keywords, as_namedtuple=True):
            field = self.get_field(with_value=True)
            if k.with_corners:
                field.spatial.grid.corners
            ret = field.get_intersects(irregular, use_spatial_index=k.b)
            self.assertEqual(ret.shape, (2, 31, 2, 2, 2))
            self.assertNumpyAll(ret.variables['tmax'].value.mask[0, 2, 1, :, :],
                                np.array([[True, False], [False, False]]))
            self.assertEqual(ret.spatial.uid.data[ret.spatial.get_mask()][0], 5)
            if k.with_corners:
                self.assertNumpyAll(ret.spatial.grid.corners.mask, np.array([
                    [[[True, True, True, True], [False, False, False, False]],
                     [[False, False, False, False], [False, False, False, False]]],
                    [[[True, True, True, True], [False, False, False, False]],
                     [[False, False, False, False], [False, False, False, False]]]]))
            else:
                self.assertIsNone(ret.spatial.grid._corners)

    def test_get_intersects_single_bounds_row(self):
        field = self.get_field(with_value=True)
        sub = field[:, 0, :, 0, 0]
        irregular = wkt.loads(
            'POLYGON((-100.106049 38.211305,-99.286894 38.251591,-99.286894 38.258306,-99.286894 38.258306,-99.260036 39.252035,-98.769886 39.252035,-98.722885 37.734583,-100.092620 37.714440,-100.106049 38.211305))')
        # # the intersects operations is empty. this was testing that contiguous
        ## bounds check fails appropriately with a single bounds row.
        with self.assertRaises(EmptySubsetError):
            sub.get_intersects(irregular)

    def test_get_iter_two_variables(self):
        field = self.get_field(with_value=True)
        field2 = self.get_field(with_value=True)
        var2 = field2.variables['tmax']
        var2.alias = 'tmax2'
        var2._value = var2._value + 3
        field.variables.add_variable(deepcopy(var2), assign_new_uid=True)
        aliases = set([row[1]['alias'] for row in field.get_iter(melted=True)])
        self.assertEqual(set(['tmax', 'tmax2']), aliases)

        vids = []
        for row in field.get_iter():
            vids.append(row[1]['vid'])
            if row[1]['alias'] == 'tmax2':
                self.assertTrue(row[1]['value'] > 3)
        self.assertEqual(set(vids), set([1, 2]))

    def test_get_spatially_aggregated_all(self):
        for wv in [True, False]:
            field = self.get_field(with_value=wv)
            try:
                agg = field.get_spatially_aggregated()
            except NotImplementedError:
                if not wv:
                    continue
                else:
                    raise
            self.assertNotEqual(field.spatial.grid, None)
            self.assertEqual(agg.spatial.grid, None)
            self.assertEqual(agg.shape, (2, 31, 2, 1, 1))
            self.assertNumpyAll(field.variables['tmax'].value, agg._raw.variables['tmax'].value)
            self.assertTrue(np.may_share_memory(field.variables['tmax'].value, agg._raw.variables['tmax'].value))

            to_test = field.variables['tmax'].value[0, 0, 0, :, :].mean()
            self.assertNumpyAll(to_test, agg.variables['tmax'].value[0, 0, 0, 0, 0])

    def test_get_spatially_aggregated_irregular(self):
        single = wkt.loads(
            'POLYGON((-99.894355 40.230645,-98.725806 40.196774,-97.726613 40.027419,-97.032258 39.942742,-97.681452 39.626613,-97.850806 39.299194,-98.178226 39.643548,-98.844355 39.920161,-99.894355 40.230645))')
        field = self.get_field(with_value=True)
        for b in [True, False]:
            ret = field.get_clip(single, use_spatial_index=b)
            agg = ret.get_spatially_aggregated()
            to_test = agg.spatial.geom.polygon.value[0, 0]
            self.assertAlmostEqual(to_test.area, single.area)
            self.assertAlmostEqual(to_test.bounds, single.bounds)
            self.assertAlmostEqual(to_test.exterior.length, single.exterior.length)

    def test_get_time_subset_by_function(self):
        field = self.get_field(with_value=True)

        def _func_(value, bounds=None):
            return [2, 3, 4]

        ret = field.get_time_subset_by_function(_func_)
        self.assertEqual(ret.shape, (2, 3, 2, 3, 4))

    def test_iter(self):
        field = self.get_field(with_value=True)
        with self.assertRaises(ValueError):
            list(field.iter())

        field = self.get_field(with_value=True, with_realization=False, with_level=False, with_temporal=False)
        other = deepcopy(field.variables.first())
        other.alias = 'tmax2'
        other.uid = 2
        field.variables.add_variable(other)
        gids = []
        for row in field.iter():
            self.assertIsInstance(row, OrderedDict)
            self.assertEqual(row.keys(), ['geom', 'did', 'gid', 'tmax', 'tmax2'])
            for variable in field.variables.itervalues():
                self.assertIn(variable.alias, row)
            gids.append(row[field.spatial.name_uid])
        self.assertEqual(len(gids), len(set(gids)))
        self.assertEqual(len(gids), 12)

    def test_name(self):
        field = self.get_field(field_name='foo')
        self.assertEqual(field.name, 'foo')
        field.name = 'svelt'
        self.assertEqual(field.name, 'svelt')

    def test_name_none_one_variable(self):
        field = self.get_field(field_name=None)
        self.assertEqual(field.name, field.variables.values()[0].alias)

    def test_name_none_two_variables(self):
        field = self.get_field()
        field2 = self.get_field()
        var2 = field2.variables['tmax']
        var2.alias = 'tmax2'
        field.variables.add_variable(var2, assign_new_uid=True)
        self.assertEqual(field.name, 'tmax_tmax2')

    def test_loading_from_source_spatial_bounds(self):
        """Test row bounds may be set to None when loading from source."""

        field = self.test_data.get_rd('cancm4_tas').get()
        field.spatial.grid.row.bounds
        field.spatial.grid.row.bounds = None
        self.assertIsNone(field.spatial.grid.row.bounds)

    def test_should_regrid(self):
        field = self.get_field()
        self.assertFalse(field._should_regrid)

    def test_shape_as_dict(self):
        field = self.get_field(with_value=False)
        to_test = field.shape_as_dict
        for variable in field.variables.values():
            self.assertEqual(variable._value, None)
        self.assertEqual(to_test, {'Y': 3, 'X': 4, 'Z': 2, 'R': 2, 'T': 31})

    def test_subsetting(self):
        for wv in [True, False]:
            field = self.get_field(with_value=wv)
            self.assertNotIsInstance(field.temporal.value, np.ma.MaskedArray)

            temporal_start = dt(2000, 1, 1, 12)
            temporal_stop = dt(2000, 1, 31, 12)
            ret = field.temporal.get_between(temporal_start, temporal_stop)
            self.assertIsInstance(ret, VectorDimension)
            self.assertNumpyAll(ret.value, field.temporal.value)
            self.assertNumpyAll(ret.bounds, field.temporal.bounds)

            ret = field.get_between('temporal', temporal_start, temporal_stop)
            self.assertIsInstance(ret, Field)
            self.assertEqual(ret.shape, field.shape)
            if wv:
                self.assertNumpyAll(field.variables['tmax'].value, ret.variables['tmax'].value)
            else:
                with self.assertRaises(NotImplementedError):
                    ret.variables['tmax'].value

            # # try empty subset
            with self.assertRaises(EmptySubsetError):
                field.get_between('level', 100000, 2000000000)

            ret = field.get_between('realization', 1, 1)
            self.assertEqual(ret.shape, (1, 31, 2, 3, 4))
            if wv:
                self.assertNumpyAll(ret.variables['tmax'].value, field.variables['tmax'].value[0:1, :, :, :, :])

            ret = field.get_between('temporal', dt(2000, 1, 15), dt(2000, 1, 30))
            self.assertEqual(ret.temporal.value[0], dt(2000, 1, 15, 12))
            self.assertEqual(ret.temporal.value[-1], dt(2000, 1, 30, 12))

    def test_variables(self):
        row = VectorDimension(value=[5, 6])
        col = VectorDimension(value=[7, 8])
        grid = SpatialGridDimension(row=row, col=col)
        sdim = SpatialDimension(grid=grid)
        temporal = TemporalDimension(value=[5000])
        field = Field(spatial=sdim, temporal=temporal)
        self.assertIsInstance(field.variables, VariableCollection)
        self.assertEqual(len(field.variables), 0)
        self.assertEqual(field.shape, (1, 1, 1, 2, 2))
        with self.assertRaises(ValueError):
            field.variables = 'foo'

    def test_write_fiona(self):

        keywords = dict(with_realization=[True, False],
                        with_level=[True, False],
                        with_temporal=[True, False],
                        driver=['ESRI Shapefile', 'GeoJSON'],
                        melted=[False, True])

        for ii, k in enumerate(self.iter_product_keywords(keywords)):
            path = os.path.join(self.current_dir_output, '{0}'.format(ii))
            field = self.get_field(with_value=True, crs=WGS84(), with_dimension_names=False,
                                   with_realization=k.with_realization, with_level=k.with_level,
                                   with_temporal=k.with_temporal)
            newvar = deepcopy(field.variables.first())
            newvar.alias = 'newvar'
            newvar.value += 10
            field.variables.add_variable(newvar, assign_new_uid=True)
            field = field[:, 0:2, :, 0:2, 0:2]

            field.write_fiona(path, driver=k.driver, melted=k.melted)

            with fiona.open(path) as source:
                records = list(source)

            if k.melted:
                dd = {a: [] for a in field.variables.keys()}
                for r in records:
                    dd[r['properties']['alias']].append(r['properties']['value'])
                for kk, v in dd.iteritems():
                    self.assertAlmostEqual(np.mean(v), field.variables[kk].value.mean(), places=6)
            else:
                for alias in field.variables.keys():
                    values = [r['properties'][alias] for r in records]
                    self.assertAlmostEqual(np.mean(values), field.variables[alias].value.mean(), places=6)

            n = reduce(lambda x, y: x * y, field.shape)
            if k.melted:
                n *= len(field.variables)
            self.assertEqual(n, len(records))

        # test with a point abstraction
        field = self.get_field(with_value=True, crs=WGS84())
        field = field[0, 0, 0, 0, 0]
        field.spatial.abstraction = 'point'
        path = self.get_temporary_file_path('foo.shp')
        field.write_fiona(path)
        with fiona.open(path) as source:
            gtype = source.meta['schema']['geometry']
            self.assertEqual(gtype, 'Point')

        # test with a fake object passed in as a fiona object. this should raise an exception as the method will attempt
        # to use the object instead of creating a new collection. the object should not be closed when done.

        class DontHateMe(Exception):
            pass

        class WriteMe(Exception):
            pass

        class Nothing(object):
            def close(self):
                raise DontHateMe()

            def write(self, *args, **kwargs):
                raise WriteMe()

        with self.assertRaises(WriteMe):
            field.write_fiona(path, fobject=Nothing())

        # Test all geometries are accounted for as well as properties.
        path = GeomCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(path)
        field = rd.get()
        out = self.get_temporary_file_path('foo.shp')
        field.write_fiona(out)

        with fiona.open(out, 'r') as source:
            for record in source:
                target = shape(record['geometry'])
                self.assertEqual(record['properties'].keys(),
                                 [u'UGID', u'STATE_FIPS', u'ID', u'STATE_NAME', u'STATE_ABBR'])
                found = False
                for geom in field.spatial.abstraction_geometry.value.flat:
                    if target.almost_equals(geom):
                        found = True
                        break
                self.assertTrue(found)

        # Test with upper keys.
        field = self.get_field(with_value=True, crs=WGS84())[0, 0, 0, 0, 0]
        path = self.get_temporary_file_path('what.shp')
        field.write_fiona(path=path, use_upper_keys=True)
        with fiona.open(path) as source:
            for row in source:
                for key in row['properties']:
                    self.assertTrue(key.isupper())

        # test with upper keys
        field = self.get_field(with_value=True, crs=WGS84())[0, 0, 0, 0, 0]
        path = self.get_temporary_file_path('what2.shp')
        headers = ['time', 'tid']
        field.write_fiona(path=path, headers=headers)
        with fiona.open(path) as source:
            self.assertEqual(source.meta['schema']['properties'].keys(), headers)

        # test passing a ugid
        field = self.get_field(with_value=True, crs=WGS84())[0, 0, 0, 0, 0]
        path = self.get_temporary_file_path('what3.shp')
        record = {'geom': Point(1, 2), 'properties': {'ugid': 10}}
        ugeom = SpatialDimension.from_records([record], uid='ugid')
        field.write_fiona(path=path, ugeom=ugeom)
        with fiona.open(path) as source:
            for row in source:
                self.assertEqual(row['properties'][constants.HEADERS.ID_SELECTION_GEOMETRY], 10)

    def test_write_to_netcdf_dataset(self):
        keywords = dict(file_only=[False, True],
                        second_variable_alias=[None, 'tmin_alias'],
                        with_realization=[False, True],
                        remove_dimension_names=[False, True],
                        crs=[None, Spherical()],
                        with_level=[True, False])
        path = os.path.join(self.current_dir_output, 'foo.nc')

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            field = self.get_field(with_value=True, with_realization=k.with_realization, crs=k.crs,
                                   with_level=k.with_level)

            if k.remove_dimension_names:
                try:
                    field.level.name = None
                except AttributeError:
                    self.assertFalse(k.with_level)
                field.temporal.name = None
                field.spatial.grid.row.name = None
                field.spatial.grid.col.name = None

            # add another variable
            value = np.random.rand(*field.shape)
            second_variable_name = 'tmin'
            second_variable_alias = k.second_variable_alias or second_variable_name
            variable = Variable(value=value, name=second_variable_name, alias=k.second_variable_alias)
            variable.attrs['open'] = 'up'
            field.variables.add_variable(variable, assign_new_uid=True)

            # add some attributes
            field.attrs['foo'] = 'some information'
            field.attrs['another'] = 'some more information'

            with nc_scope(path, 'w') as ds:
                try:
                    field.write_netcdf(ds, file_only=k.file_only)
                except ValueError:
                    self.assertTrue(k.with_realization)
                    self.assertIsNotNone(field.realization)
                    continue

            with nc_scope(path) as ds:
                self.assertEqual(ds.another, 'some more information')
                try:
                    variable_names = ['time', 'time_bounds', 'latitude', 'latitude_bounds', 'longitude',
                                      'longitude_bounds', 'tmax', second_variable_alias]
                    dimension_names = ['time', 'bounds', 'latitude', 'longitude']
                    if k.crs is not None:
                        variable_names.append(k.crs.name)
                    if k.with_level:
                        variable_names += ['level', 'level_bounds']
                        dimension_names.append('level')
                    self.assertEqual(set(ds.variables.keys()), set(variable_names))
                    self.assertEqual(set(ds.dimensions.keys()), set(dimension_names))
                except AssertionError:
                    self.assertTrue(k.remove_dimension_names)
                    variable_names = ['time', 'time_bounds', 'yc', 'yc_bounds', 'xc', 'xc_bounds', 'tmax',
                                      second_variable_alias]
                    dimension_names = ['time', 'bounds', 'yc', 'xc']
                    if k.crs is not None:
                        variable_names.append(k.crs.name)
                    if k.with_level:
                        variable_names += ['level', 'level_bounds']
                        dimension_names.append('level')
                    self.assertEqual(set(ds.variables.keys()), set(variable_names))
                    self.assertEqual(set(ds.dimensions.keys()), set(dimension_names))
                nc_second_variable = ds.variables[second_variable_alias]

                try:
                    for field_variable in field.variables.itervalues():
                        self.assertEqual(ds.variables[field_variable.alias].grid_mapping,
                                         k.crs.name)
                except AttributeError:
                    self.assertIsNone(k.crs)

                self.assertEqual(nc_second_variable.open, 'up')
                try:
                    self.assertNumpyAll(nc_second_variable[:], value.squeeze())
                except AssertionError:
                    self.assertTrue(k.file_only)
                    self.assertTrue(nc_second_variable[:].mask.all())
                self.assertEqual(ds.variables['tmax'].units, field.variables['tmax'].units)
                self.assertEqual(nc_second_variable.units, '')

            new_field = RequestDataset(path).get()
            self.assertEqual(new_field.variables.keys(), ['tmax', second_variable_alias])
            if k.with_level:
                level_shape = 2
            else:
                level_shape = 1
            self.assertEqual(new_field.shape, (1, 31, level_shape, 3, 4))

    def test_write_to_netcdf_dataset_scale_offset(self):
        """Test with a scale and offset in the attributes."""

        var = Variable(value=np.random.rand(1, 1, 1, 3, 4), attrs={'scale_value': 2, 'add_offset': 10}, name='tas')
        grid = SpatialGridDimension(value=np.ma.array(np.random.rand(2, 3, 4)))
        sdim = SpatialDimension(grid=grid)
        field = Field(variables=var, spatial=sdim)
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with self.nc_scope(path, 'w') as ds:
            field.write_netcdf(ds)
        with self.nc_scope(path, 'r') as out:
            var_out = out.variables['tas'][:]
        target = var_out.reshape(*var.shape)
        self.assertNumpyAllClose(var.value.data, target)

    def test_write_to_netcdf_dataset_with_metadata(self):
        """Test writing to netCDF with a source metadata dictionary attached and data loaded from file."""

        rd = self.test_data.get_rd('narccap_lambert_conformal')
        field = rd.get()[:, 0:31, :, 20:30, 30:40]
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with nc_scope(path, 'w') as ds:
            field.write_netcdf(ds)
            self.assertSetEqual(set(ds.variables.keys()), {u'time', 'time_bnds', u'yc', u'xc', u'Lambert_Conformal',
                                                           'pr'})
            self.assertGreater(len(ds.__dict__), 0)
            self.assertGreater(len(ds.variables['time'].__dict__), 0)

    def test_write_to_netcdf_dataset_without_row_column_on_grid(self):
        """Test writing a field without rows and columns on the grid."""

        field = self.get_field(with_value=True, with_realization=False)
        field.spatial.grid.value
        field.spatial.grid.corners
        field.spatial.grid.row = None
        field.spatial.grid.col = None
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with nc_scope(path, 'w') as ds:
            field.write_netcdf(ds)
            self.assertAsSetEqual(ds.variables.keys(), ['time', 'time_bounds', 'level', 'level_bounds',
                                                        constants.DEFAULT_NAME_ROW_COORDINATES,
                                                        constants.DEFAULT_NAME_COL_COORDINATES, 'yc_corners',
                                                        'xc_corners', 'tmax'])
            self.assertAsSetEqual(ds.dimensions.keys(),
                                  ['time', 'bounds', 'level', constants.DEFAULT_NAME_ROW_COORDINATES,
                                   constants.DEFAULT_NAME_COL_COORDINATES, constants.DEFAULT_NAME_CORNERS_DIMENSION])

        # test with name on the grid
        field = self.get_field(with_value=True, with_realization=False)
        field.spatial.grid.value
        field.spatial.grid.corners
        field.spatial.grid.name_row = 'nr'
        field.spatial.grid.name_col = 'nc'
        field.spatial.grid.row = None
        field.spatial.grid.col = None
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with nc_scope(path, 'w') as ds:
            field.write_netcdf(ds)
            self.assertAsSetEqual(ds.variables.keys(),
                                  ['time', 'time_bounds', 'level', 'level_bounds', 'nr', 'nc', 'nr_corners',
                                   'nc_corners', 'tmax'])
            self.assertAsSetEqual(ds.dimensions.keys(),
                                  ['time', 'bounds', 'level', 'nr', 'nc', constants.DEFAULT_NAME_CORNERS_DIMENSION])

    def test_write_to_netcdf_dataset_without_temporal(self):
        """Test without a temporal dimensions."""

        path = os.path.join(self.current_dir_output, 'foo.nc')
        field = self.get_field(with_temporal=False, with_realization=False, with_value=True, with_level=False)
        with self.nc_scope(path, 'w') as ds:
            field.write_netcdf(ds)
        with self.nc_scope(path) as ds:
            vars = ds.variables.keys()
            self.assertAsSetEqual(vars, [u'latitude', u'latitude_bounds', u'longitude', u'longitude_bounds', u'tmax'])


class TestDerivedField(AbstractTestField):
    def test_init(self):
        field = self.get_field(with_value=True, month_count=2)
        tgd = field.temporal.get_grouping(['month'])
        new_data = np.random.rand(2, 2, 2, 3, 4)
        mu = Variable(name='mu', value=new_data)
        df = DerivedField(variables=mu, temporal=tgd, spatial=field.spatial,
                          level=field.level, realization=field.realization)
        self.assertIsInstance(df, Field)
        self.assertIsInstance(df.temporal.value[0], datetime.datetime)
        self.assertEqual(df.temporal.value.tolist(),
                         [datetime.datetime(2000, 1, 16, 0, 0), datetime.datetime(2000, 2, 16, 0, 0)])
        self.assertEqual(df.temporal.bounds[1, 1], datetime.datetime(2000, 3, 1, 0, 0))
