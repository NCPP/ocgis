from netCDF4 import date2num
import os
import unittest
from datetime import datetime as dt
import datetime
import itertools
from copy import deepcopy
from importlib import import_module
from collections import OrderedDict

import numpy as np
from shapely import wkt
from shapely.ops import cascaded_union

from ocgis import constants
from ocgis import RequestDataset
from ocgis.interface.base.attributes import Attributes
from ocgis.interface.base.crs import WGS84, Spherical
from ocgis.interface.nc.temporal import NcTemporalDimension
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
        super(AbstractTestField,self).setUp()

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
            self.assertEqual(field.level.name, 'level')
            self.assertEqual(field.level.name_uid, 'level_uid')
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
        sub = field[:,(3,5,10,15),:,:,:]
        self.assertEqual(sub.shape,(2,4,2,3,4))
        self.assertNumpyAll(sub.variables['tmax'].value,field.variables['tmax'].value[:,(3,5,10,15),:,:,:])

        sub = field[:,(3,15),:,:,:]
        self.assertEqual(sub.shape,(2,2,2,3,4))
        self.assertNumpyAll(sub.variables['tmax'].value,field.variables['tmax'].value[:,(3,15),:,:,:])

        sub = field[:,3:15,:,:,:]
        self.assertEqual(sub.shape,(2,12,2,3,4))
        self.assertNumpyAll(sub.variables['tmax'].value,field.variables['tmax'].value[:,3:15,:,:,:])

    def test_get_aggregated_all(self):
        for wv in [True,False]:
            field = self.get_field(with_value=wv)
            try:
                agg = field.get_spatially_aggregated()
            except NotImplementedError:
                if not wv:
                    continue
                else:
                    raise
            self.assertNotEqual(field.spatial.grid,None)
            self.assertEqual(agg.spatial.grid,None)
            self.assertEqual(agg.shape,(2,31,2,1,1))
            self.assertNumpyAll(field.variables['tmax'].value,agg._raw.variables['tmax'].value)
            self.assertTrue(np.may_share_memory(field.variables['tmax'].value,agg._raw.variables['tmax'].value))

            to_test = field.variables['tmax'].value[0,0,0,:,:].mean()
            self.assertNumpyAll(to_test,agg.variables['tmax'].value[0,0,0,0,0])

    def test_get_aggregated_irregular(self):
        single = wkt.loads('POLYGON((-99.894355 40.230645,-98.725806 40.196774,-97.726613 40.027419,-97.032258 39.942742,-97.681452 39.626613,-97.850806 39.299194,-98.178226 39.643548,-98.844355 39.920161,-99.894355 40.230645))')
        field = self.get_field(with_value=True)
        for b in [True,False]:
            try:
                ret = field.get_clip(single,use_spatial_index=b)
                agg = ret.get_spatially_aggregated()
                to_test = agg.spatial.geom.polygon.value[0,0]
                self.assertAlmostEqual(to_test.area,single.area)
                self.assertAlmostEqual(to_test.bounds,single.bounds)
                self.assertAlmostEqual(to_test.exterior.length,single.exterior.length)
            except ImportError:
                with self.assertRaises(ImportError):
                    import_module('rtree')

    def test_get_clip_single_cell(self):
        single = wkt.loads('POLYGON((-97.997731 39.339322,-97.709012 39.292322,-97.742584 38.996888,-97.668726 38.641026,-98.158876 38.708170,-98.340165 38.916316,-98.273021 39.218463,-97.997731 39.339322))')
        field = self.get_field(with_value=True)
        for b in [True,False]:
            try:
                ret = field.get_clip(single,use_spatial_index=b)
                self.assertEqual(ret.shape,(2,31,2,1,1))
                self.assertEqual(ret.spatial.grid._value.sum(),-59.0)
                self.assertTrue(ret.spatial.geom.polygon.value[0,0].almost_equals(single))
                self.assertEqual(ret.spatial.uid,np.array([[7]]))

                self.assertEqual(ret.spatial.geom.point.value.shape,ret.spatial.geom.polygon.shape)
                ref_pt = ret.spatial.geom.point.value[0,0]
                ref_poly = ret.spatial.geom.polygon.value[0,0]
                self.assertTrue(ref_poly.intersects(ref_pt))
            except ImportError:
                with self.assertRaises(ImportError):
                    import_module('rtree')

    def test_get_clip_irregular(self):
        for wv in [True,False]:
            single = wkt.loads('POLYGON((-99.894355 40.230645,-98.725806 40.196774,-97.726613 40.027419,-97.032258 39.942742,-97.681452 39.626613,-97.850806 39.299194,-98.178226 39.643548,-98.844355 39.920161,-99.894355 40.230645))')
            field = self.get_field(with_value=wv)
            for b in [True,False]:
                try:
                    ret = field.get_clip(single,use_spatial_index=b)
                    self.assertEqual(ret.shape,(2,31,2,2,4))
                    unioned = cascaded_union([geom for geom in ret.spatial.geom.polygon.value.compressed().flat])
                    self.assertAlmostEqual(unioned.area,single.area)
                    self.assertAlmostEqual(unioned.bounds,single.bounds)
                    self.assertAlmostEqual(unioned.exterior.length,single.exterior.length)
                    self.assertAlmostEqual(ret.spatial.weights[1,2],0.064016424)
                    self.assertAlmostEqual(ret.spatial.weights.sum(),1.776435)
                    if not wv:
                        with self.assertRaises(NotImplementedError):
                            ret.variables['tmax'].value
                except ImportError:
                    with self.assertRaises(ImportError):
                        import_module('rtree')

    def test_get_iter(self):
        field = self.get_field(with_value=True)
        rows = list(field.get_iter())
        self.assertEqual(len(rows), 2 * 31 * 2 * 3 * 4)
        rows[100]['geom'] = rows[100]['geom'].bounds
        real = {'realization_bounds_lower': None, 'vid': 1, 'time_bounds_upper': datetime.datetime(2000, 1, 6, 0, 0),
                'realization_bounds_upper': None, 'year': 2000, 'gid': 5, 'level_bounds_upper': 100,
                'realization_uid': 1, 'realization': 1, 'geom': (-100.5, 38.5, -99.5, 39.5), 'level_bounds_lower': 0,
                'variable': 'tmax', 'month': 1, 'time_bounds_lower': datetime.datetime(2000, 1, 5, 0, 0), 'day': 5,
                'level': 50, 'did': None, 'value': 0.32664490177209615, 'alias': 'tmax', 'level_uid': 1,
                'time': datetime.datetime(2000, 1, 5, 12, 0), 'tid': 5, 'name': 'tmax'}
        for k, v in rows[100].iteritems():
            self.assertEqual(real[k], v)
        self.assertEqual(set(real.keys()), set(rows[100].keys()))
        self.assertEqual(set(field.variables['tmax'].value.flatten().tolist()), set([r['value'] for r in rows]))

        # test without names
        field = self.get_field(with_value=True, with_dimension_names=False)
        rows = list(field.get_iter())
        self.assertAsSetEqual(rows[10].keys(), ['vid', 'gid', 'month', 'year', 'alias', 'geom', 'realization', 'realization_uid', 'time_bounds_lower', 'level_bounds_upper', 'variable', 'day', 'realization_bounds_lower', 'name', 'level', 'did', 'level_bounds_lower', 'value', 'realization_bounds_upper', 'level_uid', 'time', 'tid', 'time_bounds_upper'])

    def test_get_iter_spatial_only(self):
        """Test with only a spatial dimension."""

        field = self.get_field(with_temporal=False, with_level=False, with_realization=False, with_value=True)
        self.assertIsNone(field.level)
        rows = list(field.get_iter())
        for xx in ['lid', 'level']:
            self.assertNotIn(xx, rows[0].keys())
        self.assertEqual(len(rows), 12)

    def test_get_intersects_domain_polygon(self):
        regular = make_poly((36.61,41.39),(-101.41,-95.47))
        field = self.get_field(with_value=True)
        for b in [True,False]:
            try:
                ret = field.get_intersects(regular,use_spatial_index=b)
                self.assertNumpyAll(ret.variables['tmax'].value,field.variables['tmax'].value)
                self.assertNumpyAll(field.spatial.grid.value,ret.spatial.grid.value)
            except ImportError:
                with self.assertRaises(ImportError):
                    import_module('rtree')

    def test_get_intersects_irregular_polygon(self):
        irregular = wkt.loads('POLYGON((-100.106049 38.211305,-99.286894 38.251591,-99.286894 38.258306,-99.286894 38.258306,-99.260036 39.252035,-98.769886 39.252035,-98.722885 37.734583,-100.092620 37.714440,-100.106049 38.211305))')
        keywords = dict(b=[True, False],
                        with_corners=[True, False])
        for k in itr_products_keywords(keywords, as_namedtuple=True):
            try:
                field = self.get_field(with_value=True)
                if k.with_corners:
                    field.spatial.grid.corners
                ret = field.get_intersects(irregular,use_spatial_index=k.b)
                self.assertEqual(ret.shape,(2,31,2,2,2))
                self.assertNumpyAll(ret.variables['tmax'].value.mask[0,2,1,:,:],np.array([[True,False],[False,False]]))
                self.assertEqual(ret.spatial.uid.data[ret.spatial.get_mask()][0],5)
                if k.with_corners:
                    self.assertNumpyAll(ret.spatial.grid.corners.mask, np.array([[[[True, True, True, True], [False, False, False, False]], [[False, False, False, False], [False, False, False, False]]], [[[True, True, True, True], [False, False, False, False]], [[False, False, False, False], [False, False, False, False]]]]))
                else:
                    self.assertIsNone(ret.spatial.grid._corners)
            except ImportError:
                with self.assertRaises(ImportError):
                    import_module('rtree')

    def test_get_intersects_single_bounds_row(self):
        field = self.get_field(with_value=True)
        sub = field[:,0,:,0,0]
        irregular = wkt.loads('POLYGON((-100.106049 38.211305,-99.286894 38.251591,-99.286894 38.258306,-99.286894 38.258306,-99.260036 39.252035,-98.769886 39.252035,-98.722885 37.734583,-100.092620 37.714440,-100.106049 38.211305))')
        ## the intersects operations is empty. this was testing that contiguous
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
        aliases = set([row['alias'] for row in field.get_iter()])
        self.assertEqual(set(['tmax', 'tmax2']), aliases)

        vids = []
        for row in field.get_iter():
            vids.append(row['vid'])
            if row['alias'] == 'tmax2':
                self.assertTrue(row['value'] > 3)
        self.assertEqual(set(vids), set([1, 2]))

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
            self.assertEqual(variable._value,None)
        self.assertEqual(to_test,{'Y': 3, 'X': 4, 'Z': 2, 'R': 2, 'T': 31})

    def test_slicing(self):
        field = self.get_field(with_value=True)
        with self.assertRaises(IndexError):
            field[0]
        sub = field[0,0,0,0,0]
        self.assertEqual(sub.shape,(1,1,1,1,1))
        self.assertEqual(sub.variables['tmax'].value.shape,(1,1,1,1,1))

    def test_slicing_general(self):
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

            ## now good slices

            ## if data is loaded prior to slicing then memory is shared
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

    def test_slicing_specific(self):
        field = self.get_field(with_value=True)
        field_slc = field[:,0:2,0,:,:]
        self.assertEqual(field_slc.shape,(2,2,1,3,4))
        self.assertEqual(field_slc.variables['tmax'].value.shape,(2,2,1,3,4))
        ref_field_real_slc = field.variables['tmax'].value[:,0:2,0,:,:]
        self.assertNumpyAll(ref_field_real_slc.flatten(),field_slc.variables['tmax'].value.flatten())

    def test_subsetting(self):
        for wv in [True,False]:
            field = self.get_field(with_value=wv)
            self.assertNotIsInstance(field.temporal.value,np.ma.MaskedArray)

            temporal_start = dt(2000,1,1,12)
            temporal_stop = dt(2000,1,31,12)
            ret = field.temporal.get_between(temporal_start,temporal_stop)
            self.assertIsInstance(ret,VectorDimension)
            self.assertNumpyAll(ret.value,field.temporal.value)
            self.assertNumpyAll(ret.bounds,field.temporal.bounds)

            ret = field.get_between('temporal',temporal_start,temporal_stop)
            self.assertIsInstance(ret,Field)
            self.assertEqual(ret.shape,field.shape)
            if wv:
                self.assertNumpyAll(field.variables['tmax'].value,ret.variables['tmax'].value)
            else:
                with self.assertRaises(NotImplementedError):
                    ret.variables['tmax'].value

            ## try empty subset
            with self.assertRaises(EmptySubsetError):
                field.get_between('level',100000,2000000000)

            ret = field.get_between('realization',1,1)
            self.assertEqual(ret.shape,(1, 31, 2, 3, 4))
            if wv:
                self.assertNumpyAll(ret.variables['tmax'].value,field.variables['tmax'].value[0:1,:,:,:,:])

            ret = field.get_between('temporal',dt(2000,1,15),dt(2000,1,30))
            self.assertEqual(ret.temporal.value[0],dt(2000,1,15,12))
            self.assertEqual(ret.temporal.value[-1],dt(2000,1,30,12))

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
                    field.write_to_netcdf_dataset(ds, file_only=k.file_only)
                except ValueError:
                    self.assertTrue(k.with_realization)
                    self.assertIsNotNone(field.realization)
                    continue

            with nc_scope(path) as ds:
                self.assertEqual(ds.another, 'some more information')
                try:
                    variable_names = ['time', 'time_bounds', 'latitude', 'latitude_bounds', 'longitude', 'longitude_bounds', 'tmax', second_variable_alias]
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
                    variable_names = ['time', 'time_bounds', 'yc', 'yc_bounds', 'xc', 'xc_bounds', 'tmax', second_variable_alias]
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

    def test_write_to_netcdf_dataset_with_metadata(self):
        """Test writing to netCDF with a source metadata dictionary attached and data loaded from file."""

        rd = self.test_data.get_rd('narccap_lambert_conformal')
        field = rd.get()[:, 0:31, :, 20:30, 30:40]
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with nc_scope(path, 'w') as ds:
            field.write_to_netcdf_dataset(ds)
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
            field.write_to_netcdf_dataset(ds)
            self.assertAsSetEqual(ds.variables.keys(), ['time', 'time_bounds', 'level', 'level_bounds',
                                                        constants.default_name_row_coordinates,
                                                        constants.default_name_col_coordinates, 'yc_corners',
                                                        'xc_corners', 'tmax'])
            self.assertAsSetEqual(ds.dimensions.keys(),
                                  ['time', 'bounds', 'level', constants.default_name_row_coordinates,
                                   constants.default_name_col_coordinates, constants.default_name_corners_dimension])

    def test_write_to_netcdf_dataset_without_temporal(self):
        """Test without a temporal dimensions."""

        path = os.path.join(self.current_dir_output, 'foo.nc')
        field = self.get_field(with_temporal=False, with_realization=False, with_value=True, with_level=False)
        with self.nc_scope(path, 'w') as ds:
            field.write_to_netcdf_dataset(ds)
        with self.nc_scope(path) as ds:
            vars = ds.variables.keys()
            self.assertAsSetEqual(vars, [u'latitude', u'latitude_bounds', u'longitude', u'longitude_bounds', u'tmax'])


class TestDerivedField(AbstractTestField):
    
    def test_init(self):
        field = self.get_field(with_value=True,month_count=2)
        tgd = field.temporal.get_grouping(['month'])
        new_data = np.random.rand(2,2,2,3,4)
        mu = Variable(name='mu',value=new_data)
        df = DerivedField(variables=mu,temporal=tgd,spatial=field.spatial,
                          level=field.level,realization=field.realization)
        self.assertIsInstance(df, Field)
        self.assertIsInstance(df.temporal.value[0],datetime.datetime)
        self.assertEqual(df.temporal.value.tolist(),[datetime.datetime(2000, 1, 16, 0, 0),datetime.datetime(2000, 2, 16, 0, 0)])
        self.assertEqual(df.temporal.bounds[1,1],datetime.datetime(2000, 3, 1, 0, 0))
