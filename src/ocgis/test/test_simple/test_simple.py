import re
import itertools
import os.path
from abc import ABCMeta, abstractproperty
import netCDF4 as nc
import csv
from collections import OrderedDict
import datetime
from copy import deepcopy
from csv import DictReader
import tempfile

import numpy as np
from fiona.crs import from_string
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
import fiona
from shapely.geometry.geo import mapping
from shapely import wkt

from ocgis.test import strings
from ocgis import osr
from ocgis.api.operations import OcgOperations
from ocgis.api.interpreter import OcgInterpreter
from ocgis.api.parms.definition import SpatialOperation
from ocgis.util.helpers import make_poly, project_shapely_geometry
from ocgis import exc, env, constants
from ocgis.test.base import TestBase, nc_scope, attr
import ocgis
from ocgis.exc import ExtentError, DefinitionValidationError
from ocgis.interface.base import crs
from ocgis.interface.base.crs import CoordinateReferenceSystem, WGS84, CFWGS84, WrappableCoordinateReferenceSystem
from ocgis.api.request.base import RequestDataset
from ocgis.test.test_simple.make_test_data import SimpleNcNoLevel, SimpleNc, SimpleMaskNc, \
    SimpleNc360, SimpleNcProjection, SimpleNcNoSpatialBounds, SimpleNcMultivariate
from ocgis.api.parms.definition import OutputFormat
from ocgis.interface.base.field import DerivedMultivariateField
from ocgis.util.itester import itr_products_keywords
from ocgis.util.geom_cabinet import GeomCabinetIterator
from ocgis.util.spatial.fiona_maker import FionaMaker


class TestSimpleBase(TestBase):
    __metaclass__ = ABCMeta

    base_value = None
    return_shp = False
    var = 'foo'

    @abstractproperty
    def nc_factory(self):
        pass

    @abstractproperty
    def fn(self):
        pass

    def setUp(self):
        TestBase.setUp(self)
        self.nc_factory().write()

    def get_dataset(self, time_range=None, level_range=None, time_region=None):
        uri = os.path.join(env.DIR_OUTPUT, self.fn)
        return ({'uri': uri, 'variable': self.var,
                 'time_range': time_range, 'level_range': level_range,
                 'time_region': time_region})

    def get_ops(self, kwds={}, time_range=None, level_range=None):
        dataset = self.get_dataset(time_range, level_range)
        if 'output_format' not in kwds:
            kwds.update({'output_format': 'numpy'})
        kwds.update({'dataset': dataset})
        ops = OcgOperations(**kwds)
        return (ops)

    def get_ret(self, ops=None, kwds={}, shp=False, time_range=None, level_range=None):
        """
        :param ops:
        :type ops: :class:`ocgis.api.operations.OcgOperations`
        :param dict kwds:
        :param bool shp: If ``True``, override output format to shapefile.
        :param time_range:
        :type time_range: list[:class:`datetime.datetime`]
        :param level_range:
        :type level_range: list[int]
        """

        if ops is None:
            ops = self.get_ops(kwds, time_range=time_range, level_range=level_range)
        self.ops = ops
        ret = OcgInterpreter(ops).execute()

        if shp or self.return_shp:
            kwds2 = kwds.copy()
            kwds2.update({'output_format': 'shp'})
            ops2 = OcgOperations(**kwds2)
            OcgInterpreter(ops2).execute()

        return ret

    def make_shp(self):
        ops = OcgOperations(dataset=self.dataset,
                            output_format='shp')
        OcgInterpreter(ops).execute()


@attr('simple')
class TestSimpleNoLevel(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNcNoLevel
    fn = 'test_simple_spatial_no_level_01.nc'

    def test_nc_write_no_level(self):
        ret = self.get_ret(kwds={'output_format': 'nc'})
        ret2 = self.get_ret(kwds={'output_format': 'nc',
                                  'dataset': {'uri': ret, 'variable': 'foo'}, 'prefix': 'level_again'})
        self.assertNcEqual(ret, ret2, ignore_attributes={'global': ['history']})

        ds = nc.Dataset(ret)
        try:
            self.assertTrue('level' not in ds.dimensions)
            self.assertTrue('level' not in ds.variables)
        finally:
            ds.close()


@attr('simple')
class TestSimple(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNc
    fn = 'test_simple_spatial_01.nc'

    def test_meta_attrs_applied(self):
        """Test overloaded metadata attributes are applied to output calculation."""

        calc = [{'func': 'mean', 'name': 'mean', 'meta_attrs': {'this_is': 'something new', 'a_number': 5}}]
        calc_grouping = ['month']
        ret = self.get_ret(kwds={'calc': calc, 'calc_grouping': calc_grouping, 'output_format': 'nc'})
        with nc_scope(ret) as ds:
            var = ds.variables['mean']
            self.assertEqual(var.__dict__['this_is'], 'something new')
            self.assertEqual(var.__dict__['a_number'], 5)

    def test_meta_attrs_eval_function(self):
        """Test metadata attributes applied with evaluation function."""

        calc = [{'func': 'up=foo+5', 'meta_attrs': {'this_is': 'something new', 'a_number': 5}}]
        ret = self.get_ret(kwds={'calc': calc, 'output_format': 'nc'})
        with nc_scope(ret) as ds:
            var = ds.variables['up']
            self.assertEqual(var.__dict__['this_is'], 'something new')
            self.assertEqual(var.__dict__['a_number'], 5)

    def test_selection_geometry_crs_differs(self):
        """Test selection is appropriate when CRS of selection geometry differs from source."""

        dataset = self.get_dataset()
        rd = RequestDataset(**dataset)

        ugeom = 'POLYGON((-104.000538 39.004301,-102.833871 39.215054,-102.833871 39.215054,-102.833871 39.215054,-102.879032 37.882796,-104.136022 37.867742,-104.000538 39.004301))'
        ugeom = wkt.loads(ugeom)
        from_sr = osr.SpatialReference()
        from_sr.ImportFromEPSG(4326)
        to_sr = osr.SpatialReference()
        to_sr.ImportFromProj4(
            '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs')
        ugeom = project_shapely_geometry(ugeom, from_sr, to_sr)
        crs = from_string(to_sr.ExportToProj4())

        # write_geom_dict({1: ugeom}, '/tmp/projected.shp', crs=crs)

        geom = [{'geom': ugeom, 'crs': crs}]
        ops = OcgOperations(dataset=rd, geom=geom)
        ret = ops.execute()

        to_test = ret[1]['foo'].variables['foo'].value[:, 0, 0, :, :]
        actual = np.loads(
            '\x80\x02cnumpy.ma.core\n_mareconstruct\nq\x01(cnumpy.ma.core\nMaskedArray\nq\x02cnumpy\nndarray\nq\x03K\x00\x85q\x04U\x01btRq\x05(K\x01K\x01K\x02K\x02\x87cnumpy\ndtype\nq\x06U\x02f8K\x00K\x01\x87Rq\x07(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U \x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@U\x04\x00\x00\x00\x00cnumpy.core.multiarray\n_reconstruct\nq\x08h\x03K\x00\x85U\x01b\x87Rq\t(K\x01)h\x07\x89U\x08@\x8c\xb5x\x1d\xaf\x15Dtbtb.')
        self.assertNumpyAll(to_test, actual)

    def test_history_attribute(self):
        ops = self.get_ops(kwds={'output_format': 'nc'})
        ret = ops.execute()
        with nc_scope(ret) as ds:
            history = ds.history
            self.assertTrue('OcgOperations' in history)
            self.assertTrue('ocgis' in history)
            self.assertTrue(len(history) > 500)

    def test_select_nearest(self):
        for spatial_operation in SpatialOperation.iter_possible():
            kwds = {'spatial_operation': spatial_operation, 'geom': [-104.0, 39.0], 'select_nearest': True}
            ops = self.get_ops(kwds=kwds)
            ret = ops.execute()
            self.assertEqual(ret[1]['foo'].variables['foo'].shape, (1, 61, 2, 1, 1))
            self.assertTrue(ret[1]['foo'].spatial.geom.point.value[0, 0].almost_equals(Point(-104.0, 39.0)))

    def test_optimizations_in_calculations(self):
        # pass optimizations to the calculation engine using operations and ensure the output values are equivalent
        rd = RequestDataset(**self.get_dataset())
        field = rd.get()
        tgd = field.temporal.get_grouping(['month'])
        optimizations = {'tgds': {rd.name: tgd}}
        calc = [{'func': 'mean', 'name': 'mean'}]
        calc_grouping = ['month']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                            optimizations=optimizations)
        ret_with_optimizations = ops.execute()
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                            optimizations=None)
        ret_without_optimizations = ops.execute()
        t1 = ret_with_optimizations[1]['foo'].variables['mean']
        t2 = ret_without_optimizations[1]['foo'].variables['mean']
        self.assertNumpyAll(t1.value, t2.value)

    def test_optimizations_in_calculations_bad_calc_grouping(self):
        # bad calculations groupings in the optimizations should be caught and raise a value error
        rd = RequestDataset(**self.get_dataset())
        field = rd.get()
        tgd1 = field.temporal.get_grouping('all')
        tgd2 = field.temporal.get_grouping(['year', 'month'])
        tgd3 = field.temporal.get_grouping([[3]])
        for t in [tgd1, tgd2, tgd3]:
            optimizations = {'tgds': {rd.alias: t}}
            calc = [{'func': 'mean', 'name': 'mean'}]
            calc_grouping = ['month']
            ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                                optimizations=optimizations)
            with self.assertRaises(ValueError):
                ops.execute()

    def test_operations_abstraction_used_for_subsetting(self):
        ret = self.get_ret(kwds={'abstraction': 'point'})
        ref = ret[1]['foo']
        self.assertEqual(ref.spatial.abstraction, 'point')
        self.assertIsInstance(ref.spatial.abstraction_geometry.value[0, 0], Point)
        with self.assertRaises(ValueError):
            ref.get_intersects(Point(-103., 38.))
        sub = ref.get_intersects(Point(-103., 38.).buffer(0.75), use_spatial_index=env.USE_SPATIAL_INDEX)
        self.assertEqual(sub.shape, (1, 61, 2, 1, 1))

    def test_to_csv_shp_and_shape_with_point_subset(self):
        rd = RequestDataset(**self.get_dataset())
        geom = [-103.5, 38.5]
        for of in [constants.OUTPUT_FORMAT_CSV_SHAPEFILE, constants.OUTPUT_FORMAT_SHAPEFILE]:
            ops = ocgis.OcgOperations(dataset=rd, geom=geom, output_format=of, prefix=of)
            ops.execute()

    def test_overwrite_add_auxiliary_files(self):
        # if overwrite is true, we should be able to write the nc output multiple times
        env.OVERWRITE = True
        kwds = {'output_format': constants.OUTPUT_FORMAT_NETCDF, 'add_auxiliary_files': False}
        self.get_ret(kwds=kwds)
        self.get_ret(kwds=kwds)
        # switching the argument to false will result in an IOError
        env.OVERWRITE = False
        with self.assertRaises(IOError):
            self.get_ret(kwds=kwds)

    def test_units_calendar_on_time_bounds(self):
        """Test units and calendar are copied to the time bounds."""

        rd = self.get_dataset()
        ops = ocgis.OcgOperations(dataset=rd, output_format=constants.OUTPUT_FORMAT_NETCDF)
        ret = ops.execute()
        with nc_scope(ret) as ds:
            time_attrs = deepcopy(ds.variables['time'].__dict__)
            time_attrs.pop('bounds')
            time_attrs.pop('axis')
            self.assertEqual(dict(time_attrs), dict(ds.variables['time_bnds'].__dict__))

    def test_units_calendar_on_time_bounds_calculation(self):
        rd = self.get_dataset()
        ops = ocgis.OcgOperations(dataset=rd, output_format='nc', calc=[{'func': 'mean', 'name': 'my_mean'}],
                                  calc_grouping=['month'])
        ret = ops.execute()
        with nc_scope(ret) as ds:
            time = ds.variables['time']
            bound = ds.variables['climatology_bounds']
            self.assertEqual(time.units, bound.units)
            self.assertEqual(bound.calendar, bound.calendar)

    def test_add_auxiliary_files_false_csv_nc(self):
        rd = self.get_dataset()
        for output_format in ['csv', 'nc']:
            dir_output = tempfile.mkdtemp(dir=self.current_dir_output)
            ops = ocgis.OcgOperations(dataset=rd, output_format=output_format, add_auxiliary_files=False,
                                      dir_output=dir_output)
            ret = ops.execute()
            filename = 'ocgis_output.{0}'.format(output_format)
            self.assertEqual(os.listdir(ops.dir_output), [filename])
            self.assertEqual(ret, os.path.join(dir_output, filename))

            # attempting the operation again will work if overwrite is True
            ocgis.env.OVERWRITE = True
            ops = ocgis.OcgOperations(dataset=rd, output_format=output_format, add_auxiliary_files=False,
                                      dir_output=dir_output)
            ops.execute()

    def test_multiple_request_datasets(self):
        aliases = ['foo1', 'foo2', 'foo3', 'foo4']
        rds = []
        for alias in aliases:
            rd = self.get_dataset()
            rd['alias'] = alias
            rds.append(rd)
        ops = ocgis.OcgOperations(dataset=rds, output_format='csv')
        ret = ops.execute()
        with open(ret, 'r') as f:
            reader = csv.DictReader(f)
            test_aliases = set([row['ALIAS'] for row in reader])
        self.assertEqual(set(aliases), test_aliases)

    def test_agg_selection(self):
        features = [
            {'NAME': 'a',
             'wkt': 'POLYGON((-105.020430 40.073118,-105.810753 39.327957,-105.660215 38.831183,-104.907527 38.763441,-104.004301 38.816129,-103.643011 39.802151,-103.643011 39.802151,-103.643011 39.802151,-103.643011 39.802151,-103.959140 40.118280,-103.959140 40.118280,-103.959140 40.118280,-103.959140 40.118280,-104.327957 40.201075,-104.327957 40.201075,-105.020430 40.073118))'},
            {'NAME': 'b',
             'wkt': 'POLYGON((-102.212903 39.004301,-102.905376 38.906452,-103.311828 37.694624,-103.326882 37.295699,-103.898925 37.220430,-103.846237 36.746237,-102.619355 37.107527,-102.634409 37.724731,-101.874194 37.882796,-102.212903 39.004301))'},
            {'NAME': 'c',
             'wkt': 'POLYGON((-105.336559 37.175269,-104.945161 37.303226,-104.726882 37.175269,-104.696774 36.844086,-105.043011 36.693548,-105.283871 36.640860,-105.336559 37.175269))'},
            {'NAME': 'd',
             'wkt': 'POLYGON((-102.318280 39.741935,-103.650538 39.779570,-103.620430 39.448387,-103.349462 39.433333,-103.078495 39.606452,-102.325806 39.613978,-102.325806 39.613978,-102.333333 39.741935,-102.318280 39.741935))'},
        ]

        geom = []
        for feature in features:
            geom.append({'geom': wkt.loads(feature['wkt']), 'properties': {'NAME': feature['NAME']}})
        ops = OcgOperations(dataset=self.get_dataset(), geom=geom, output_format='shp', agg_selection=True)
        ret = ops.execute()
        ugid_path = os.path.join(os.path.split(ret)[0], ops.prefix + '_ugid.shp')
        with fiona.open(ugid_path) as f:
            self.assertEqual(len(f), 1)

        ops = OcgOperations(dataset=self.get_dataset(), geom=geom, agg_selection=True)
        ret = ops.execute()
        self.assertEqual(ret[1]['foo'].spatial.shape, (4, 4))
        self.assertEqual(len(ret), 1)
        self.assertEqual(len(ret.geoms), 1)

        ops = OcgOperations(dataset=self.get_dataset(), geom=geom, output_format='nc',
                            prefix='nc')
        ret = ops.execute()
        with nc_scope(ret) as ds:
            ref = ds.variables['foo']
            self.assertFalse(ref[:].mask.all())

        ops = OcgOperations(dataset=self.get_dataset(), agg_selection=True)
        ops.execute()

    def test_point_subset(self):
        ops = self.get_ops(kwds={'geom': [-103.5, 38.5, ]})
        self.assertEqual(type(ops.geom[0].geom.point.value[0, 0]), Point)
        ret = ops.execute()
        ref = ret[1]['foo']
        self.assertEqual(ref.spatial.grid.shape, (4, 4))

        ops = self.get_ops(kwds={'geom': [-103, 38, ], 'search_radius_mult': 0.01})
        ret = ops.execute()
        ref = ret[1]['foo']
        self.assertEqual(ref.spatial.grid.shape, (1, 1))
        self.assertTrue(ref.spatial.geom.polygon.value[0, 0].intersects(ops.geom[0].geom.point.value[0, 0]))

        ops = self.get_ops(kwds={'geom': [-103, 38, ], 'abstraction': 'point', 'search_radius_mult': 0.01})
        ret = ops.execute()
        ref = ret[1]['foo']
        self.assertEqual(ref.spatial.grid.shape, (1, 1))
        # this is a point abstraction. polygons are not available.
        self.assertIsNone(ref.spatial.geom.polygon)
        # self.assertTrue(ref.spatial.geom.polygon.value[0,0].intersects(ops.geom[0].geom.point.value[0, 0]))

    def test_slicing(self):
        ops = self.get_ops(kwds={'slice': [None, None, 0, [0, 2], [0, 2]]})
        ret = ops.execute()
        ref = ret.gvu(1, 'foo')
        self.assertTrue(np.all(ref.flatten() == 1.0))
        self.assertEqual(ref.shape, (1, 61, 1, 2, 2))

        ops = self.get_ops(kwds={'slice': [0, None, None, [1, 3], [1, 3]]})
        ret = ops.execute()
        ref = ret.gvu(1, 'foo').data
        self.assertTrue(np.all(np.array([1., 2., 3., 4.] == ref[0, 0, 0, :].flatten())))

        # pass only three slices
        with self.assertRaises(DefinitionValidationError):
            self.get_ops(kwds={'slice': [None, [1, 3], [1, 3]]})

    def test_file_only(self):
        ret = self.get_ret(
            kwds={'output_format': 'nc', 'file_only': True, 'calc': [{'func': 'mean', 'name': 'my_mean'}],
                  'calc_grouping': ['month']})
        try:
            ds = nc.Dataset(ret, 'r')
            self.assertTrue(isinstance(ds.variables['my_mean'][:].sum(), np.ma.core.MaskedConstant))
            self.assertEqual(set(ds.variables['my_mean'].ncattrs()),
                             set([u'_FillValue', u'units', u'long_name', u'standard_name', 'grid_mapping']))
        finally:
            ds.close()

        with self.assertRaises(DefinitionValidationError):
            self.get_ret(kwds={'file_only': True, 'output_format': 'shp'})

        with self.assertRaises(DefinitionValidationError):
            self.get_ret(kwds={'file_only': True})

    def test_return_all(self):
        ret = self.get_ret()

        # confirm size of geometry array
        ref = ret[1][self.var].spatial
        shps = [ref.geom, ref.grid, ref.geom.uid, ref.grid.uid]
        for attr in shps:
            self.assertEqual(attr.shape, (4, 4))

        # confirm value array
        ref = ret.gvu(1, self.var)
        self.assertEqual(ref.shape, (1, 61, 2, 4, 4))
        for tidx, lidx in itertools.product(range(0, 61), range(0, 2)):
            slice = ref[0, tidx, lidx, :, :]
            idx = self.base_value == slice
            self.assertTrue(np.all(idx))

    def test_aggregate(self):
        ret = self.get_ret(kwds={'aggregate': True})

        # test area-weighting
        ref = ret.gvu(1, self.var)
        self.assertTrue(np.all(ref.compressed() == np.ma.average(self.base_value)))

        # test geometry reduction
        ref = ret[1][self.var]
        self.assertEqual(ref.spatial.shape, (1, 1))

    def test_time_level_subset(self):
        ret = self.get_ret(time_range=[datetime.datetime(2000, 3, 1),
                                       datetime.datetime(2000, 3, 31, 23)],
                           level_range=[1, 1])
        ref = ret.gvu(1, self.var)
        self.assertEqual(ref.shape, (1, 31, 1, 4, 4))

    def test_time_level_subset_aggregate(self):
        ret = self.get_ret(kwds={'aggregate': True},
                           time_range=[datetime.datetime(2000, 3, 1), datetime.datetime(2000, 3, 31)],
                           level_range=[1, 1], )
        ref = ret.gvu(1, self.var)
        self.assertTrue(np.all(ref.compressed() == np.ma.average(self.base_value)))
        ref = ret[1][self.var]
        self.assertEqual(ref.level.value.shape, (1,))

    def test_time_region_subset(self):
        """Test subsetting a Field object by a time region."""

        rd = ocgis.RequestDataset(uri=os.path.join(env.DIR_OUTPUT, self.fn),
                                  variable=self.var)
        ops = ocgis.OcgOperations(dataset=rd)
        ret = ops.execute()
        all = ret[1]['foo'].temporal.value_datetime

        def get_ref(month, year):
            rd = ocgis.RequestDataset(uri=os.path.join(env.DIR_OUTPUT, self.fn),
                                      variable=self.var,
                                      time_region={'month': month, 'year': year})
            ops = ocgis.OcgOperations(dataset=rd)
            ret = ops.execute()
            ref = ret[1]['foo'].temporal.value_datetime
            months = set([i.month for i in ref.flat])
            years = set([i.year for i in ref.flat])
            if month is not None:
                for m in month:
                    self.assertTrue(m in months)
            if year is not None:
                for m in year:
                    self.assertTrue(m in years)
            return (ref)

        ref = get_ref(None, None)
        self.assertTrue(np.all(ref == all))

        ref = get_ref([3], None)
        self.assertEqual(ref.shape[0], 31)

        ref = get_ref([3, 4], None)
        self.assertTrue(np.all(ref == all))

        ref = get_ref([4], None)
        self.assertEqual(ref.shape[0], 30)

        ref = get_ref(None, [2000])
        self.assertTrue(np.all(ref == all))

        with self.assertRaises(ExtentError):
            get_ref([1], None)

    def test_spatial_aggregate_arbitrary(self):
        poly = Polygon(((-103.5, 39.5), (-102.5, 38.5), (-103.5, 37.5), (-104.5, 38.5)))
        ret2 = self.get_ret(kwds={'output_format': 'numpy', 'geom': poly,
                                  'prefix': 'subset', 'spatial_operation': 'clip', 'aggregate': True})
        self.assertEqual(ret2.gvu(1, self.var).data.mean(), 2.5)

    def test_spatial(self):
        # intersects
        geom = make_poly((37.5, 39.5), (-104.5, -102.5))
        ret = self.get_ret(kwds={'geom': geom})
        ref = ret[1][self.var]
        gids = set([6, 7, 10, 11])
        ret_gids = set(ref.spatial.uid.compressed())
        self.assertEqual(ret_gids, gids)
        to_test = ref.variables[self.var].value[0, 0, 0, :, :]
        self.assertEqual(to_test.shape, (2, 2))
        self.assertTrue(np.all(to_test == np.array([[1.0, 2.0], [3.0, 4.0]])))

        # intersection
        geom = make_poly((38, 39), (-104, -103))
        ret = self.get_ret(kwds={'geom': geom, 'spatial_operation': 'clip'})
        self.assertEqual(len(ret[1][self.var].spatial.uid.compressed()), 4)
        self.assertEqual(ret[1][self.var].variables[self.var].value.shape, (1, 61, 2, 2, 2))
        ref = ret[1][self.var].variables[self.var].value
        self.assertTrue(np.all(ref[0, 0, :, :] == np.array([[1, 2], [3, 4]], dtype=float)))
        # # compare areas to intersects returns
        ref = ret[1][self.var]
        intersection_areas = [g.area for g in ref.spatial.abstraction_geometry.value.flat]
        for ii in intersection_areas:
            self.assertAlmostEqual(ii, 0.25)

        # intersection + aggregation
        geom = make_poly((38, 39), (-104, -103))
        ret = self.get_ret(kwds={'geom': geom, 'spatial_operation': 'clip', 'aggregate': True})
        ref = ret[1][self.var]
        self.assertEqual(len(ref.spatial.uid.flatten()), 1)
        self.assertEqual(ref.spatial.abstraction_geometry.value.flatten()[0].area, 1.0)
        self.assertEqual(ref.variables[self.var].value.flatten().mean(), 2.5)

    def test_empty_intersection(self):
        geom = make_poly((20, 25), (-90, -80))

        with self.assertRaises(exc.ExtentError):
            self.get_ret(kwds={'geom': geom})

        ret = self.get_ret(kwds={'geom': geom, 'allow_empty': True})
        self.assertEqual(ret[1]['foo'], None)

    def test_empty_time_subset(self):
        ds = self.get_dataset(time_range=[datetime.datetime(2900, 1, 1), datetime.datetime(3100, 1, 1)])

        ops = OcgOperations(dataset=ds)
        with self.assertRaises(ExtentError):
            ops.execute()

        ops = OcgOperations(dataset=ds, allow_empty=True)
        ret = ops.execute()
        self.assertEqual(ret[1]['foo'], None)

    def test_snippet(self):
        ret = self.get_ret(kwds={'snippet': True})
        ref = ret.gvu(1, self.var)
        self.assertEqual(ref.shape, (1, 1, 1, 4, 4))
        with nc_scope(os.path.join(self.current_dir_output, self.fn)) as ds:
            to_test = ds.variables['foo'][0, 0, :, :].reshape(1, 1, 1, 4, 4)
            self.assertNumpyAll(to_test, ref.data)

        calc = [{'func': 'mean', 'name': 'my_mean'}]
        group = ['month', 'year']
        with self.assertRaises(DefinitionValidationError):
            self.get_ret(kwds={'calc': calc, 'calc_grouping': group, 'snippet': True})

    def test_snippet_time_region(self):
        with self.assertRaises(DefinitionValidationError):
            rd = self.get_dataset(time_region={'month': [1]})
            OcgOperations(dataset=rd, snippet=True)

    def test_calc(self):
        calc = [{'func': 'mean', 'name': 'my_mean'}]
        group = ['month', 'year']

        # raw
        ret = self.get_ret(kwds={'calc': calc, 'calc_grouping': group})
        ref = ret.gvu(1, 'my_mean')
        self.assertEqual(ref.shape, (1, 2, 2, 4, 4))
        with self.assertRaises(KeyError):
            ret.gvu(1, 'n')

        # aggregated
        for calc_raw in [True, False]:
            ret = self.get_ret(kwds={'calc': calc, 'calc_grouping': group, 'aggregate': True, 'calc_raw': calc_raw})
            ref = ret.gvu(1, 'my_mean')
            self.assertEqual(ref.shape, (1, 2, 2, 1, 1))
            self.assertEqual(ref.flatten().mean(), 2.5)
            self.assertDictEqual(ret[1]['foo'].variables['my_mean'].attrs,
                                 {'long_name': 'Mean', 'standard_name': 'mean'})

    def test_calc_multivariate(self):
        rd1 = self.get_dataset()
        rd1['alias'] = 'var1'
        rd2 = self.get_dataset()
        rd2['alias'] = 'var2'
        calc = [{'name': 'divide', 'func': 'divide', 'kwds': {'arr1': 'var1', 'arr2': 'var2'}}]

        calc_grouping = [None, ['month']]
        for cg in calc_grouping:
            ops = OcgOperations(dataset=[rd1, rd2], calc=calc, calc_grouping=cg)
            ret = ops.execute()
            ref = ret.gvu(1, 'divide').shape
            if cg is None:
                self.assertEqual(ref, (1, 61, 2, 4, 4))
            else:
                self.assertEqual(ref, (1, 2, 2, 4, 4))

    def test_calc_eval(self):
        rd = self.get_dataset()
        calc = 'foo2=foo+4'
        ocgis.env.OVERWRITE = True
        for of in OutputFormat.iter_possible():
            ops = ocgis.OcgOperations(dataset=rd, calc=calc, output_format=of)
            ret = ops.execute()
            if of == 'nc':
                with nc_scope(ret) as ds:
                    self.assertEqual(ds.variables['foo2'][:].mean(), 6.5)

    def test_calc_eval_multivariate(self):
        rd = self.get_dataset()
        rd2 = self.get_dataset()
        rd2['alias'] = 'foo2'
        calc = 'foo3=foo+foo2+4'
        ocgis.env.OVERWRITE = True
        for of in OutputFormat.iter_possible():
            try:
                ops = ocgis.OcgOperations(dataset=[rd, rd2], calc=calc, output_format=of,
                                          slice=[None, [0, 10], None, None, None])
            except DefinitionValidationError:
                # Only one dataset allowed for ESMPy and JSON metadata output.
                self.assertIn(of, [constants.OUTPUT_FORMAT_ESMPY_GRID, constants.OUTPUT_FORMAT_METADATA_JSON])
                continue
            ret = ops.execute()
            if of == 'numpy':
                self.assertIsInstance(ret[1]['foo_foo2'], DerivedMultivariateField)
            if of == 'nc':
                with nc_scope(ret) as ds:
                    self.assertEqual(ds.variables['foo3'][:].mean(), 9.0)

    def test_nc_conversion(self):
        rd = self.get_dataset()
        ops = OcgOperations(dataset=rd, output_format='nc')
        ret = self.get_ret(ops)

        self.assertNcEqual(ret, rd['uri'], ignore_attributes={'global': ['history'],
                                                              'time_bnds': ['calendar', 'units'],
                                                              rd['variable']: ['grid_mapping'],
                                                              'time': ['axis'],
                                                              'level': ['axis'],
                                                              'latitude': ['axis'],
                                                              'longitude': ['axis']},
                           ignore_variables=['latitude_longitude'])

        with self.nc_scope(ret) as ds:
            expected = {'time': 'T', 'level': 'Z', 'latitude': 'Y', 'longitude': 'X'}
            for k, v in expected.iteritems():
                var = ds.variables[k]
                self.assertEqual(var.axis, v)
        with self.nc_scope(ret) as ds:
            self.assertEqual(ds.file_format, constants.NETCDF_DEFAULT_DATA_MODEL)

    def test_nc_conversion_calc(self):
        calc_grouping = ['month']
        calc = [{'func': 'mean', 'name': 'my_mean'},
                {'func': 'std', 'name': 'my_stdev'}]
        kwds = dict(calc_grouping=calc_grouping, calc=calc, output_format='nc')
        ret = self.get_ret(kwds=kwds)

        ds = nc.Dataset(ret)
        try:
            for alias in ['my_mean', 'my_stdev']:
                self.assertEqual(ds.variables[alias].shape, (2, 2, 4, 4))

            # output variable should not have climatology bounds and no time bounds directly
            with self.assertRaises(AttributeError):
                ds.variables['time'].bounds

            self.assertEqual(ds.variables['time'].climatology, 'climatology_bounds')
            self.assertEqual(ds.variables['climatology_bounds'].shape, (2, 2))
        finally:
            ds.close()

    def test_nc_conversion_multiple_request_datasets(self):
        rd1 = self.get_dataset()
        rd2 = self.get_dataset()
        rd2['alias'] = 'foo2'
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=[rd1, rd2], output_format='nc')

    def test_nc_conversion_level_subset(self):
        rd = self.get_dataset(level_range=[1, 1])
        ops = OcgOperations(dataset=rd, output_format='nc', prefix='no_level')
        no_level = ops.execute()

        ops = OcgOperations(dataset={'uri': no_level, 'variable': 'foo'}, output_format='nc', prefix='no_level_again')
        no_level_again = ops.execute()

        self.assertNcEqual(no_level, no_level_again, ignore_attributes={'global': ['history']})

        ds = nc.Dataset(no_level_again)
        try:
            ref = ds.variables['foo'][:]
            self.assertEqual(ref.shape[1], 1)
        finally:
            ds.close()

    def test_nc_conversion_multivariate_calculation(self):
        rd1 = self.get_dataset()
        rd2 = self.get_dataset()
        rd2['alias'] = 'foo2'
        calc = [{'func': 'divide', 'name': 'my_divide', 'kwds': {'arr1': 'foo', 'arr2': 'foo2'}}]
        calc_grouping = ['month']
        ops = OcgOperations(dataset=[rd1, rd2], calc=calc, calc_grouping=calc_grouping,
                            output_format='nc')
        ret = ops.execute()

        ds = nc.Dataset(ret)
        try:
            ref = ds.variables['my_divide'][:]
            self.assertEqual(ref.shape, (2, 2, 4, 4))
            self.assertEqual(np.unique(ref)[0], 1.)
        finally:
            ds.close()

    def test_shp_projection(self):
        output_crs = CoordinateReferenceSystem(epsg=2163)
        ret = self.get_ret(kwds=dict(output_crs=output_crs, output_format='shp'))
        with fiona.open(ret) as f:
            self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), output_crs)

    def test_geojson_projection(self):
        output_crs = CoordinateReferenceSystem(epsg=2163)
        with self.assertRaises(DefinitionValidationError):
            self.get_ret(kwds=dict(output_crs=output_crs, output_format='geojson'))

    def test_limiting_headers(self):
        headers = ['value']
        ops = OcgOperations(dataset=self.get_dataset(), headers=headers, output_format='csv', melted=True)
        ret = ops.execute()
        with open(ret) as f:
            reader = DictReader(f)
            self.assertEqual(reader.fieldnames, [c.upper() for c in constants.HEADERS_REQUIRED] + ['VALUE'])

        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=self.get_dataset(), headers=['foo'], output_format='csv')

    def test_empty_dataset_for_operations(self):
        with self.assertRaises(DefinitionValidationError):
            OcgOperations()

    def test_shp_conversion(self):
        ocgis.env.OVERWRITE = True
        calc = [None, [{'func': 'mean', 'name': 'my_mean'}]]
        group = ['month', 'year']

        keywords = dict(calc=calc, melted=[False, True])

        for k in self.iter_product_keywords(keywords):
            ops = OcgOperations(dataset=self.get_dataset(),
                                output_format='shp',
                                calc_grouping=group,
                                calc=k.calc,
                                melted=k.melted)
            ret = self.get_ret(ops)

            if k.calc is None:
                with fiona.open(ret) as f:
                    target = f.meta['schema']['properties']
                    if k.melted:
                        schema_properties = OrderedDict(
                            [(u'DID', 'int:10'), (u'VID', 'int:10'), (u'UGID', 'int:10'), (u'TID', 'int:10'),
                             (u'LID', 'int:10'), (u'GID', 'int:10'), (u'VARIABLE', 'str:80'), (u'ALIAS', 'str:80'),
                             (u'TIME', 'str:80'), (u'YEAR', 'int:10'), (u'MONTH', 'int:10'), (u'DAY', 'int:10'),
                             (u'LEVEL', 'int:10'), (u'VALUE', 'float:24.15')])
                    else:
                        schema_properties = OrderedDict(
                            [(u'TID', 'int:10'), (u'TIME', 'str:80'), (u'LB_TIME', 'str:80'), (u'UB_TIME', 'str:80'),
                             (u'YEAR', 'int:10'), (u'MONTH', 'int:10'), (u'DAY', 'int:10'), (u'LID', 'int:10'),
                             (u'LEVEL', 'int:10'), (u'LB_LEVEL', 'int:10'), (u'UB_LEVEL', 'int:10'),
                             (u'FOO', 'float:24.15'), (u'UGID', 'int:10')])
                    self.assertAsSetEqual(target.keys(), schema_properties.keys())
                    fiona_meta_actual = {'crs': {'init': u'epsg:4326'},
                                         'driver': u'ESRI Shapefile',
                                         'schema': {'geometry': 'Polygon', 'properties': schema_properties}}
                    self.assertFionaMetaEqual(f.meta, fiona_meta_actual, abs_dtype=False)
                    self.assertEqual(len(f), 1952)
                    if k.melted:
                        record_properties = OrderedDict(
                            [(u'DID', 1), (u'VID', 1), (u'UGID', 1), (u'TID', 11), (u'LID', 2), (u'GID', 5.0),
                             (u'VARIABLE', u'foo'), (u'ALIAS', u'foo'), (u'TIME', '2000-03-11 12:00:00'),
                             (u'YEAR', 2000),
                             (u'MONTH', 3), (u'DAY', 11), (u'LEVEL', 150), (u'VALUE', 1.0)])
                    else:
                        record_properties = OrderedDict(
                            [(u'TID', 11), (u'TIME', u'2000-03-11 12:00:00'), (u'LB_TIME', u'2000-03-11 00:00:00'),
                             (u'UB_TIME', u'2000-03-12 00:00:00'), (u'YEAR', 2000), (u'MONTH', 3), (u'DAY', 11),
                             (u'LID', 2), (u'LEVEL', 150), (u'LB_LEVEL', 100), (u'UB_LEVEL', 200), (u'FOO', 1.0),
                             (u'UGID', 1)])
                    record = list(f)[340]
                    self.assertDictEqual(record['properties'], record_properties)
                    record_coordinates = [
                        [(-105.5, 37.5), (-105.5, 38.5), (-104.5, 38.5), (-104.5, 37.5), (-105.5, 37.5)]]
                    self.assertDictEqual(record, {'geometry': {'type': 'Polygon',
                                                               'coordinates': record_coordinates},
                                                  'type': 'Feature',
                                                  'id': '340',
                                                  'properties': record_properties})
            else:
                with fiona.open(ret) as f:
                    if k.melted:
                        actual = OrderedDict(
                            [(u'DID', 'int:10'), (u'VID', 'int:10'), (u'CID', 'int:10'), (u'UGID', 'int:10'),
                             (u'TID', 'int:10'), (u'LID', 'int:10'), (u'GID', 'int:10'), (u'VARIABLE', 'str:80'),
                             (u'ALIAS', 'str:80'), (u'CALC_KEY', 'str:80'), (u'CALC_ALIAS', 'str:80'),
                             (u'TIME', 'str:80'),
                             (u'YEAR', 'int:10'), (u'MONTH', 'int:10'), (u'DAY', 'int:10'), (u'LEVEL', 'int:10'),
                             (u'VALUE', 'float:24.15')])
                    else:
                        actual = OrderedDict(
                            [(u'TID', 'int:10'), (u'TIME', 'str:80'), (u'LB_TIME', 'str:80'), (u'UB_TIME', 'str:80'),
                             (u'YEAR', 'int:10'), (u'MONTH', 'int:10'), (u'DAY', 'int:10'), (u'LID', 'int:10'),
                             (u'LEVEL', 'int:10'), (u'LB_LEVEL', 'int:10'), (u'UB_LEVEL', 'int:10'),
                             (u'MY_MEAN', 'float:24.15'), (u'UGID', 'int:10')])
                    fiona_meta_actual = {'crs': {'init': u'epsg:4326'},
                                         'driver': u'ESRI Shapefile',
                                         'schema': {'geometry': 'Polygon', 'properties': actual}}
                    self.assertFionaMetaEqual(f.meta, fiona_meta_actual, abs_dtype=False)
                    self.assertEqual(len(f), 64)

    def test_shp_conversion_with_external_geometries(self):

        def _make_record_(wkt_str, ugid, state_name):
            geom = wkt.loads(wkt_str)
            record = {'geometry': mapping(geom),
                      'properties': {'UGID': ugid, 'STATE_NAME': state_name}}
            return record

        nebraska = strings.S5
        kansas = strings.S6
        fiona_crs = crs.WGS84().value
        fiona_properties = {'UGID': 'int', 'STATE_NAME': 'str'}
        fiona_path = os.path.join(self.current_dir_output, 'states.shp')
        fiona_schema = {'geometry': 'Polygon',
                        'properties': fiona_properties}
        with fiona.open(fiona_path, 'w', driver='ESRI Shapefile', crs=fiona_crs, schema=fiona_schema) as f:
            record_nebraska = _make_record_(nebraska, 1, 'Nebraska')
            record_kansas = _make_record_(kansas, 2, 'Kansas')
            f.write(record_nebraska)
            f.write(record_kansas)

        ocgis.env.DIR_GEOMCABINET = self.current_dir_output
        ops = OcgOperations(dataset=self.get_dataset(),
                            geom='states',
                            output_format='shp',
                            melted=True)
        ret = ops.execute()

        output_folder = os.path.join(self.current_dir_output, ops.prefix)
        contents = os.listdir(output_folder)
        self.assertEqual(set(contents),
                         {'ocgis_output_metadata.txt', 'ocgis_output_source_metadata.txt', 'ocgis_output_ugid.shp',
                          'ocgis_output_ugid.dbf', 'ocgis_output_ugid.cpg', 'ocgis_output.dbf', 'ocgis_output.log',
                          'ocgis_output.shx', 'ocgis_output.shp', 'ocgis_output_ugid.shx', 'ocgis_output.cpg',
                          'ocgis_output.prj', 'ocgis_output_ugid.prj', 'ocgis_output_did.csv'})

        with fiona.open(ret) as f:
            rows = list(f)
            fiona_meta = deepcopy(f.meta)
            properties = OrderedDict(
                [(u'DID', 'int:10'), (u'VID', 'int:10'), (u'UGID', 'int:10'), (u'TID', 'int:10'), (u'LID', 'int:10'),
                 (u'GID', 'int:10'), (u'VARIABLE', 'str:80'), (u'ALIAS', 'str:80'), (u'TIME', 'str:80'),
                 (u'YEAR', 'int:10'), (u'MONTH', 'int:10'), (u'DAY', 'int:10'), (u'LEVEL', 'int:10'),
                 (u'VALUE', 'float:24.15')])
            fiona_meta_actual = {'crs': {'init': 'epsg:4326'},
                                 'driver': u'ESRI Shapefile',
                                 'schema': {'geometry': 'Polygon',
                                            'properties': properties}}
            self.assertFionaMetaEqual(fiona_meta, fiona_meta_actual, abs_dtype=False)

        self.assertEqual(len(rows), 610)
        ugids = set([r['properties']['UGID'] for r in rows])
        self.assertEqual(ugids, set([1, 2]))
        properties = OrderedDict(
            [(u'DID', 1), (u'VID', 1), (u'UGID', 2), (u'TID', 26), (u'LID', 1), (u'GID', 16.0), (u'VARIABLE', u'foo'),
             (u'ALIAS', u'foo'), (u'TIME', '2000-03-26 12:00:00'), (u'YEAR', 2000), (u'MONTH', 3), (u'DAY', 26),
             (u'LEVEL', 50), (u'VALUE', 4.0)])
        self.assertDictEqual(properties, rows[325]['properties'])
        coordinates = [[(-102.5, 39.5), (-102.5, 40.5), (-101.5, 40.5), (-101.5, 39.5), (-102.5, 39.5)]]
        self.assertEqual(rows[325], {'geometry': {'type': 'Polygon',
                                                  'coordinates': coordinates},
                                     'type': 'Feature', 'id': '325', 'properties': properties})

        with fiona.open(os.path.join(output_folder, ops.prefix + '_ugid.shp')) as f:
            rows = list(f)
            fiona_meta = deepcopy(f.meta)
            meta_actual = {'crs': {'init': 'epsg:4326'},
                           'driver': u'ESRI Shapefile',
                           'schema': {'geometry': 'Polygon', 'properties': OrderedDict([(u'UGID', 'int:10'),
                                                                                        (u'STATE_NAME', 'str:80')])}}
            self.assertFionaMetaEqual(fiona_meta, meta_actual)
            record1 = {'geometry': {'type': 'Polygon', 'coordinates': strings.S7}, 'type': 'Feature', 'id': '0',
                       'properties': OrderedDict([(u'UGID', 1), (u'STATE_NAME', u'Nebraska')])}
            record2 = {'geometry': {'type': 'Polygon', 'coordinates': strings.S8}, 'type': 'Feature', 'id': '1',
                       'properties': OrderedDict([(u'UGID', 2), (u'STATE_NAME', u'Kansas')])}
            self.assertEqual(rows, [record1, record2])

        # test aggregation
        ops = OcgOperations(dataset=self.get_dataset(),
                            geom='states',
                            output_format='shp',
                            aggregate=True,
                            prefix='aggregation_clip',
                            spatial_operation='clip',
                            melted=True)
        ret = ops.execute()

        with fiona.open(ret) as f:
            self.assertTrue(f.meta['schema']['properties']['GID'].startswith('int'))
            rows = list(f)
        for row in rows:
            self.assertEqual(row['properties']['UGID'], row['properties']['GID'])
        self.assertEqual(set([row['properties']['GID'] for row in rows]), set([1, 2]))
        self.assertEqual(len(rows), 244)
        self.assertEqual(set(os.listdir(os.path.join(self.current_dir_output, ops.prefix))),
                         {'aggregation_clip_ugid.shp', 'aggregation_clip.cpg', 'aggregation_clip_metadata.txt',
                          'aggregation_clip_did.csv', 'aggregation_clip.log', 'aggregation_clip.dbf',
                          'aggregation_clip.shx', 'aggregation_clip_ugid.prj', 'aggregation_clip_ugid.cpg',
                          'aggregation_clip_ugid.shx', 'aggregation_clip.shp', 'aggregation_clip_ugid.dbf',
                          'aggregation_clip.prj', 'aggregation_clip_source_metadata.txt'})

    def test_csv_conversion(self):
        ocgis.env.OVERWRITE = True
        # ops = OcgOperations(dataset=self.get_dataset(), output_format='csv')
        # self.get_ret(ops)

        # test with a geometry to check writing of user-geometry overview shapefile
        geom = make_poly((38, 39), (-104, -103))

        for melted in [True, False]:
            ops = OcgOperations(dataset=self.get_dataset(), output_format='csv', geom=geom, melted=melted)
            ret = ops.execute()

            output_dir = os.path.join(self.current_dir_output, ops.prefix)
            contents = set(os.listdir(output_dir))
            self.assertEqual(contents,
                             {'ocgis_output_source_metadata.txt', 'ocgis_output_metadata.txt', 'ocgis_output.log',
                              'ocgis_output_did.csv', 'ocgis_output.csv'})
            with open(ret, 'r') as f:
                reader = csv.DictReader(f)
                row = reader.next()
                if melted:
                    actual = {'LID': '1', 'UGID': '1', 'VID': '1', 'ALIAS': 'foo', 'DID': '1', 'YEAR': '2000',
                              'VALUE': '1.0',
                              'MONTH': '3', 'VARIABLE': 'foo', 'GID': '6', 'TIME': '2000-03-01 12:00:00', 'TID': '1',
                              'LEVEL': '50', 'DAY': '1'}
                else:
                    actual = {'LID': '1', 'LB_LEVEL': '0', 'LEVEL': '50', 'TIME': '2000-03-01 12:00:00', 'MONTH': '3',
                              'UB_LEVEL': '100', 'LB_TIME': '2000-03-01 00:00:00', 'YEAR': '2000', 'TID': '1',
                              'FOO': '1.0', 'UB_TIME': '2000-03-02 00:00:00', 'DAY': '1', 'UGID': '1'}
                self.assertDictEqual(row, actual)

            did_file = os.path.join(output_dir, ops.prefix + '_did.csv')
            uri = os.path.join(self.current_dir_output, self.fn)
            with open(did_file, 'r') as f:
                reader = csv.DictReader(f)
                row = reader.next()
                self.assertDictEqual(row, {'ALIAS': 'foo', 'DID': '1', 'URI': uri, 'UNITS': 'K',
                                           'STANDARD_NAME': 'Maximum Temperature Foo', 'VARIABLE': 'foo',
                                           'LONG_NAME': 'foo_foo'})

            with open(ret, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        ops = OcgOperations(dataset=self.get_dataset(), output_format='numpy', geom=geom)
        npy = ops.execute()
        self.assertEqual(len(rows), reduce(lambda x, y: x * y, npy.gvu(1, 'foo').shape))

    def test_csv_calc_conversion(self):
        calc = [{'func': 'mean', 'name': 'my_mean'}]
        calc_grouping = ['month', 'year']

        for melted in [True, False]:
            ops = OcgOperations(dataset=self.get_dataset(), output_format='csv', calc=calc, calc_grouping=calc_grouping,
                                melted=melted, prefix=str(melted))
            ret = ops.execute()

            with open(ret, 'r') as f:
                reader = csv.DictReader(f)
                row = reader.next()
                if melted:
                    actual = {'LID': '1', 'UGID': '1', 'VID': '1', 'CID': '1', 'DID': '1', 'YEAR': '2000',
                              'TIME': '2000-03-16 00:00:00', 'CALC_ALIAS': 'my_mean', 'VALUE': '1.0',
                              'MONTH': '3', 'VARIABLE': 'foo', 'ALIAS': 'foo', 'GID': '1', 'CALC_KEY': 'mean',
                              'TID': '1', 'LEVEL': '50', 'DAY': '16'}
                else:
                    actual = {'LID': '1', 'LB_LEVEL': '0', 'LEVEL': '50', 'TIME': '2000-03-16 00:00:00', 'MONTH': '3',
                              'MY_MEAN': '1.0', 'UB_LEVEL': '100', 'LB_TIME': '2000-03-01 00:00:00', 'YEAR': '2000',
                              'TID': '1', 'UB_TIME': '2000-04-01 00:00:00', 'DAY': '16', 'UGID': '1'}
                self.assertDictEqual(row, actual)

    def test_csv_calc_conversion_two_calculations(self):
        calc = [{'func': 'mean', 'name': 'my_mean'}, {'func': 'min', 'name': 'my_min'}]
        calc_grouping = ['month', 'year']
        d1 = self.get_dataset()
        d1['alias'] = 'var1'
        d2 = self.get_dataset()
        d2['alias'] = 'var2'
        ops = OcgOperations(dataset=[d1, d2], output_format='csv', calc=calc, calc_grouping=calc_grouping)
        ret = ops.execute()

        with open(ret, 'r') as f:
            with open(os.path.join(self.path_bin, 'test_csv_calc_conversion_two_calculations.csv')) as f2:
                reader = csv.DictReader(f)
                reader2 = csv.DictReader(f2)
                for row, row2 in zip(reader, reader2):
                    self.assertDictEqual(row, row2)

    def test_calc_multivariate_conversion(self):
        rd1 = self.get_dataset()
        rd1['alias'] = 'var1'
        rd2 = self.get_dataset()
        rd2['alias'] = 'var2'
        calc = [{'name': 'divide', 'func': 'divide', 'kwds': {'arr1': 'var1', 'arr2': 'var2'}}]

        for o in OutputFormat.iter_possible():
            calc_grouping = ['month']

            try:
                ops = OcgOperations(dataset=[rd1, rd2], calc=calc, calc_grouping=calc_grouping, output_format=o,
                                    prefix=o + 'yay')
            except DefinitionValidationError:
                # Only one dataset allowed for ESMPy and JSON metadata output.
                self.assertIn(o, [constants.OUTPUT_FORMAT_ESMPY_GRID, constants.OUTPUT_FORMAT_METADATA_JSON])
                continue

            ret = ops.execute()

            if o in [constants.OUTPUT_FORMAT_CSV, constants.OUTPUT_FORMAT_CSV_SHAPEFILE]:
                with open(ret, 'r') as f:
                    reader = csv.DictReader(f)
                    row = reader.next()
                    self.assertDictEqual(row,
                                         {'LID': '1', 'UGID': '1', 'CID': '1', 'LEVEL': '50', 'DID': '', 'YEAR': '2000',
                                          'TIME': '2000-03-16 00:00:00', 'CALC_ALIAS': 'divide', 'VALUE': '1.0',
                                          'MONTH': '3', 'GID': '1', 'CALC_KEY': 'divide', 'TID': '1', 'DAY': '16'})

            if o == 'nc':
                with nc_scope(ret) as ds:
                    self.assertIn('divide', ds.variables)
                    self.assertTrue(np.all(ds.variables['divide'][:] == 1.))

            if o == 'shp':
                with fiona.open(ret) as f:
                    row = f.next()
                    self.assertIn('CID', row['properties'])

    def test_meta_conversion(self):
        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OUTPUT_FORMAT_METADATA_OCGIS)
        self.get_ret(ops)

    def test_csv_shapefile_conversion(self):
        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OUTPUT_FORMAT_CSV_SHAPEFILE)
        ops.execute()

        geom = make_poly((38, 39), (-104, -103))
        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OUTPUT_FORMAT_CSV_SHAPEFILE, geom=geom,
                            prefix='with_ugid')
        ops.execute()

        path = os.path.join(self.current_dir_output, 'with_ugid')
        contents = os.listdir(path)
        self.assertEqual(set(contents), set(
            ['with_ugid_metadata.txt', 'with_ugid.log', 'with_ugid.csv', 'with_ugid_source_metadata.txt', 'shp',
             'with_ugid_did.csv']))

        shp_path = os.path.join(path, 'shp')
        contents = os.listdir(shp_path)
        self.assertEqual(set(contents), set(
            ['with_ugid_gid.dbf', 'with_ugid_gid.prj', 'with_ugid_ugid.shx', 'with_ugid_gid.shp', 'with_ugid_ugid.prj',
             'with_ugid_ugid.cpg', 'with_ugid_ugid.shp', 'with_ugid_gid.cpg', 'with_ugid_gid.shx',
             'with_ugid_ugid.dbf']))

        gid_path = os.path.join(shp_path, 'with_ugid_gid.shp')
        with fiona.open(gid_path) as f:
            to_test = list(f)
        self.assertEqual(to_test, [{'geometry': {'type': 'Polygon', 'coordinates': [
            [(-104.5, 37.5), (-104.5, 38.5), (-103.5, 38.5), (-103.5, 37.5), (-104.5, 37.5)]]}, 'type': 'Feature',
                                    'id': '0', 'properties': OrderedDict([(u'DID', 1), (u'UGID', 1), (u'GID', 6)])}, {
                                       'geometry': {'type': 'Polygon', 'coordinates': [
                                           [(-103.5, 37.5), (-103.5, 38.5), (-102.5, 38.5), (-102.5, 37.5),
                                            (-103.5, 37.5)]]}, 'type': 'Feature', 'id': '1',
                                       'properties': OrderedDict([(u'DID', 1), (u'UGID', 1), (u'GID', 7)])}, {
                                       'geometry': {'type': 'Polygon', 'coordinates': [
                                           [(-104.5, 38.5), (-104.5, 39.5), (-103.5, 39.5), (-103.5, 38.5),
                                            (-104.5, 38.5)]]}, 'type': 'Feature', 'id': '2',
                                       'properties': OrderedDict([(u'DID', 1), (u'UGID', 1), (u'GID', 10)])}, {
                                       'geometry': {'type': 'Polygon', 'coordinates': [
                                           [(-103.5, 38.5), (-103.5, 39.5), (-102.5, 39.5), (-102.5, 38.5),
                                            (-103.5, 38.5)]]}, 'type': 'Feature', 'id': '3',
                                       'properties': OrderedDict([(u'DID', 1), (u'UGID', 1), (u'GID', 11)])}])

        ugid_path = os.path.join(shp_path, 'with_ugid_ugid.shp')
        with fiona.open(ugid_path) as f:
            to_test = list(f)
            fiona_meta = deepcopy(f.meta)
            fiona_crs = fiona_meta.pop('crs')
            a = CoordinateReferenceSystem(value=fiona_crs)
            b = WGS84()
            self.assertEqual(a, b)

            # Comparison of the whole schema dictionary is not possible. Some installations of Fiona produce different
            # data type lengths (i.e. int:9 v int:10)
            self.assertEqual(fiona_meta['driver'], 'ESRI Shapefile')
            fiona_meta_schema = fiona_meta['schema']
            self.assertEqual(fiona_meta_schema['geometry'], 'Polygon')
            self.assertTrue(fiona_meta_schema['properties']['UGID'].startswith('int'))

        self.assertEqual(to_test, [{'geometry': {'type': 'Polygon', 'coordinates': [
            [(-104.0, 38.0), (-104.0, 39.0), (-103.0, 39.0), (-103.0, 38.0), (-104.0, 38.0)]]}, 'type': 'Feature',
                                    'id': '0', 'properties': OrderedDict([(u'UGID', 1)])}])


@attr('simple')
class TestSimpleNoSpatialBounds(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNcNoSpatialBounds
    fn = 'test_simple_spatial_no_bounds_01.nc'

    def test_interpolate_bounds(self):
        ret = self.get_ops(kwds={'interpolate_spatial_bounds': False}).execute()
        self.assertIsNone(ret[1]['foo'].spatial.geom.polygon)

        ret = self.get_ops(kwds={'interpolate_spatial_bounds': True}).execute()
        polygons = ret[1]['foo'].spatial.geom.polygon.value
        self.assertIsInstance(polygons[0, 0], Polygon)


@attr('simple')
class TestSimpleMask(TestSimpleBase):
    base_value = None
    nc_factory = SimpleMaskNc
    fn = 'test_simple_mask_spatial_01.nc'

    def test_spatial(self):
        self.return_shp = False
        ret = self.get_ret()
        ref = ret[1][self.var].variables[self.var].value.mask
        cmp = np.array([[True, False, False, False],
                        [False, False, False, True],
                        [False, False, False, False],
                        [True, True, False, False]])
        for tidx, lidx in itertools.product(range(0, ref.shape[0]), range(ref.shape[1])):
            self.assertTrue(np.all(cmp == ref[tidx, lidx, :]))

        # aggregation
        ret = self.get_ret(kwds={'aggregate': True})
        ref = ret[1][self.var].variables[self.var]
        self.assertAlmostEqual(ref.value.mean(), 2.58333333333, 5)
        ref = ret[1][self.var]
        self.assertEqual(ref.spatial.uid.shape, (1, 1))

    def test_empty_mask(self):
        geom = make_poly((37.762, 38.222), (-102.281, -101.754))
        with self.assertRaises(exc.MaskedDataError):
            self.get_ret(kwds={'geom': geom, 'output_format': 'shp'})
        self.get_ret(kwds={'geom': geom, 'allow_empty': True})
        with self.assertRaises(exc.MaskedDataError):
            self.get_ret(kwds={'geom': geom, 'output_format': 'numpy'})
        ret = self.get_ret(kwds={'geom': geom, 'output_format': 'numpy', 'allow_empty': True})
        self.assertTrue(ret[1]['foo'].variables['foo'].value.mask.all())


@attr('simple')
class TestSimpleMultivariate(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNcMultivariate
    fn = 'test_simple_multivariate_01.nc'
    var = ['foo', 'foo2']

    def get_request_dataset(self, **kwargs):
        ds = self.get_dataset(**kwargs)
        return RequestDataset(**ds)

    def get_multiple_request_datasets(self):
        rd1 = self.get_request_dataset()
        rd1.name = 'rd1'
        rd1.alias = ['f1', 'f2']
        rd2 = self.get_request_dataset()
        rd2.name = 'rd2'
        rd2.alias = ['ff1', 'ff2']
        return [rd1, rd2]

    def run_field_tst(self, field):
        self.assertEqual([v.name for v in field.variables.itervalues()], ['foo', 'foo2'])
        self.assertEqual(field.variables.values()[0].value.mean(), 2.5)
        self.assertEqual(field.variables.values()[1].value.mean(), 5.5)
        sub = field[:, 3, 1, 1:3, 1:3]
        self.assertNumpyAll(sub.variables.values()[0].value.data.flatten(), np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertNumpyAll(sub.variables.values()[1].value.data.flatten(), np.array([1.0, 2.0, 3.0, 4.0]) + 3)

    def test_field(self):
        rd = self.get_request_dataset()
        self.assertEqual(rd.name, 'foo_foo2')
        self.run_field_tst(rd.get())

    def test_operations_convert_numpy(self):
        name = [None, 'custom_name']
        for n in name:
            n = n or '_'.join(self.var)
            rd = self.get_request_dataset()
            rd.name = n
            ops = OcgOperations(dataset=rd, output_format='numpy')
            ret = ops.execute()
            self.run_field_tst(ret[1][rd.name])

    def test_operations_convert_numpy_multiple_request_datasets(self):
        rds = self.get_multiple_request_datasets()
        ops = OcgOperations(dataset=rds, output_format='numpy')
        ret = ops.execute()
        self.assertEqual(set(ret[1].keys()), set(['rd1', 'rd2']))
        self.assertEqual(ret[1]['rd1'].variables.keys(), ['f1', 'f2'])
        self.assertEqual(ret[1]['rd2'].variables.keys(), ['ff1', 'ff2'])
        for row in ret.get_iter_melted():
            self.run_field_tst(row['field'])

    def test_operations_convert_nc_one_request_dataset(self):
        rd = self.get_request_dataset()
        ops = OcgOperations(dataset=rd, output_format='nc')
        ret = ops.execute()
        out_folder = os.path.split(ret)[0]
        csv_path = os.path.join(out_folder, 'ocgis_output_did.csv')
        with nc_scope(ret) as ds:
            self.assertEqual(ds.variables['foo'][:].mean(), 2.5)
            self.assertEqual(ds.variables['foo2'][:].mean(), 5.5)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            lines = list(reader)
        for row in lines:
            self.assertEqual(row.pop('URI'), os.path.join(self.current_dir_output, self.fn))
        actual = [
            {'ALIAS': 'foo', 'DID': '1', 'UNITS': 'K', 'STANDARD_NAME': 'Maximum Temperature Foo', 'VARIABLE': 'foo',
             'LONG_NAME': 'foo_foo'},
            {'ALIAS': 'foo2', 'DID': '1', 'UNITS': 'mm/s', 'STANDARD_NAME': 'Precipitation Foo', 'VARIABLE': 'foo2',
             'LONG_NAME': 'foo_foo_pr'}]
        for a, l in zip(actual, lines):
            self.assertDictEqual(a, l)

    def test_operations_convert_multiple_request_datasets(self):
        for o in OutputFormat.iter_possible():
            if o in [constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH, constants.OUTPUT_FORMAT_NETCDF]:
                continue
            rds = self.get_multiple_request_datasets()
            try:
                ops = OcgOperations(dataset=rds, output_format=o, prefix=o, slice=[None, [0, 2], None, None, None])
            except DefinitionValidationError:
                # Only one dataset allowed for ESMPy and JSON metadata output.
                self.assertIn(o, [constants.OUTPUT_FORMAT_ESMPY_GRID, constants.OUTPUT_FORMAT_METADATA_JSON])
                continue
            ret = ops.execute()
            path_source_metadata = os.path.join(self.current_dir_output, ops.prefix,
                                                '{0}_source_metadata.txt'.format(ops.prefix))
            if o not in [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_METADATA_OCGIS]:
                self.assertTrue(os.path.exists(ret))
                with open(path_source_metadata, 'r') as f:
                    lines = f.readlines()
                search = re.search('URI = (.*)\n', lines[0]).groups()[0]
                self.assertTrue(os.path.exists(search))

    def test_calculation_multiple_request_datasets(self):
        rds = self.get_multiple_request_datasets()
        calc = [{'func': 'mean', 'name': 'mean'}]
        calc_grouping = ['month']
        ops = OcgOperations(dataset=rds, calc=calc, calc_grouping=calc_grouping)
        ret = ops.execute()
        self.assertEqual(set(ret[1].keys()), set(['rd1', 'rd2']))
        for row in ret.get_iter_melted():
            self.assertEqual(row['variable'].name, 'mean')
            self.assertEqual(row['field'].shape, (1, 2, 2, 4, 4))
            try:
                self.assertEqual(row['variable'].value.mean(), 2.5)
            except AssertionError:
                if row['variable'].parents.values()[0].name == 'foo2':
                    self.assertEqual(row['variable'].value.mean(), 5.5)
                else:
                    raise


@attr('simple')
class TestSimple360(TestSimpleBase):
    fn = 'test_simple_360_01.nc'
    nc_factory = SimpleNc360

    def test_select_geometries_wrapped(self):
        """
        Selection geometries should be wrapped appropriately when used to subset a 0-360 dataset when vector wrap is
        False.
        """

        def _get_is_wrapped_(bounds):
            return np.any(np.array(bounds) < 0)

        geom = wkt.loads(
            'POLYGON((267.404839 39.629032,268.767204 39.884946,269.775806 39.854839,269.926344 39.305376,269.753226 38.665591,269.647849 38.319355,269.128495 37.980645,268.654301 37.717204,267.954301 37.626882,267.961828 37.197849,267.269355 37.769892,266.968280 38.959140,266.968280 38.959140,266.968280 38.959140,267.404839 39.629032))')
        keywords = dict(vector_wrap=[False, True],
                        output_format=[constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_SHAPEFILE])

        for ctr, k in enumerate(itr_products_keywords(keywords, as_namedtuple=True)):
            dct = self.get_dataset()
            rd = RequestDataset(**dct)
            ops = OcgOperations(dataset=rd, vector_wrap=k.vector_wrap, output_format=k.output_format, geom=geom,
                                prefix=str(ctr))
            ret = ops.execute()
            if k.output_format == constants.OUTPUT_FORMAT_SHAPEFILE:
                path_ugid = ret.replace('.', '_ugid.')
                rows = list(GeomCabinetIterator(path=path_ugid))
                bounds = rows[0]['geom'].bounds
            else:
                bounds = ret.geoms[1].bounds
            if k.vector_wrap:
                self.assertTrue(_get_is_wrapped_(bounds))
            else:
                self.assertFalse(_get_is_wrapped_(bounds))

    def test_vector_wrap_in_operations(self):
        """Test output is appropriately wrapped."""

        rd = RequestDataset(**self.get_dataset())
        field = rd.get()
        self.assertEqual(field.spatial.wrapped_state, WrappableCoordinateReferenceSystem._flag_unwrapped)
        ops = OcgOperations(dataset=rd, vector_wrap=True)
        ret = ops.execute()
        self.assertEqual(ret[1]['foo'].spatial.wrapped_state, WrappableCoordinateReferenceSystem._flag_wrapped)

    def test_wrap(self):

        def _get_longs_(geom):
            ret = np.array([g.centroid.x for g in geom.flat])
            return (ret)

        ret = self.get_ret(kwds={'vector_wrap': False})
        longs_unwrap = _get_longs_(ret[1][self.var].spatial.abstraction_geometry.value)
        self.assertTrue(np.all(longs_unwrap > 180))

        ret = self.get_ret(kwds={'vector_wrap': True})
        longs_wrap = _get_longs_(ret[1][self.var].spatial.abstraction_geometry.value)
        self.assertTrue(np.all(np.array(longs_wrap) < 180))

        self.assertTrue(np.all(longs_unwrap - 360 == longs_wrap))

    def test_spatial_touch_only(self):
        geom = [make_poly((38.2, 39.3), (-93, -92)), make_poly((38, 39), (-93.1, -92.1))]

        for abstraction, g in itertools.product(['polygon', 'point'], geom):
            try:
                ops = self.get_ops(kwds={'geom': g, 'abstraction': abstraction})
                ret = ops.execute()
                self.assertEqual(len(ret[1][self.var].spatial.uid.compressed()), 4)
                self.get_ret(kwds={'vector_wrap': False})
                ret = self.get_ret(kwds={'geom': g, 'vector_wrap': False, 'abstraction': abstraction})
                self.assertEqual(len(ret[1][self.var].spatial.uid.compressed()), 4)
            except ExtentError:
                if abstraction == 'point':
                    pass
                else:
                    raise

    def test_spatial(self):
        geom = make_poly((38.1, 39.1), (-93.1, -92.1))

        for abstraction in ['polygon', 'point']:
            n = 1 if abstraction == 'point' else 4
            ops = self.get_ops(kwds={'geom': geom, 'abstraction': abstraction})
            ret = ops.execute()
            self.assertEqual(len(ret[1][self.var].spatial.uid.compressed()), n)
            self.get_ret(kwds={'vector_wrap': False})
            ret = self.get_ret(kwds={'geom': geom, 'vector_wrap': False, 'abstraction': abstraction})
            self.assertEqual(len(ret[1][self.var].spatial.uid.compressed()), n)


@attr('simple')
class TestSimpleProjected(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNcProjection
    fn = 'test_simple_spatial_projected_01.nc'

    def test_differing_projection_no_output_crs(self):
        nc_normal = SimpleNc()
        nc_normal.write()
        uri = os.path.join(self.current_dir_output, nc_normal.filename)

        rd_projected = self.get_dataset()
        rd_projected['alias'] = 'projected'
        rd_normal = {'uri': uri, 'variable': 'foo', 'alias': 'normal'}
        dataset = [rd_projected, rd_normal]

        output_format = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_SHAPEFILE,
                         constants.OUTPUT_FORMAT_NETCDF, constants.OUTPUT_FORMAT_CSV_SHAPEFILE]
        for o in output_format:
            try:
                OcgOperations(dataset=dataset, output_format=o)
            except DefinitionValidationError:
                if o != constants.OUTPUT_FORMAT_NUMPY:
                    pass

    def test_differing_projection_with_output_crs(self):
        nc_normal = SimpleNc()
        nc_normal.write()
        uri = os.path.join(self.current_dir_output, nc_normal.filename)

        rd_projected = self.get_dataset()
        rd_projected['alias'] = 'projected'
        rd_normal = {'uri': uri, 'variable': 'foo', 'alias': 'normal'}
        dataset = [rd_projected, rd_normal]

        output_format = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_SHAPEFILE,
                         constants.OUTPUT_FORMAT_NETCDF, constants.OUTPUT_FORMAT_CSV_SHAPEFILE]

        for o in output_format:
            try:
                ops = OcgOperations(dataset=dataset, output_format=o, output_crs=CFWGS84(), prefix=o, melted=True)
                self.assertTrue(ops.melted)
                ret = ops.execute()

                if o == constants.OUTPUT_FORMAT_NUMPY:
                    uids = []
                    for field in ret[1].itervalues():
                        uids.append(field.uid)
                        self.assertIsInstance(field.spatial.crs, CFWGS84)
                    self.assertEqual(set(uids), set([1, 2]))
                if o == constants.OUTPUT_FORMAT_SHAPEFILE:
                    with fiona.open(ret) as f:
                        self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), CFWGS84())
                        aliases = set([row['properties']['ALIAS'] for row in f])
                    self.assertEqual(set(['projected', 'normal']), aliases)
                if o == constants.OUTPUT_FORMAT_CSV_SHAPEFILE:
                    with open(ret, 'r') as f:
                        reader = csv.DictReader(f)
                        collect = {'dids': [], 'aliases': []}
                        for row in reader:
                            collect['dids'].append(int(row['DID']))
                            collect['aliases'].append(row['ALIAS'])
                        self.assertEqual(set(['projected', 'normal']), set(collect['aliases']))
                        self.assertEqual(set([1, 2]), set(collect['dids']), msg='did missing in csv file')

                    gid_shp = os.path.join(ops.dir_output, ops.prefix, 'shp', ops.prefix + '_gid.shp')
                    with fiona.open(gid_shp) as f:
                        dids = set([row['properties']['DID'] for row in f])
                        self.assertEqual(dids, set([1, 2]), msg='did missing in overview file')

            except DefinitionValidationError:
                if o == constants.OUTPUT_FORMAT_NETCDF:
                    pass
                else:
                    raise

    def test_nc_projection(self):
        dataset = self.get_dataset()
        ret = self.get_ret(kwds={'output_format': 'nc'})
        self.assertNcEqual(dataset['uri'], ret,
                           ignore_attributes={'global': ['history'], 'time_bnds': ['calendar', 'units'],
                                              'crs': ['units']})

    def test_nc_projection_to_shp(self):
        ret = self.get_ret(kwds={'output_format': constants.OUTPUT_FORMAT_SHAPEFILE})
        with fiona.open(ret) as f:
            self.assertEqual(f.meta['crs']['proj'], 'lcc')

    def test_with_geometry(self):
        self.get_ret(kwds={'output_format': constants.OUTPUT_FORMAT_SHAPEFILE, 'prefix': 'as_polygon'})

        features = [
            {'NAME': 'a',
             'wkt': 'POLYGON((-425985.928175 -542933.565515,-425982.789465 -542933.633257,-425982.872261 -542933.881644,-425985.837852 -542933.934332,-425985.837852 -542933.934332,-425985.928175 -542933.565515))'},
            {'NAME': 'b',
             'wkt': 'POLYGON((-425982.548605 -542936.839709,-425982.315272 -542936.854762,-425982.322799 -542936.937558,-425982.526024 -542936.937558,-425982.548605 -542936.839709))'},
        ]

        from_crs = RequestDataset(**self.get_dataset()).get().spatial.crs
        to_sr = CoordinateReferenceSystem(epsg=4326).sr
        for feature in features:
            geom = wkt.loads(feature['wkt'])
            geom = project_shapely_geometry(geom, from_crs.sr, to_sr)
            feature['wkt'] = geom.wkt

        path = os.path.join(self.current_dir_output, 'ab_{0}.shp'.format('polygon'))
        with FionaMaker(path, geometry='Polygon') as fm:
            fm.write(features)
        ocgis.env.DIR_GEOMCABINET = self.current_dir_output

        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OUTPUT_FORMAT_SHAPEFILE,
                            geom='ab_polygon')
        ret = ops.execute()
        ugid_shp = os.path.join(os.path.split(ret)[0], ops.prefix + '_ugid.shp')

        with fiona.open(ugid_shp) as f:
            self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), from_crs)

        ops = OcgOperations(dataset=self.get_dataset(), output_format=constants.OUTPUT_FORMAT_SHAPEFILE,
                            geom='ab_polygon', output_crs=CFWGS84(), prefix='xx')
        ret = ops.execute()
        ugid_shp = os.path.join(os.path.split(ret)[0], ops.prefix + '_ugid.shp')

        with fiona.open(ugid_shp) as f:
            self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), WGS84())
        with fiona.open(ret) as f:
            self.assertEqual(CoordinateReferenceSystem(value=f.meta['crs']), WGS84())
